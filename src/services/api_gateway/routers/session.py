"""
Session Router - Session Negotiation and Management
Creates sessions and negotiates optimal transport configuration
"""

import logging
import os
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, Optional, List
from pydantic import ValidationError, BaseModel, Field
from datetime import datetime
from enum import Enum
from src.core.exceptions import UltravoxError, wrap_exception

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)

# Global Communication Manager (set by API Gateway Service)
_comm_manager = None


def set_comm_manager(comm_manager):
    """Set the Communication Manager instance for this router"""
    global _comm_manager
    _comm_manager = comm_manager


class TransportType(str, Enum):
    """Available transport types"""
    WEBRTC = "webrtc"
    SOCKETIO = "socketio"
    REST = "rest"


class TransportConfig(BaseModel):
    """Transport connection configuration"""
    type: TransportType
    config: Dict[str, Any]


class SessionNegotiateRequest(BaseModel):
    """Request to create session and negotiate transport"""
    scenario_id: str = Field(..., description="Scenario ID to use for this session")
    user_id: Optional[str] = Field(None, description="User identifier (optional)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    supported_transports: List[TransportType] = Field(
        default=[TransportType.WEBRTC, TransportType.SOCKETIO, TransportType.REST],
        description="Client-supported transport types in preference order"
    )


class ScenarioInfo(BaseModel):
    """Scenario information included in session response"""
    id: str
    name: str
    type: str
    language: str
    voice_id: str
    system_prompt: str
    user_role: str
    ai_role: str


class SessionNegotiateResponse(BaseModel):
    """Response with session info and transport configuration"""
    session_id: str
    scenario_id: str
    scenario: ScenarioInfo
    transport: TransportConfig
    expires_at: str
    expires_in: int
    created_at: str


class SessionInfoResponse(BaseModel):
    """Session information response"""
    id: str
    scenario_id: str
    user_id: Optional[str]
    transport_type: str
    transport_config: Dict[str, Any]
    active: bool
    created_at: str
    last_activity: str
    ttl_seconds: int


def determine_transport(
    supported_transports: List[TransportType],
    server_health: Dict[str, Any]
) -> TransportType:
    """
    Determine best transport based on:
    - Client preferences
    - Server availability
    - Current load
    """
    # For now, simple logic: use first supported transport
    # TODO: Add load balancing, health checks

    # Check server health for each transport
    transport_priority = {
        TransportType.WEBRTC: 1,
        TransportType.SOCKETIO: 2,
        TransportType.REST: 3
    }

    # Sort by priority and client support
    available = [t for t in supported_transports if transport_priority.get(t)]
    available.sort(key=lambda t: transport_priority[t])

    if not available:
        return TransportType.REST  # Fallback

    return available[0]


def get_transport_config(
    transport_type: TransportType,
    session_id: str
) -> Dict[str, Any]:
    """
    Get transport-specific configuration
    """
    # TODO: Load from configuration file or service registry

    if transport_type == TransportType.WEBRTC:
        webrtc_signaling_url = os.getenv('WEBRTC_SIGNALING_URL', 'ws://localhost:8090')
        webrtc_gateway_url = os.getenv('WEBRTC_GATEWAY_URL', 'http://localhost:8010')
        return {
            "signaling": f"{webrtc_signaling_url}/ws/{session_id}",
            "gateway": webrtc_gateway_url
        }
    elif transport_type == TransportType.SOCKETIO:
        socketio_url = os.getenv('SOCKETIO_SERVICE_URL', 'http://localhost:8020')
        return {
            "url": socketio_url,
            "path": "/socket.io"
        }
    elif transport_type == TransportType.REST:
        rest_polling_url = os.getenv('REST_POLLING_SERVICE_URL', 'http://localhost:8106')
        return {
            "url": f"{rest_polling_url}/api",
            "poll_interval": 1000
        }
    else:
        raise ValueError(f"Unknown transport type: {transport_type}")


@router.post("/negotiate", response_model=SessionNegotiateResponse, status_code=status.HTTP_201_CREATED)
async def negotiate_session(request: SessionNegotiateRequest):
    """
    Create session and negotiate transport

    Flow:
    1. Validate scenario exists
    2. Create session in Session Service
    3. Determine optimal transport
    4. Return session + transport config
    """
    try:
        # Use global Communication Manager (set by API Gateway Service)
        if _comm_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Communication Manager not initialized"
            )

        # 1. Fetch scenario details (REQUIRED - must exist)
        logger.info(f"🔍 Validating scenario: {request.scenario_id}")

        # Direct HTTP call to scenarios service (bypassing Communication Manager to avoid prefix duplication)
        import httpx

        scenarios_base_url = os.getenv('SCENARIOS_SERVICE_URL', 'http://localhost:8700')
        scenario_data = None

        try:
            # Try by ID (UUID)
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{scenarios_base_url}/{request.scenario_id}")
                if response.status_code == 200:
                    scenario_data = response.json()
                    logger.info(f"✅ Found scenario by ID: {scenario_data.get('name')}")
        except Exception as id_error:
            logger.debug(f"Failed to fetch scenario by ID, trying by name: {id_error}")

        # If not found by ID, try to find by name
        if not scenario_data:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{scenarios_base_url}/")
                    if response.status_code == 200:
                        scenarios_list = response.json()
                        # Find scenario by name
                        for scenario in scenarios_list.get("scenarios", []):
                            if scenario.get("name") == request.scenario_id:
                                scenario_data = scenario
                                logger.info(f"✅ Found scenario by name: {scenario_data.get('id')}")
                                break
            except Exception as name_error:
                logger.error(f"❌ Failed to fetch scenarios list: {name_error}")

        if not scenario_data:
            logger.error(f"❌ Scenario '{request.scenario_id}' not found by ID or name")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario '{request.scenario_id}' not found. Please create the scenario first or use an existing scenario ID."
            )

        # Validate scenario data is not empty
        if not scenario_data or not scenario_data.get("id"):
            logger.error(f"❌ Invalid scenario data received for {request.scenario_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid scenario data for '{request.scenario_id}'"
            )

        logger.info(f"✅ Scenario validated: {scenario_data.get('name')} (type: {scenario_data.get('type')})")

        # 2. Determine optimal transport
        # TODO: Get actual server health metrics
        server_health = {}
        transport_type = determine_transport(request.supported_transports, server_health)

        logger.info(f"🎯 Selected transport: {transport_type} for session")

        # 3. Create session in Session Service (with validated scenario_id)
        logger.info(f"📝 Creating session for scenario: {request.scenario_id}")

        # Communication Manager returns response directly or raises exception
        session_data = await comm_manager.call_service(
            service_name="session",
            endpoint_path="/api/sessions",
            method="POST",
            json_data={
                "scenario_id": request.scenario_id,  # Already validated above
                "user_id": request.user_id,
                "metadata": {
                    **request.metadata,
                    "transport_type": transport_type.value,
                    "scenario_name": scenario_data.get("name"),
                    "scenario_type": scenario_data.get("type"),
                    "created_via": "api_gateway_negotiate"
                }
            }
        )

        session_id = session_data.get("id")

        if not session_id:
            logger.error("❌ Session created but no ID returned")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Session created but no ID returned from Session Service"
            )

        logger.info(f"✅ Session created: {session_id}")

        # 4. Get transport configuration
        transport_config = get_transport_config(transport_type, session_id)

        # 5. Build response
        response = SessionNegotiateResponse(
            session_id=session_id,
            scenario_id=request.scenario_id,
            scenario=ScenarioInfo(
                id=scenario_data.get("id"),
                name=scenario_data.get("name"),
                type=scenario_data.get("type"),
                language=scenario_data.get("language"),
                voice_id=scenario_data.get("voice_id"),
                system_prompt=scenario_data.get("system_prompt"),
                user_role=scenario_data.get("user_role"),
                ai_role=scenario_data.get("ai_role")
            ),
            transport=TransportConfig(
                type=transport_type,
                config=transport_config
            ),
            expires_at=session_data.get("created_at"),  # TODO: Calculate expiry
            expires_in=session_data.get("ttl_seconds", 3600),
            created_at=session_data.get("created_at")
        )

        logger.info(f"✅ Session negotiated: {session_id} with {transport_type} transport")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to negotiate session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    """
    Get session information
    Proxies to Session Service
    """
    try:
        # Use global Communication Manager (set by API Gateway Service)
        if _comm_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Communication Manager not initialized"
            )

        result = await _comm_manager.call_service(
            service_name="session",
            endpoint_path=f"/api/sessions/{session_id}",
            method="GET"
        )

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            if "not found" in error.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {session_id} not found"
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch session: {error}"
            )

        session_data = result.get("data", {})

        # Map session data to response
        return SessionInfoResponse(
            id=session_data.get("id"),
            scenario_id=session_data.get("scenario_id"),
            user_id=session_data.get("user_id"),
            transport_type=session_data.get("metadata", {}).get("transport_type", "unknown"),
            transport_config={},  # TODO: Store in session metadata
            active=True,
            created_at=session_data.get("created_at"),
            last_activity=session_data.get("last_activity"),
            ttl_seconds=session_data.get("ttl_seconds", 3600)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    End/delete session
    Proxies to Session Service
    """
    try:
        # Use global Communication Manager (set by API Gateway Service)
        if _comm_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Communication Manager not initialized"
            )

        result = await _comm_manager.call_service(
            service_name="session",
            endpoint_path=f"/api/sessions/{session_id}",
            method="DELETE"
        )

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            if "not found" in error.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {session_id} not found"
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete session: {error}"
            )

        logger.info(f"✅ Session deleted: {session_id}")
        return {"success": True, "message": f"Session {session_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
