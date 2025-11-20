"""
Scenarios Router - Proxy to Scenarios Service
Provides unified access to scenario management through API Gateway
"""

import logging
from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
from pydantic import ValidationError, BaseModel, Field
from datetime import datetime
from src.core.exceptions import UltravoxError, wrap_exception

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])
logger = logging.getLogger(__name__)

# Global Communication Manager (set by API Gateway Service)
_comm_manager = None


def set_comm_manager(comm_manager):
    """Set the Communication Manager instance for this router"""
    global _comm_manager
    _comm_manager = comm_manager


# Models (mirror from Scenarios Service)
class ScenarioCreate(BaseModel):
    """Request to create a new scenario"""
    name: str = Field(..., min_length=1, max_length=100, description="Scenario name")
    type: str = Field(..., description="Scenario type")
    system_prompt: str = Field(..., min_length=10, max_length=5000, description="System prompt for LLM")
    user_role: str = Field(..., description="Description of user's role")
    ai_role: str = Field(..., description="Description of AI assistant's role")
    language: str = Field(default="pt-BR", description="Primary language code")
    voice_id: Optional[str] = Field(default=None, description="TTS voice identifier")
    is_template: bool = Field(default=False, description="Mark as template scenario")


class ScenarioResponse(BaseModel):
    """Scenario response with metadata"""
    id: str
    name: str
    type: str
    system_prompt: str
    user_role: str
    ai_role: str
    language: str
    voice_id: str
    is_template: bool
    created_at: str
    updated_at: str


class ScenarioListResponse(BaseModel):
    """List of scenarios"""
    scenarios: List[ScenarioResponse]
    total: int
    templates_count: int
    custom_count: int


@router.get("", response_model=ScenarioListResponse)
async def list_scenarios():
    """
    List all available scenarios
    Proxies to Scenarios Service via Communication Manager
    """
    try:
        # Use global Communication Manager (set by API Gateway Service)
        if _comm_manager is None:
            logger.error("❌ Communication Manager is None!")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Communication Manager not initialized"
            )

        logger.info(f"🔍 Calling scenarios service via Communication Manager (type: {type(_comm_manager).__name__})")

        # Call scenarios service
        # Communication Manager returns the response directly (not wrapped in success/data)
        result = await _comm_manager.call_service(
            service_name="scenarios",
            endpoint_path="/",
            method="GET"
        )

        logger.info(f"✅ Got result from scenarios service: {type(result)}")
        return result

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        import traceback
        logger.error(f"❌ Failed to list scenarios: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("", response_model=ScenarioResponse, status_code=status.HTTP_201_CREATED)
async def create_scenario(request: ScenarioCreate):
    """
    Create a new scenario
    Proxies to Scenarios Service via Communication Manager
    """
    try:
        # Use global Communication Manager (set by API Gateway Service)
        if _comm_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Communication Manager not initialized"
            )

        # Call scenarios service
        result = await _comm_manager.call_service(
            service_name="scenarios",
            endpoint_path="/",
            method="POST",
            json_data=request.model_dump()
        )

        logger.info(f"✅ Created scenario via API Gateway: {result.get('id')}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create scenario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(scenario_id: str):
    """
    Get scenario by ID
    Proxies to Scenarios Service via Communication Manager
    """
    try:
        # Use global Communication Manager (set by API Gateway Service)
        if _comm_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Communication Manager not initialized"
            )

        # Call scenarios service
        # Communication Manager returns the response directly or raises exception
        result = await _comm_manager.call_service(
            service_name="scenarios",
            endpoint_path=f"/{scenario_id}",
            method="GET"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get scenario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
