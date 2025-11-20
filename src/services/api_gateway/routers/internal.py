"""
Internal API endpoints for API Gateway
These endpoints are used for inter-service communication and system management
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError, BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)

# ============================================================================
# Models
# ============================================================================

class PortUpdateRequest(BaseModel):
    """Request model for port change notification"""
    service: str = Field(..., description="Service name (e.g., 'orchestrator')")
    new_port: int = Field(..., description="New port number", ge=1024, le=65535)
    event: str = Field(..., description="Event type (e.g., 'port_changed')")
    timestamp: float = Field(..., description="Unix timestamp of the event")

    model_config = {
        "json_schema_extra": {
            "example": {
                "service": "orchestrator",
                "new_port": 9051,
                "event": "port_changed",
                "timestamp": 1234567890.123
            }
        }
    }


class PortUpdateResponse(BaseModel):
    """Response model for port update"""
    status: str
    service: str
    old_port: Optional[int] = None
    new_port: int
    updated_at: str
    message: str


# ============================================================================
# Global State - Service Registry Cache
# ============================================================================

# In-memory cache of service ports
# This is updated when port changes are notified
_service_ports: Dict[str, int] = {}


def get_service_port(service_name: str) -> Optional[int]:
    """
    Get cached port for a service

    Args:
        service_name: Service name

    Returns:
        Port number or None if not cached
    """
    return _service_ports.get(service_name)


def update_service_port(service_name: str, port: int) -> Optional[int]:
    """
    Update cached port for a service

    Args:
        service_name: Service name
        port: New port number

    Returns:
        Previous port number or None if not cached
    """
    old_port = _service_ports.get(service_name)
    _service_ports[service_name] = port
    logger.info(f"📝 Service port cache updated: {service_name} → {port} (was: {old_port})")
    return old_port


def get_all_service_ports() -> Dict[str, int]:
    """
    Get all cached service ports

    Returns:
        Dictionary of service_name → port
    """
    return _service_ports.copy()


# ============================================================================
# Router
# ============================================================================

router = APIRouter(
    prefix="/internal",
    tags=["internal"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/routes/update",
    response_model=PortUpdateResponse,
    status_code=status.HTTP_200_OK,
    summary="Update service route after port change",
    description="""
    This endpoint is called by the Service Launcher when a service's port changes
    (e.g., during auto-recovery). The API Gateway updates its internal routing cache.

    **Phase 3 Implementation**: Port Pool & Auto-Recovery Integration
    """
)
async def update_service_route(request: PortUpdateRequest) -> PortUpdateResponse:
    """
    Update service route when port changes

    This endpoint receives notifications from ServiceLauncher when a service
    recovers on a new port. It updates the internal routing cache so that
    subsequent requests are routed to the correct port.

    Args:
        request: Port update notification

    Returns:
        Update confirmation with old/new ports
    """
    try:
        logger.info(
            f"🔄 Received port update notification: {request.service} → {request.new_port}"
        )

        # Update internal routing cache
        old_port = update_service_port(request.service, request.new_port)

        # Log the update
        if old_port:
            logger.info(
                f"✅ Service route updated: {request.service} "
                f"({old_port} → {request.new_port})"
            )
        else:
            logger.info(
                f"✅ Service route registered: {request.service} → {request.new_port}"
            )

        return PortUpdateResponse(
            status="success",
            service=request.service,
            old_port=old_port,
            new_port=request.new_port,
            updated_at=datetime.now().isoformat(),
            message=f"Route updated successfully for {request.service}"
        )

    except Exception as e:
        logger.error(f"❌ Failed to update service route: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update service route: {str(e)}"
        )


@router.get(
    "/routes",
    response_model=Dict[str, Any],
    summary="Get all registered service routes",
    description="Returns all services and their cached ports"
)
async def get_service_routes() -> Dict[str, Any]:
    """
    Get all registered service routes

    Returns dictionary of service_name → port for all cached services
    """
    return {
        "status": "success",
        "routes": get_all_service_ports(),
        "count": len(_service_ports),
        "timestamp": datetime.now().isoformat()
    }


@router.get(
    "/routes/{service_name}",
    response_model=Dict[str, Any],
    summary="Get route for specific service",
    description="Returns the cached port for a specific service"
)
async def get_service_route(service_name: str) -> Dict[str, Any]:
    """
    Get route for a specific service

    Args:
        service_name: Name of the service

    Returns:
        Service port information
    """
    port = get_service_port(service_name)

    if port is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found in route cache"
        )

    return {
        "status": "success",
        "service": service_name,
        "port": port,
        "timestamp": datetime.now().isoformat()
    }


@router.delete(
    "/routes/{service_name}",
    response_model=Dict[str, Any],
    summary="Remove service route from cache",
    description="Removes a service's cached port (used when service is stopped)"
)
async def delete_service_route(service_name: str) -> Dict[str, Any]:
    """
    Remove service route from cache

    Args:
        service_name: Name of the service to remove

    Returns:
        Deletion confirmation
    """
    if service_name not in _service_ports:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found in route cache"
        )

    old_port = _service_ports.pop(service_name)
    logger.info(f"🗑️  Service route removed: {service_name} (port {old_port})")

    return {
        "status": "success",
        "service": service_name,
        "old_port": old_port,
        "message": f"Route deleted for {service_name}",
        "timestamp": datetime.now().isoformat()
    }
