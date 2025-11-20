"""
HTTP Routes for Session Service
All FastAPI endpoints organized by domain
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from .utils.route_helpers import add_standard_endpoints
from .models import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionListResponse,
    HealthResponse,
    LLMType
)

logger = logging.getLogger(__name__)

def create_router(session_service: Any) -> APIRouter:
    """
    Create and configure the Session Service router

    Args:
        session_service: SessionService instance for accessing state

    Returns:
        Configured APIRouter with all endpoints
    """
    router = APIRouter()

    # ==================== Health & Info ====================

    @router.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint"""
        try:
            is_connected = await session_service.session_manager.is_connected()
            active_count = await session_service.session_manager.get_active_count() if is_connected else 0

            return HealthResponse(
                status="healthy" if is_connected else "degraded",
                redis_connected=is_connected,
                active_sessions=active_count,
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            # Redis connection failed - return degraded status but don't crash
            logger.warning(f"⚠️ Health check failed (Redis unavailable): {e}")
            return HealthResponse(
                status="degraded",
                redis_connected=False,
                active_sessions=0,
                timestamp=datetime.utcnow().isoformat()
            )

    # ==================== Validation ====================

    