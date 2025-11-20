"""
HTTP Routes for Webrtc Service
All FastAPI endpoints organized by domain
"""

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from typing import Dict, Optional, List, Any, Union
from datetime import datetime
import logging
import os
import time
import json
from .utils.route_helpers import add_standard_endpoints

logger = logging.getLogger(__name__)

# ✅ Phase 3c: Dynamic port support for webrtc service
def _get_webrtc_port(service_name="webrtc"):
    """Get webrtc port from environment or registry"""
    try:
        from src.config.service_config import ServiceType, get_service_port
        env_var = "WEBRTC_PORT" if service_name == "webrtc" else "WEBRTC_SIGNALING_PORT"
        return int(os.getenv(env_var) or get_service_port(ServiceType.WEBRTC_GATEWAY))
    except (ImportError, ValueError, TypeError, KeyError):
        logger.debug("Could not retrieve webrtc port from config, using fallback")
        return 8020  # Fallback to PORT_MATRIX default

def create_router(webrtc_service: Any) -> APIRouter:
    """
    Create and configure the Webrtc Service router

    Args:
        webrtc_service: WebrtcService instance

    Returns:
        Configured APIRouter with all endpoints
    """
    router = APIRouter()

    @router.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint"""
        return {
            "service": "WebRTC Gateway",
            "description": "Real-time WebRTC communication gateway",
            "version": "2.0.0",
            "endpoints": {
                "health": "/health",
                "info": "/info",
                "ws": "/ws"
            },
            "features": [
                "WebRTC signaling",
                "Audio processing via Orchestrator",
                "WebSocket support"
            ]
        }

    