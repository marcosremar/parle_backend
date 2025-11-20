"""
HTTP Routes for Webrtc Signaling Service
All FastAPI endpoints organized by domain
"""

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from typing import Dict, Optional, List, Any
from datetime import datetime
import asyncio
import logging
import os
from .utils.route_helpers import add_standard_endpoints

logger = logging.getLogger(__name__)

# ✅ Phase 3c: Dynamic port support
def _get_webrtc_signaling_port():
    """Get webrtc_signaling port from environment or registry"""
    try:
        from src.config.service_config import ServiceType, get_service_port
        return int(os.getenv("WEBRTC_SIGNALING_PORT") or get_service_port(ServiceType.WEBRTC_SIGNALING))
    except (ImportError, ValueError, TypeError, KeyError):
        logger.debug("Could not retrieve webrtc_signaling port from config, using fallback")
        return 8090  # Fallback to PORT_MATRIX default

def create_router(webrtc_signaling_service: Any) -> APIRouter:
    """
    Create and configure the Webrtc Signaling Service router

    Args:
        webrtc_signaling_service: WebrtcSignalingService instance

    Returns:
        Configured APIRouter with all endpoints
    """
    router = APIRouter()

    