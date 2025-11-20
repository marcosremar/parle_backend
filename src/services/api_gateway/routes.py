"""
HTTP Routes for API Gateway Service
Core endpoints for the API Gateway
"""

from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
import logging
from .utils.route_helpers import add_standard_endpoints

logger = logging.getLogger(__name__)

def create_router(api_gateway_service: Any) -> APIRouter:
    """
    Create and configure the API Gateway Service router

    Args:
        api_gateway_service: APIGatewayService instance (can be None for basic routes)

    Returns:
        Configured APIRouter with all endpoints
    """
    # Always return a valid router, even if service is None
    router = APIRouter()
    
    # If service is None, only add basic routes without service-specific functionality
    if api_gateway_service is None:
        from loguru import logger
        logger.warning("⚠️  create_router called with None service - creating basic router only")

    # ==================== Root & Documentation ====================

    @router.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint"""
        return {
            "message": "Ultravox Pipeline API Server",
            "version": "1.0.0",
            "docs": "/api/docs",
            "status": "running",
            "services": {
                "health": "/api/health",
                "tts": "/api/tts",
                "llm": "/api/llm",
                "process": "/api/process",
                "models": "/api/models",
                "conversation": "/api/conversation"
            }
        }

    @router.get("/api/docs")
    async def api_docs() -> Dict[str, Any]:
        """API Documentation endpoint"""
        return {
            "title": "Ultravox Pipeline API",
            "version": "1.0.0",
            "description": "Unified API for speech-to-speech processing with Ultravox",
            "openapi_url": "/openapi.json",
            "swagger_ui_url": "/docs",
            "redoc_url": "/redoc",
            "endpoints": {
                "health": {
                    "path": "/api/health",
                    "method": "GET",
                    "description": "Health check endpoint"
                },
                "tts": {
                    "path": "/api/tts",
                    "method": "POST",
                    "description": "Text-to-speech synthesis"
                },
                "llm": {
                    "path": "/api/llm",
                    "method": "POST",
                    "description": "Language model inference"
                },
                "process": {
                    "path": "/api/process",
                    "method": "POST",
                    "description": "Full speech-to-speech processing"
                },
                "models": {
                    "path": "/api/models",
                    "method": "GET",
                    "description": "List available models"
                },
                "conversation": {
                    "path": "/api/conversation",
                    "method": "POST",
                    "description": "Conversation processing"
                }
            }
        }

    # ==================== Validation ====================
    
    # Add standard endpoints if service is available
    if api_gateway_service is not None:
        try:
            add_standard_endpoints(router, api_gateway_service, "api_gateway")
        except Exception as e:
            logger.warning(f"Failed to add standard endpoints: {e}")
    
    return router