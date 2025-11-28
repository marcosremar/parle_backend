"""
Realtime Router - REST Polling
(WebSocket fica em processo separado)
"""

from fastapi import APIRouter
from loguru import logger

router = APIRouter()

try:
    from src.services.rest_polling.app_complete import app as rest_polling_app
    logger.info("✅ Realtime router ready")
except ImportError as e:
    logger.warning(f"⚠️  Realtime router not available: {e}")

@router.get("/health")
async def health():
    """Health check for realtime services"""
    return {"status": "ok", "services": ["rest_polling"]}
