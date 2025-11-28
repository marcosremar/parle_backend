"""
Storage Router - Conversation Store, History, File Storage, Database
"""

from fastapi import APIRouter
from loguru import logger

router = APIRouter()

try:
    from src.services.conversation_store.routes import create_router as create_conversation_store_router
    
    conversation_store_router = create_conversation_store_router(None)
    if conversation_store_router:
        router.include_router(conversation_store_router, prefix="/conversation-store")
    
    logger.info("✅ Storage routers ready")
except ImportError as e:
    logger.warning(f"⚠️  Some storage routers not available: {e}")
except Exception as e:
    logger.warning(f"⚠️  Error setting up storage routers: {e}")

@router.get("/health")
async def health():
    """Health check for storage services"""
    return {"status": "ok", "services": ["conversation_store", "conversation_history", "file_storage", "database"]}
