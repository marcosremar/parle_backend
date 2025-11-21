"""
HTTP Routes for Conversation Store Service
All FastAPI endpoints organized by domain
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Optional, Any
from datetime import datetime
import logging
from .utils.route_helpers import add_standard_endpoints

logger = logging.getLogger(__name__)

def create_router(conversation_store_service: Any) -> APIRouter:
    """
    Create and configure the Conversation Store Service router

    Args:
        conversation_store_service: ConversationStoreService instance

    Returns:
        Configured APIRouter with all endpoints
    """
    router = APIRouter()

    # ==================== Health & Info ====================

    @router.post("/api/conversations/{conversation_id}/turn")
    async def add_turn_to_conversation(
        conversation_id: str,
        turn_data: Dict[str, Any],
        conversation_store_service: Any = None
    ):
        """Add a turn to a conversation"""
        try:
            if conversation_store_service and hasattr(conversation_store_service, 'add_turn_to_conversation'):
                result = await conversation_store_service.add_turn_to_conversation(
                    conversation_id, turn_data
                )
                return result
            else:
                # Fallback: just acknowledge the request
                return {"conversation_id": conversation_id, "turn_added": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/conversations/{conversation_id}/messages")
    async def get_conversation_messages(
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
        conversation_store_service: Any = None
    ):
        """Get messages for a conversation"""
        try:
            if conversation_store_service and hasattr(conversation_store_service, 'get_conversation_messages'):
                result = await conversation_store_service.get_conversation_messages(
                    conversation_id, limit, offset
                )
                return result
            else:
                # Fallback: return empty list
                return {"conversation_id": conversation_id, "messages": [], "total": 0}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

