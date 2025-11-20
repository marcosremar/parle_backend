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

    