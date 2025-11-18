"""
HTTP Routes for User Service
All FastAPI endpoints organized by domain
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict
from datetime import datetime

from loguru import logger
from src.core.metrics import (
    track_request, increment_metric, set_gauge
)

from .models import (
    UserRegister, UserLogin, UserUpdate, UserProfile,
    SessionInfo, APIKeyCreate, APIKeyInfo, LoginResponse, APIKeyResponse
)
from .dependencies import get_current_user, require_admin
from .core import user_manager, session_manager, api_key_manager
from .storage import get_stats, users_db
from .config import get_config
from src.core.route_helpers import add_standard_endpoints

def create_router(service) -> APIRouter:
    """
    Create and configure the User Service router

    Args:
        service: UserService instance (for future use)

    Returns:
        Configured APIRouter with all endpoints
    """
    router = APIRouter()

    # ==================== Health & Info ====================

    