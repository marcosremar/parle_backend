"""
HTTP Routes for Rest Polling Service
All FastAPI endpoints organized by domain
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging
import time
import base64
from .utils.route_helpers import add_standard_endpoints

logger = logging.getLogger(__name__)

def create_router(rest_polling_service: Any) -> APIRouter:
    """
    Create and configure the Rest Polling Service router

    Args:
        rest_polling_service: RestPollingService instance

    Returns:
        Configured APIRouter with all endpoints
    """
    router = APIRouter()

    