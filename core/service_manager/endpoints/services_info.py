"""
Service Manager - Info & Discovery Endpoints
Handles service listing, info retrieval
"""

from fastapi import APIRouter, Depends
from typing import List, Dict, Any
from loguru import logger

from .dependencies import get_manager, validate_service_id

router = APIRouter(prefix="/services", tags=["info"])


@router.get("")
async def list_services(manager=Depends(get_manager)) -> List[str]:
    """List all available services."""
    pass


@router.get("/{service_id}")
async def get_service_info(
    service_id: str,
    manager=Depends(get_manager)
) -> Dict[str, Any]:
    """Get detailed information about a service."""
    service_id = validate_service_id(service_id)
    pass
