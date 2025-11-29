"""
Base Service - Simplified version for modules
Abstract base class for services in module mode
"""

from abc import ABC
from fastapi import APIRouter
from typing import Dict, Any, Optional
from loguru import logger


class BaseService(ABC):
    """
    Simplified BaseService for module mode
    Provides basic structure without HTTP server dependencies
    """

    def __init__(self, context: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the service
        
        Args:
            context: ServiceContext with dependencies (optional in module mode)
            config: Optional service configuration dictionary
        """
        self.context = context
        self.config = config or {}
        
        # Use logger from context if available, otherwise use loguru
        if context and hasattr(context, 'logger'):
            self.logger = context.logger
        else:
            self.logger = logger
        
        # Communication manager (optional in module mode)
        if context and hasattr(context, 'comm'):
            self.comm = context.comm
        else:
            self.comm = None
        
        # Settings (optional)
        if context and hasattr(context, 'settings'):
            self.settings = context.settings
        else:
            self.settings = None
        
        self.router = APIRouter()
        self.initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the service
        
        Returns:
            True if initialization successful
        """
        self.initialized = True
        return True

    def get_router(self) -> APIRouter:
        """Get FastAPI router (empty in module mode)"""
        return self.router

    async def health_check(self) -> Dict[str, Any]:
        """Health check"""
        return {
            "status": "healthy" if self.initialized else "not_initialized",
            "service": self.__class__.__name__
        }
