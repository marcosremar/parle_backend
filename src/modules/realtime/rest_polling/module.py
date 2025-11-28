"""
REST Polling Module - Direct Python calls for REST polling
"""

from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule
from .service import RestPollingService


class RestPollingModule(BaseModule):
    """REST Polling Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("rest_polling")
        self.service = None
    
    async def _initialize(self) -> bool:
        """Initialize REST polling service"""
        try:
            self.service = RestpollingService()
            self.logger.info("✅ REST Polling Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  REST polling service not available: {e}")
            return True
    
    async def poll(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Poll an endpoint"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service:
                return await self.service.poll(endpoint=endpoint, **kwargs)
            else:
                return {"success": False, "error": "Service not available"}
        except Exception as e:
            self.logger.error(f"❌ Polling failed: {e}")
            raise
