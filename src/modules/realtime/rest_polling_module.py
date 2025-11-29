"""
Rest Polling Module - Direct Python calls for Rest polling
"""

from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule


class RestPollingModule(BaseModule):
    """Rest Polling Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("rest_polling")
        self.service = None
    
    async def _initialize(self) -> bool:
        """Initialize rest polling service"""
        try:
            # Import from local module
            from .rest_polling.service import RestPollingService
            from src.core.unified_context import ServiceContext
            from src.core.communication.facade import ServiceCommunicationManager
            
            # Create minimal ServiceContext
            try:
                comm = ServiceCommunicationManager()
            except:
                # Fallback mock communication manager
                class MockComm:
                    def get_service_url(self, service_name): return None
                    def send_request(self, *args, **kwargs): return None
                comm = MockComm()
            
            config = {"name": "rest_polling", "port": 8700}
            context = ServiceContext.create(
                service_name="rest_polling",
                comm=comm,
                config=config,
                profile="standalone",
                execution_mode="external"
            )
            
            self.service = RestpollingService(config=config, context=context)
            success = await self.service.initialize()
            
            if success:
                self.logger.info("✅ Rest Polling Module initialized")
                return True
            else:
                self.logger.warning("⚠️  Rest Polling Service initialization returned False")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to initialize Rest Polling Module: {e}")
            # Fallback: service not available but module can still exist
            self.service = None
            return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service:
                return await self.service.health_check()
            else:
                return {
                    "status": "degraded",
                    "service": "rest_polling",
                    "message": "Service not available"
                }
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "rest_polling",
                "error": str(e)
            }
