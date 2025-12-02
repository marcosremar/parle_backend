"""
WebRTC Signaling Module - Signaling for WebRTC connections
"""

from typing import Any, Dict

from src.modules.base_module import BaseModule
from .service import WebRTCSignalingService

class WebRTCSignalingModule(BaseModule):
    """WebRTC Signaling Module"""

    def __init__(self):
        super().__init__("webrtc_signaling")
        self.service = None

    async def _initialize(self) -> bool:
        """Initialize WebRTC Signaling module"""
        try:
            # Import ServiceContext
            from src.modules.conversation.orchestrator.utils.unified_context import ServiceContext
            
            # Mock Communication Manager
            class MockComm:
                def get_service_url(self, service_name):
                    return None
                def send_request(self, *args, **kwargs):
                    return None
                def register_internal_service(self, *args, **kwargs):
                    pass
            
            comm = MockComm()
            
            # Create ServiceContext manually to avoid logging reconfiguration issues
            from src.modules.conversation.orchestrator.utils.unified_context import ResourceLimits
            
            config = {"name": "webrtc_signaling", "port": 8080}
            # Bypass ServiceContext.create to avoid setup_logging side effects
            context = ServiceContext(
                service_name="webrtc_signaling",
                comm=comm,
                config=config,
                profile="monolith",
                execution_mode="module",
                logger=self.logger,
                limits=ResourceLimits()
            )
            
            # Create service instance
            self.service = WebRTCSignalingService(context=context)
            
            # Initialize service
            success = await self.service.initialize()
            
            if success:
                self.logger.info("✅ WebRTC Signaling Module initialized")
                return True
            else:
                self.logger.error("❌ WebRTC Signaling Service initialization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize WebRTC Signaling Module: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_router(self):
        """Get FastAPI router"""
        if not self.service:
            return None
        return self.service.get_router()

    async def cleanup(self):
        """Cleanup resources"""
        if self.service:
            await self.service.shutdown()
        await super().cleanup()
