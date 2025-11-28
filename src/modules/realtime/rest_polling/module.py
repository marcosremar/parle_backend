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
            # Create minimal ServiceContext for module mode
            try:
                from src.modules.conversation.orchestrator.utils.context import ServiceContext
            except ImportError:
                try:
                    from src.services.orchestrator.utils.context import ServiceContext
                except ImportError:
                    ServiceContext = None
            
            context = None
            if ServiceContext:
                try:
                    context = ServiceContext.create(
                        service_name="rest_polling"
                    )
                except Exception as ctx_error:
                    self.logger.warning(f"⚠️  Could not create ServiceContext: {ctx_error}")
            
            self.service = RestPollingService(context=context)
            await self.service.initialize()
            self.logger.info("✅ REST Polling Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  REST polling service not available: {e}")
            self.service = None
            return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service:
                return await self.service.health_check()
            else:
                return {"status": "unavailable", "error": "Service not available"}
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a polling session"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service and hasattr(self.service, 'session_manager'):
                return self.service.session_manager.create_session(session_id)
            else:
                return {"success": False, "error": "Service not available"}
        except Exception as e:
            self.logger.error(f"❌ Session creation failed: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service and hasattr(self.service, 'session_manager'):
                session = self.service.session_manager.get_session(session_id)
                if session:
                    return session
                else:
                    return {"success": False, "error": "Session not found"}
            else:
                return {"success": False, "error": "Service not available"}
        except Exception as e:
            self.logger.error(f"❌ Get session failed: {e}")
            raise
    
    async def queue_message(self, session_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a message for a session"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service and hasattr(self.service, 'session_manager'):
                self.service.session_manager.queue_message(session_id, message)
                return {"success": True}
            else:
                return {"success": False, "error": "Service not available"}
        except Exception as e:
            self.logger.error(f"❌ Queue message failed: {e}")
            raise
    
    async def get_messages(self, session_id: str) -> Dict[str, Any]:
        """Get queued messages for a session"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service and hasattr(self.service, 'session_manager'):
                messages = self.service.session_manager.get_messages(session_id)
                return {"success": True, "messages": messages}
            else:
                return {"success": False, "error": "Service not available", "messages": []}
        except Exception as e:
            self.logger.error(f"❌ Get messages failed: {e}")
            raise
