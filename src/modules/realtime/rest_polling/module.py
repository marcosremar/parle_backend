"""
REST Polling Module - Direct Python calls for REST polling
"""

from typing import Any

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
            context = None
            try:
                from src.modules.conversation.orchestrator.utils.context import ServiceContext
            except ImportError:
                ServiceContext = None

            if ServiceContext:
                try:
                    # Create a minimal mock communication manager
                    class MockComm:
                        """Minimal mock communication manager for module mode"""

                        def get_service_url(self, service_name):
                            return None

                        def call_service(self, *args, **kwargs):
                            return {"success": False, "error": "Not available in module mode"}

                        async def send_request(self, *args, **kwargs):
                            return {"success": False, "error": "Not available in module mode"}

                    mock_comm = MockComm()

                    # Create ServiceContext with mock comm
                    context = ServiceContext.create(
                        service_name="rest_polling", comm=mock_comm, execution_mode="module"
                    )
                except Exception as ctx_error:
                    self.logger.warning(f"⚠️  Could not create ServiceContext: {ctx_error}")

                    # Create minimal context manually
                    class MinimalContext:
                        def __init__(self, logger):
                            self.logger = logger
                            self.comm = None

                    context = MinimalContext(self.logger)
            else:
                # Fallback: create minimal context manually
                class MinimalContext:
                    def __init__(self, logger):
                        self.logger = logger
                        self.comm = None

                context = MinimalContext(self.logger)

            self.service = RestPollingService(context=context)
            init_result = await self.service.initialize()
            if not init_result:
                self.logger.warning("⚠️  REST polling service initialization returned False")
                # Keep service instance even if init failed - it might still work
            self.logger.info("✅ REST Polling Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  REST polling service not available: {e}")
            import traceback

            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            # Don't set service to None - keep it if it was created
            if not hasattr(self, "service") or self.service is None:
                self.service = None
            return True

    async def health_check(self) -> dict[str, Any]:
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

    async def create_session(self, session_id: str) -> dict[str, Any]:
        """Create a polling session"""
        if not self.initialized:
            await self.initialize()

        try:
            if self.service and hasattr(self.service, "session_manager"):
                return self.service.session_manager.create_session(session_id)
            else:
                return {"success": False, "error": "Service not available"}
        except Exception as e:
            self.logger.error(f"❌ Session creation failed: {e}")
            raise

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get session information"""
        if not self.initialized:
            await self.initialize()

        try:
            if self.service and hasattr(self.service, "session_manager"):
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

    async def queue_message(self, session_id: str, message: dict[str, Any]) -> dict[str, Any]:
        """Queue a message for a session"""
        if not self.initialized:
            await self.initialize()

        try:
            if self.service and hasattr(self.service, "session_manager"):
                self.service.session_manager.queue_message(session_id, message)
                return {"success": True}
            else:
                return {"success": False, "error": "Service not available"}
        except Exception as e:
            self.logger.error(f"❌ Queue message failed: {e}")
            raise

    async def get_messages(self, session_id: str) -> dict[str, Any]:
        """Get queued messages for a session"""
        if not self.initialized:
            await self.initialize()

        try:
            if self.service and hasattr(self.service, "session_manager"):
                messages = self.service.session_manager.get_messages(session_id)
                return {"success": True, "messages": messages}
            else:
                return {"success": False, "error": "Service not available", "messages": []}
        except Exception as e:
            self.logger.error(f"❌ Get messages failed: {e}")
            raise
