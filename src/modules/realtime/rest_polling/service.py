"""
REST Polling Service - BaseService Implementation
HTTP Long-Polling fallback transport when real-time transports fail
"""

import base64
import os
from pathlib import Path
import sys
import time

# Add project to path FIRST (before src.core imports)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# Logging and Metrics (simplified for module mode)
def increment_metric(name, value=1, labels=None):
    """Increment metric (no-op in module mode)"""


def set_gauge(name, value, labels=None):
    """Set gauge (no-op in module mode)"""


from datetime import datetime

# Setup logging and metrics for Rest Polling Service
# (metrics path setup removed - handled by try/except above)
from loguru import logger

# Import BaseService from local utils
from .utils.base_service import BaseService

# ConversationController is not used in module mode (optional import)
try:
    from src.core.controllers.conversation_controller import ConversationController
except ImportError:
    ConversationController = None  # Not needed for module mode

# Context system
from src.modules.conversation.orchestrator.utils.context import ServiceContext

# Centralized config models (optional - only needed if settings available)
try:
    from src.core.shared.models.config_models import PortConfig
except ImportError:
    PortConfig = None  # Not critical for module mode


class SessionManager:
    """Manages REST polling sessions and message queues"""

    def __init__(self, context: ServiceContext) -> None:

        # Context-based DI (NEW)
        self.context = context

        # Use context logger if available, otherwise use default
        self.logger = context.logger
        self.logger.info("🎯 Service using ServiceContext (DI enabled)")
        self.sessions: dict[str, dict] = {}
        self.message_queues: dict[str, list] = {}

    def create_session(self, session_id: str) -> dict:
        """Create a new session"""
        self.sessions[session_id] = {
            "session_id": session_id,
            "transport_type": "rest",
            "created_at": datetime.now().isoformat(),
            "last_activity": time.time(),
            "messages_sent": 0,
            "messages_received": 0,
            "status": "active",
        }
        self.message_queues[session_id] = []
        self.logger.info(f"✅ Session created: {session_id}")
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> dict | None:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def update_activity(self, session_id: str) -> None:
        """Update last activity timestamp"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = time.time()

    def queue_message(self, session_id: str, message: dict) -> None:
        """Queue message for session"""
        if session_id not in self.message_queues:
            self.message_queues[session_id] = []

        self.message_queues[session_id].append({**message, "timestamp": datetime.now().isoformat()})

        # Limit queue size
        if len(self.message_queues[session_id]) > 50:
            self.message_queues[session_id].pop(0)

    def get_messages(self, session_id: str) -> list[dict]:
        """Get and clear queued messages"""
        messages = self.message_queues.get(session_id, [])
        self.message_queues[session_id] = []
        return messages

    def close_session(self, session_id: str) -> None:
        """Close a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.message_queues:
            del self.message_queues[session_id]
        self.logger.info(f"❌ Session closed: {session_id}")


class AudioProcessor:
    """Process audio via Orchestrator Service HTTP API"""

    def __init__(self, context: ServiceContext) -> None:
        # Context-based DI (NEW)
        self.context = context

        # Use context logger if available, otherwise use default
        self.logger = context.logger
        self.logger.info("🎯 Service using ServiceContext (DI enabled)")
        self.orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8500")
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize audio processor (lightweight - just sets flag)"""
        if not self._initialized:
            self._initialized = True
            self.logger.info("✅ AudioProcessor initialized (lightweight mode)")
            self.logger.info(f"   Will call orchestrator at: {self.orchestrator_url}")

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        session_id: str = "rest_session",
        force_external_llm: bool = False,
    ) -> dict:
        """Process audio by calling Orchestrator Service via Communication Manager"""
        try:
            if not self._initialized:
                await self.initialize()

            # Encode audio to base64 for HTTP transport
            audio_b64 = base64.b64encode(audio_data).decode()

            # Get Communication Manager from context
            if not self.context or not hasattr(self.context, "comm"):
                raise Exception("Communication Manager not available in context")

            comm = self.context.comm

            # Call orchestrator service via Communication Manager (optimized!)
            result = await comm.call_service(
                service_name="orchestrator",
                endpoint_path="/process-turn",
                method="POST",
                json_data={
                    "audio": audio_b64,
                    "session_id": session_id,
                    "sample_rate": sample_rate,
                    "force_external_llm": force_external_llm,
                },
                timeout=30.0,
            )

            # Return result in expected format
            if result.get("success"):
                return {
                    "type": "audio_processed",
                    "success": True,
                    "transcription": result.get("transcript", ""),
                    "text": result.get("text", ""),
                    "audio": result.get("audio"),  # Already base64 encoded
                    "audio_generated": bool(result.get("audio")),
                    "llm_used": result.get("llm_used", "unknown"),
                }
            else:
                self.logger.error(f"Orchestrator processing failed: {result.get('error')}")
                return {
                    "type": "audio_processed",
                    "success": False,
                    "error": result.get("error", "Processing failed"),
                }

        except Exception as e:
            increment_metric("service_initializations", "rest_polling", status="error")
            self.logger.error(f"Error processing audio: {e}")
            return {
                "type": "audio_processed",
                "success": False,
                "error": f"Failed to process audio: {e!s}",
            }

    async def cleanup(self) -> None:
        """Cleanup (lightweight - nothing to clean up)"""
        self.logger.info("AudioProcessor cleanup complete (no resources to clean)")


class RestPollingService(BaseService):
    """REST Polling Service using BaseService"""

    def __init__(self, config: dict = None, context: ServiceContext | None = None) -> None:
        # Pass context to BaseService (DI support)
        super().__init__(context=context, config=config)

        # Load port configuration from SettingsService
        if self.settings and PortConfig:
            try:
                self.port_config = PortConfig.from_settings(self.settings)
                self.logger.info(
                    f"🎯 REST Polling Service port configured: {self.port_config.rest_polling_port}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️  Could not load PortConfig: {e}")
                self.port_config = None
        else:
            # Fallback for legacy mode
            self.port_config = None
            if not self.settings:
                self.logger.warning(
                    "⚠️  SettingsService not available, using legacy port configuration"
                )
            if not PortConfig:
                self.logger.warning("⚠️  PortConfig not available, using legacy port configuration")

        # Always create a minimal context for standalone mode
        # This is a workaround for tests that run service.py directly
        if not self.context:
            self.logger.warning(
                "⚠️  REST Polling Service initialized without ServiceContext (legacy mode)"
            )

            # Create a minimal mock context for standalone execution
            class MinimalContext:
                def __init__(self):
                    self.logger = logger

            self.context = MinimalContext()

        self.logger.info("🎯 REST Polling Service initialized with ServiceContext (DI enabled)")
        self.session_manager = SessionManager(self.context)
        self.processor = AudioProcessor(self.context)

        # Create ConversationController (will be initialized with pipeline later)
        self.controller = None

    def _setup_router(self) -> None:
        """Setup FastAPI routes using the new modular structure"""
        # In module mode, we don't need routes (no FastAPI app)
        # Only setup router if we're running as a service
        try:
            # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
            from .routes import create_router

            router = create_router(self)
            if hasattr(self, "router") and self.router:
                self.router.include_router(router)
        except ImportError:
            # Routes not available in module mode - that's OK
            self.logger.debug("Routes not available in module mode (expected)")
        except AttributeError:
            # Router not available in module mode - that's OK
            self.logger.debug("Router not available in module mode (expected)")

    async def initialize(self) -> bool:
        """Initialize REST Polling service (lightweight mode)"""
        try:
            # Initialize audio processor (lightweight - just configures HTTP client)
            await self.processor.initialize()

            # Update metrics
            increment_metric("service_initializations", "rest_polling", status="success")

            self.logger.info("✅ REST Polling Service initialized (lightweight mode)")
            return True
        except Exception as e:
            increment_metric("service_initializations", "rest_polling", status="error")
            self.logger.error(f"❌ Failed to initialize REST Polling Service: {e}")
            return False

    async def health_check(self) -> dict:
        """Perform health check"""
        return {
            "status": "healthy",
            "active_sessions": len(self.session_manager.sessions),
            "timestamp": datetime.now().isoformat(),
        }

    async def shutdown(self) -> None:
        """Cleanup resources"""
        # Close all sessions
        for session_id in list(self.session_manager.sessions.keys()):
            self.session_manager.close_session(session_id)

        # Cleanup audio processor
        await self.processor.cleanup()

        self.logger.info("🛑 REST Polling Service shutdown complete")


if __name__ == "__main__":
    import os

    from fastapi import FastAPI
    import uvicorn

    # telemetry_middleware removed import add_telemetry_middleware

    # Note: In standalone mode, we still use os.getenv for backward compatibility
    # When running via Service Manager, the port comes from SettingsService
    port = int(os.getenv("REST_POLLING_PORT", "8106"))  # Dynamic allocation supported via PortPool

    config = {"name": "rest_polling", "port": port, "host": "0.0.0.0"}

    service = RestPollingService(config)

    # Create FastAPI app
    app = FastAPI(title="REST Polling Service")
    # add_telemetry_middleware removed, "rest_polling")

    # Add basic health endpoint for testing
    @app.get("/health")
    async def health():
        return await service.health_check()

    # Include service router
    app.include_router(service.get_router())

    uvicorn.run(app, host="0.0.0.0", port=port)
