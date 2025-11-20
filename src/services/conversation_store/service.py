"""
Conversation Store Service
Manages conversation history and message storage

Can run in two modes (determined by Service Manager):
- Internal: In-process within Service Manager
- External: As standalone HTTP server
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from fastapi import HTTPException

# Add project to path FIRST (before src.core imports)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Logging and Metrics (after adding to path)
from loguru import logger

# Setup logging and metrics for Conversation Store Service
.parent / "tmp" / "metrics")

from .utils.base_service import BaseService

# Context system (NEW)
from src.services.orchestrator.utils.context import ServiceContext

# Centralized config models
from src.core.shared.models.config_models import PortConfig

class ConversationStoreService(BaseService):
    """
    Conversation Store Service - manages conversation history

    Uses ServiceContext for dependency injection (REQUIRED)
    """

    def __init__(self, config: Dict[str, Any] = None, context: ServiceContext = None) -> None:
        """
        Initialize Conversation Store Service

        Args:
            config: Service configuration (optional)
            context: ServiceContext for dependency injection (optional, recommended)
        """
        # In-memory storage (legacy)
        self.conversations_db: Dict[str, Dict] = {}
        self.contexts_db: Dict[str, List] = {}
        self.sessions_db: Dict[str, Dict] = {}

        # Fast storage with semantic search
        self.fast_storage = None

        # Pass context to BaseService first
        super().__init__(context=context, config=config)

        # Load port configuration from SettingsService
        if self.settings:
            self.port_config = PortConfig.from_settings(self.settings)
            self.logger.info(f"🎯 Conversation Store Service port configured: {self.port_config.conversation_store_port}")
        else:
            # Fallback for legacy mode
            self.port_config = None
            self.logger.warning("⚠️  SettingsService not available, using legacy port configuration")

        # Log initialization
        if self.context:
            self.logger.info("🎯 Conversation Store Service using ServiceContext (DI enabled)")
        else:
            self.logger.warning("⚠️ Conversation Store Service initialized without ServiceContext")

    def _setup_router(self) -> None:
        """Setup FastAPI routes using the new modular structure"""
        # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
        from .routes import create_router

        router = create_router(self)
        self.router.include_router(router)

    async def initialize(self) -> bool:
        """Initialize conversation store"""
        try:
            # Initialize fast storage with semantic search
            from .storage import (
                FastConversationStorage,
                PERFORMANCE_CONFIG,
                SEMANTIC_SEARCH_CONFIG,
                DEFAULT_STORAGE_DIR
            )

            # Override with service-specific config if exists
            try:
                from .config import get_storage_config
                service_config = get_storage_config()
            except ImportError:
                service_config = {}

            self.fast_storage = FastConversationStorage(
                enable_redis=service_config.get("enable_redis", PERFORMANCE_CONFIG.get("enable_redis", False)),
                redis_url=service_config.get("redis_url", PERFORMANCE_CONFIG.get("redis_url")),
                storage_dir=service_config.get("storage_dir", DEFAULT_STORAGE_DIR),
                write_buffer_size=PERFORMANCE_CONFIG.get("write_buffer_size", 1000),
                flush_interval_ms=PERFORMANCE_CONFIG.get("flush_interval_ms", 100),
                enable_semantic_search=SEMANTIC_SEARCH_CONFIG.get("enable_semantic_search", True),
                enable_long_term_memory=SEMANTIC_SEARCH_CONFIG.get("enable_long_term_memory", True)
            )

            self.logger.info("✅ Conversation Store Service initialized with semantic search")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Conversation Store: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "status": "healthy",
            "active_conversations": len(self.conversations_db),
            "active_sessions": len(self.sessions_db),
            "total_contexts": len(self.contexts_db)
        }

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self.logger.info("🛑 Conversation Store Service shutdown complete")

if __name__ == "__main__":
    import uvicorn
    import os
    from fastapi import FastAPI
    # telemetry_middleware removed import add_telemetry_middleware

    # Note: In standalone mode, we still use os.getenv for backward compatibility
    # When running via Service Manager, the port comes from SettingsService
    port = int(os.getenv("CONVERSATION_STORE_PORT", "8200"))  # Dynamic allocation supported via PortPool

    config = {
        "name": "conversation_store",
        "port": port,
        "host": "0.0.0.0"
    }

    service = ConversationStoreService(config)

    # Create FastAPI app and include service router
    app = FastAPI(title="Conversation Store Service")
    # add_telemetry_middleware removed, "conversation_store")

    app.include_router(service.get_router())

    uvicorn.run(app, host="0.0.0.0", port=port)
