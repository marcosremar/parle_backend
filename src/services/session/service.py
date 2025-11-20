"""
Session Service
Manages user sessions with Redis persistence

Can run in two modes (determined by Service Manager):
- Internal: In-process within Service Manager
- External: As standalone HTTP server
"""

import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from fastapi import HTTPException, status

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Logging and Metrics (after adding to path)
from loguru import logger

# Setup logging and metrics for Session Service
.parent / "tmp" / "metrics")

from .utils.base_service import BaseService
from config.settings import get_sessions_settings
from src.services.database.database_client import UserDatabase

# Context system (NEW)
from src.services.orchestrator.utils.context import ServiceContext

# Handle both module import and direct execution
try:
    from .models import (
        SessionCreate,
        SessionUpdate,
        SessionResponse,
        SessionListResponse,
        HealthResponse,
        LLMType
    )
    from .redis_manager import SessionManager
    from .database_config import DATABASE_CONFIG
except ImportError:
    from .models import (
        SessionCreate,
        SessionUpdate,
        SessionResponse,
        SessionListResponse,
        HealthResponse,
        LLMType
    )
    from .redis_manager import SessionManager
    from .database_config import DATABASE_CONFIG

# Note: parse_env_var has been deprecated in favor of SettingsService methods
# The function is kept for backward compatibility but should not be used in new code

class SessionService(BaseService):
    """
    Session Service - manages user sessions with Redis

    Uses ServiceContext for dependency injection (REQUIRED)
    """

    def __init__(self, config: Dict[str, Any], context: ServiceContext) -> None:
        """
        Initialize Session Service

        Args:
            config: Service configuration
            context: ServiceContext for dependency injection (REQUIRED)
        """
        self.session_manager: SessionManager = None
        self.settings = None
        super().__init__(context=context, config=config)

        # Log initialization
        if self.context:
            self.logger.info("🎯 Session Service using ServiceContext (DI enabled)")
        else:
            self.logger.warning("⚠️ Session Service initialized without ServiceContext")

    def _setup_router(self) -> None:
        """Setup FastAPI routes using the new modular structure"""
        from .routes import create_router

        router = create_router(self)
        self.router.include_router(router)

    async def initialize(self) -> bool:
        """Initialize session manager and Redis connection"""
        try:
            self.settings = get_sessions_settings()

            # Use SettingsService to parse environment variables with ${VAR:-default} syntax
            # This replaces the deprecated parse_env_var function
            redis_url = self.settings.redis_url
            session_ttl = self.settings.session_ttl
            redis_db = self.settings.redis_db

            # If values contain ${VAR:-default} syntax, they should already be resolved by get_sessions_settings()
            # But for extra safety with dynamic values, we can use self.settings (from ServiceContext)
            if hasattr(self, 'settings') and self.settings:
                # Use SettingsService for dynamic variable resolution
                # Note: The session service's settings object already handles this,
                # but we document the pattern for other services
                pass

            self.logger.info(f"📊 Redis URL: {redis_url}")
            self.logger.info(f"⏱️  Default TTL: {session_ttl}s ({session_ttl // 60} minutes)")

            # Initialize session manager with Communication Manager from context
            self.session_manager = SessionManager(
                redis_url=redis_url,
                default_ttl=session_ttl,
                redis_db=redis_db,
                comm_manager=self.comm  # Inject Communication Manager from ServiceContext
            )

            # Try to connect to Database Service via Communication Manager
            try:
                await self.session_manager.connect()
                self.logger.info("✅ Session Service initialized with Database Service")
            except Exception as e:
                self.logger.warning(f"⚠️  Database Service not available: {e}")
                self.logger.info("✅ Session Service initialized (degraded mode - sessions may not persist)")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Session Service: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        is_connected = await self.session_manager.is_connected() if self.session_manager else False
        active_count = await self.session_manager.get_active_count() if is_connected else 0

        return {
            "status": "healthy" if is_connected else "unhealthy",
            "redis_connected": is_connected,
            "active_sessions": active_count,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def shutdown(self) -> None:
        """Cleanup resources"""
        if self.session_manager:
            await self.session_manager.disconnect()
            self.logger.info("🛑 Session Service shutdown complete")
