"""
User Service - Manages user authentication, profiles, and session management
"""

from typing import Dict
from datetime import datetime
import secrets
from pathlib import Path

from .utils.base_service import BaseService

# Context system (NEW)
from src.services.orchestrator.utils.context import ServiceContext
from src.core.shared.models.config_models import UserConfig
from typing import Optional, Dict
from loguru import logger

from .routes import create_router
from .storage import users_db
from .core.auth import hash_password

class UserService(BaseService):
    """User Service - manages user authentication and profiles"""

    def __init__(self, context: ServiceContext, config: Optional[Dict] = None):
        """
        Initialize User Service with Dependency Injection

        Args:
            context: ServiceContext with injected dependencies (REQUIRED)
            config: Optional service configuration
        """
        super().__init__(context=context, config=config)

    def _setup_router(self) -> None:
        """Setup FastAPI routes using the new modular structure"""
        # Import and attach the router created by the routes module
        router = create_router(service=self)

        # Include all routes from the modular router
        self.router.include_router(router)

    async def initialize(self) -> bool:
        """Initialize service resources"""
        try:
            # Use self.config (injected via ServiceContext)
            port = self.config.get('service', {}).get('port', 8200)
            self.logger.info(f"User Service initializing, port={port}")

            # Create default admin user for testing
            admin_id = "admin_" + secrets.token_hex(8)
            users_db[admin_id] = {
                "user_id": admin_id,
                "username": "admin",
                "email": "admin@ultravox.local",
                "password_hash": hash_password("admin123"),
                "full_name": "System Administrator",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "preferences": {"theme": "dark", "language": "pt-BR"},
                "is_active": True,
                "is_admin": True
            }

            self.logger.info(f"Default admin user created: username=admin, user_id={admin_id}")
            self.logger.info("✅ User Service initialized successfully")

            # Update metrics (use injected metrics collector if available)
            if self.metrics:
                self.metrics.gauge("registered_users", len(users_db), {"service": "user"})

            self.initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize User Service: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def health_check(self) -> Dict:
        """Perform health check"""
        from .storage import sessions_db, api_keys_db, get_stats

        stats = get_stats()

        return {
            "status": "healthy" if self.initialized else "unhealthy",
            "service": "user-service",
            "timestamp": datetime.now().isoformat(),
            **stats
        }

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self.logger.info("🛑 Shutting down User Service...")
        self.initialized = False
        self.logger.info("✅ User Service shut down successfully")
