"""
API Gateway Service - BaseService Implementation
Unified API for speech-to-speech processing with Ultravox
"""

import sys
import os
from pathlib import Path

# Add project to path FIRST (before core imports)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Logging and Metrics (after adding to path)
try:
    from .utils.metrics import increment_metric, set_gauge
except ImportError:
    def increment_metric(name, value=1, labels=None):
        pass
    def set_gauge(name, value, labels=None):
        pass

from loguru import logger

# Setup logging and metrics for Api Gateway Service
# Metrics path: project_root / "tmp" / "metrics"

from typing import Dict

from fastapi.staticfiles import StaticFiles

from .utils.base_service import BaseService

# Context system (NEW)
from src.services.orchestrator.utils.context import ServiceContext
from typing import Optional

# Centralized config models
from src.core.shared.models.config_models import PortConfig

class APIGatewayService(BaseService):
    """API Gateway Service using BaseService"""

    def __init__(self, config: Dict = None, context: Optional[ServiceContext] = None) -> None:
        # Pass context to BaseService (DI support)
        super().__init__(context=context, config=config)

        # Load port configuration from SettingsService
        if self.settings:
            self.port_config = PortConfig.from_settings(self.settings)
            self.logger.info(f"🎯 API Gateway Service port configured: {self.port_config.api_gateway_port}")
        else:
            # Fallback for legacy mode
            self.port_config = None
            self.logger.warning("⚠️  SettingsService not available, using legacy port configuration")

        # Log DI status
        if self.context:
            self.logger.info("🎯 API Gateway Service initialized with ServiceContext (DI enabled)")
            self.logger.info(f"   - Logger: ✅ injected (scoped)")
            self.logger.info(f"   - Communication: ✅ injected ({type(self.comm).__name__})")
        else:
            self.logger.warning("⚠️  API Gateway Service initialized without ServiceContext (legacy mode)")

        # Mount routers after initialization
        self._mount_routers()
        self._mount_static_files()

    def _setup_router(self) -> None:
        """Setup FastAPI routes using the new modular structure"""
        self.logger.info("🔧 _setup_router() called - starting router configuration")

        # Include core routes (root, docs, validate)
        from .routes import create_router

        core_router = create_router(self)
        if core_router is None:
            self.logger.error("❌ create_router returned None - cannot include router")
            return
        
        self.router.include_router(core_router)
        self.logger.info("✅ Core router included")

        # Include essential sub-routers (only routers with no missing dependencies)
        try:
            self.logger.info("🔍 Attempting to import scenarios, session, internal, and rest_polling routers...")
            # Import only working routers to avoid import errors blocking all routes
            # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
            from .routers import scenarios, session, internal, rest_polling

            # Include essential routers for SDK functionality
            self.router.include_router(scenarios.router)
            self.logger.info("✅ Scenarios router included")

            self.router.include_router(session.router)
            self.logger.info("✅ Session router included")

            # Include internal router for inter-service communication
            self.router.include_router(internal.router)
            self.logger.info("✅ Internal router included (port update notifications)")

            # Include rest_polling proxy router
            self.router.include_router(rest_polling.router)
            self.logger.info("✅ REST Polling proxy router included")

            self.logger.info("✅ Essential API Gateway sub-routers included (scenarios, session, internal, rest_polling)")
        except Exception as e:
            import traceback
            increment_metric("service_initializations", "api_gateway", status="error")
            self.logger.error(f"❌ Failed to include sub-routers: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _mount_routers(self) -> None:
        """Mount all API routers"""
        try:
            # Import routers dynamically
            # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
            from .routers import health, tts, llm, models, process, binary, validate, conversation

            # Note: When running as external service, http_server_template creates the app
            # When running as internal service, Service Manager mounts the router
            # For now, we log that routers are available
            self.logger.info("✅ API Gateway routers available for mounting")
            self.logger.info("   Routes will be mounted by http_server_template.py (external) or Service Manager (internal)")

        except Exception as e:
            self.logger.error(f"⚠️  Failed to import routers: {e}")
            # Non-critical - service can still run with basic endpoints

    def _mount_static_files(self) -> None:
        """Mount static files for admin dashboard"""
        try:
            static_path = Path(__file__).parent / "static"
            if static_path.exists():
                admin_path = static_path / "admin"
                if admin_path.exists():
                    self.logger.info("✅ Admin dashboard available at /admin")
                    self.logger.info("   Static files will be mounted by http_server_template.py")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to check static files: {e}")

    async def initialize(self) -> bool:
        """Initialize API Gateway service"""
        try:
            # Initialize Communication Manager for all routers
            if self.comm:
                try:
                    # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
                    from .routers import initialize_comm_manager_for_routers
                    initialize_comm_manager_for_routers(self.comm)
                    self.logger.info("✅ Communication Manager initialized for all routers")
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to initialize Communication Manager for routers: {e}")

            # Initialize Orchestrator Client for process router
            try:
                # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
                from .routers.process import initialize_orchestrator_client
                await initialize_orchestrator_client()
                self.logger.info("✅ Orchestrator client initialized")
            except Exception as e:
                self.logger.warning(f"⚠️  Orchestrator client not available: {e}")

            increment_metric("service_initializations", "api_gateway", status="success")
            self.logger.info("✅ API Gateway Service initialized successfully")
            return True

        except Exception as e:
            increment_metric("service_initializations", "api_gateway", status="error")
            self.logger.error("Failed to initialize API Gateway Service", error=str(e))
            self.logger.exception(e)
            return False

    async def health_check(self) -> Dict:
        """Perform health check"""
        return {
            "status": "healthy",
            "service": "api-gateway",
            "version": "1.0.0"
        }

    async def shutdown(self) -> None:
        """Cleanup resources"""
        try:
            # ✅ Phase 4a: Use proper relative imports (no sys.path manipulation)
            from .routers.process import cleanup_orchestrator_client
            await cleanup_orchestrator_client()
            self.logger.info("✅ Orchestrator client cleaned up")
        except Exception as e:
            self.logger.warning(f"⚠️  Orchestrator cleanup warning: {e}")

        self.logger.info("🛑 API Gateway Service shutdown complete")

# For standalone execution
if __name__ == "__main__":
    import uvicorn
    import asyncio
    from fastapi import FastAPI
    # telemetry_middleware removed import add_telemetry_middleware

    # Create service with default config
    # Note: In standalone mode, we still use os.getenv for backward compatibility
    # When running via Service Manager, the port comes from SettingsService
    port = int(os.getenv("API_GATEWAY_PORT", "8010"))  # Dynamic allocation supported via PortPool
    config = {
        "name": "api_gateway",
        "port": port,
        "host": "0.0.0.0"
    }

    service = APIGatewayService(config)

    # Initialize
    async def init_service():
        await service.initialize()

    asyncio.run(init_service())

    # Create FastAPI app
    app = FastAPI(title="API Gateway Service")
    # add_telemetry_middleware removed, "api_gateway")

    # Add basic health endpoint for testing
    @app.get("/health")
    async def health():
        return await service.health_check()

    # Include service router (may be empty but that's OK for testing)
    app.include_router(service.get_router())

    # Run server
    print(f"Starting API Gateway Service on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
