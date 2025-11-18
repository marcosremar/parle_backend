"""
HTTP Server Manager - Handles FastAPI application setup and lifecycle.

Part of Service Manager refactoring (Phase 3).
Manages HTTP server setup, router registration, and lifespan events.
"""

from typing import Optional, Any, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from datetime import datetime
from pathlib import Path
import os
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class HTTPServerManager:
    """
    Manages HTTP server setup - FastAPI app configuration and routers.

    Responsibilities:
    - Create and configure FastAPI application
    - Register all API routers
    - Manage lifespan events (startup/shutdown)
    - Coordinate with ServiceManagerFacade for service operations

    SOLID Principles:
    - Single Responsibility: Only handles HTTP server setup
    - Open/Closed: Easy to add new routers without changing core logic
    - Dependency Inversion: Depends on ServiceManagerFacade abstraction

    Architecture:
    ```
    HTTPServerManager
        ├── FastAPI app creation
        ├── Router registration (10+ routers)
        └── Lifespan management (startup/shutdown hooks)
    ```

    Example:
        facade = ServiceManagerFacade()
        http_manager = HTTPServerManager(facade)
        app = http_manager.create_app()

        # Run with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8888)
    """

    def __init__(self, facade: Optional[Any] = None):
        """
        Initialize HTTP Server Manager.

        Args:
            facade: ServiceManagerFacade instance (optional, can be set later)
        """
        self.facade = facade
        self.app: Optional[FastAPI] = None
        self._routers_registered = False

        logger.info("🌐 HTTP Server Manager initialized")

    # ============================================================================
    # App Creation and Configuration
    # ============================================================================

    def create_app(self) -> FastAPI:
        """
        Create and configure FastAPI application.

        Returns:
            Configured FastAPI app instance

        Example:
            http_manager = HTTPServerManager(facade)
            app = http_manager.create_app()
        """
        if self.app:
            logger.warning("⚠️  FastAPI app already created, returning existing instance")
            return self.app

        logger.info("🌐 Creating FastAPI application...")

        # Create app with lifespan
        app = FastAPI(
            title="Ultravox Service Manager API",
            description="API para gerenciar todos os serviços do Ultravox Pipeline",
            version="1.0.0",
            lifespan=self._create_lifespan()
        )

        self.app = app
        logger.info("✅ FastAPI application created")

        # Register routers
        self._register_all_routers(app)

        return app

    def _create_lifespan(self):
        """
        Create lifespan context manager for FastAPI.

        Returns:
            Async context manager for lifespan events
        """
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Manage application lifecycle with systemd integration"""
            # Startup
            await self._startup(app)
            yield
            # Shutdown
            await self._shutdown(app)

        return lifespan

    # ============================================================================
    # Lifespan Events (Startup/Shutdown)
    # ============================================================================

    async def _startup(self, app: FastAPI) -> None:
        """
        Execute startup logic.

        Args:
            app: FastAPI application instance
        """
        from src.core.service_manager.systemd_watchdog import get_watchdog

        watchdog = get_watchdog()
        watchdog.initialize()

        port = int(os.getenv("SERVICE_MANAGER_PORT", "8888"))
        watchdog.notify_status("Starting Service Manager...")

        logger.info(f"🚀 Starting Service Manager on port {port} (v1.0.0)")

        # Start watchdog heartbeat
        await watchdog.start()

        # GPU Memory Pre-flight Check
        await self._gpu_preflight_check()

        # Load active profile
        await self._load_active_profile()

        # Initialize DI Registry
        await self._initialize_di_registry(app)

        # Auto-start internal services
        await self._auto_start_services(app)

        logger.info("✅ Service Manager startup complete")

    async def _shutdown(self, app: FastAPI) -> None:
        """
        Execute shutdown logic.

        Args:
            app: FastAPI application instance
        """
        logger.info("⏹️  Shutting down Service Manager...")

        # Stop health monitoring
        if self.facade and hasattr(self.facade, 'health_monitor'):
            await self.facade.health_monitor.stop()

        # Cleanup resources
        if self.facade and hasattr(self.facade, 'cleanup'):
            self.facade.cleanup()

        logger.info("✅ Service Manager shutdown complete")

    # ============================================================================
    # Startup Helpers
    # ============================================================================

    async def _gpu_preflight_check(self) -> None:
        """Perform GPU memory pre-flight checks and cleanup."""
        logger.info("🔍 Performing GPU memory pre-flight checks...")
        try:
            from src.core.gpu_memory_manager import get_gpu_manager

            gpu_manager = get_gpu_manager()
            gpu_info = gpu_manager.get_gpu_info()

            if gpu_info:
                logger.info(f"📊 GPU Status:")
                logger.info(f"   Device: {gpu_info.device_name}")
                logger.info(f"   Total Memory: {gpu_info.total_mb}MB")
                logger.info(f"   Used Memory: {gpu_info.used_mb}MB")
                logger.info(f"   Free Memory: {gpu_info.free_mb}MB")

                # Clean GPU memory before starting services
                logger.info("🧹 Cleaning GPU memory before service startup...")
                cleanup_success = gpu_manager.cleanup_gpu_memory(level="soft")

                if cleanup_success:
                    gpu_info_after = gpu_manager.get_gpu_info()
                    if gpu_info_after:
                        freed_mb = gpu_info_after.free_mb - gpu_info.free_mb
                        logger.info(f"✅ GPU cleanup completed - freed {freed_mb}MB")
                        logger.info(f"   Free Memory Now: {gpu_info_after.free_mb}MB")
                else:
                    logger.warning("⚠️  GPU cleanup had issues, continuing anyway...")
            else:
                logger.info("ℹ️  No GPU detected or CUDA not available")
        except Exception as e:
            logger.warning(f"⚠️  GPU pre-flight check failed: {e}")
            logger.info("   Continuing with service startup...")

    async def _load_active_profile(self) -> None:
        """Load active execution profile."""
        logger.info("🎯 Loading active profile...")
        from src.core.managers.profile_manager import get_profile_manager

        profile_manager = get_profile_manager()
        active_profile = profile_manager.get_active_profile()

        if active_profile:
            logger.info(f"✅ Active profile: {active_profile.name} - {active_profile.description}")
            logger.info(f"   Enabled services: {len(active_profile.enabled_services)}")
        else:
            logger.warning("⚠️  No active profile, using default configuration")

    async def _initialize_di_registry(self, app: FastAPI) -> None:
        """Initialize Dependency Injection Registry."""
        if not self.facade:
            logger.warning("⚠️  No facade provided, skipping DI registry initialization")
            return

        # Check if facade has DI registry
        if not hasattr(self.facade, 'di_registry') or not self.facade.di_registry:
            logger.info("ℹ️  No DI registry in facade, skipping initialization")
            return

        logger.info("🔧 Initializing DI Registry...")
        from src.core.managers.profile_manager import get_profile_manager

        profile_manager = get_profile_manager()
        active_profile = profile_manager.get_active_profile()
        profile_name = active_profile.name if active_profile else "development"

        await self.facade.di_registry.initialize(profile_name=profile_name)
        logger.info("✅ DI Registry initialized with GlobalContext + ProcessContext")

    async def _auto_start_services(self, app: FastAPI) -> None:
        """Auto-start internal services based on profile."""
        if not self.facade:
            logger.warning("⚠️  No facade provided, skipping auto-start")
            return

        logger.info("🔧 Auto-starting internal services...")

        # Store app reference in facade
        if hasattr(self.facade, 'app'):
            self.facade.app = app
            logger.info("📱 Stored FastAPI app reference in facade")

        # TODO: Implement auto-start logic
        # - Get services from execution config
        # - Filter by profile
        # - Start services in order
        logger.info("ℹ️  Auto-start logic not yet implemented")

    # ============================================================================
    # Router Registration
    # ============================================================================

    def _register_all_routers(self, app: FastAPI) -> None:
        """
        Register all API routers.

        Args:
            app: FastAPI application instance
        """
        if self._routers_registered:
            logger.warning("⚠️  Routers already registered, skipping")
            return

        logger.info("📡 Registering API routers...")

        # Core service management routers (extracted modules)
        self._register_core_routers(app)

        # Additional feature routers
        self._register_feature_routers(app)

        # System and monitoring routers
        self._register_system_routers(app)

        self._routers_registered = True
        logger.info("✅ All routers registered successfully")

    def _register_core_routers(self, app: FastAPI) -> None:
        """Register core service management routers."""
        try:
            from src.core.service_manager.endpoints import (
                services_router,
                gpu_router,
                info_router,
                pipeline_router,
                config_router,
                create_venv_router,
                set_manager,
                set_info_manager,
                set_pipeline_manager,
                set_config_manager
            )

            # Configure endpoints with facade
            if self.facade:
                set_manager(self.facade)
                set_info_manager(self.facade)
                set_pipeline_manager(self.facade)
                set_config_manager(self.facade)

            # Register routers
            app.include_router(services_router)
            app.include_router(gpu_router)
            app.include_router(info_router)
            app.include_router(pipeline_router)
            app.include_router(config_router)

            # Venv router (needs facade instance)
            if self.facade:
                venv_router = create_venv_router(self.facade)
                app.include_router(venv_router)

            logger.info("✅ Core routers registered (services, gpu, info, pipeline, config, venv)")
        except Exception as e:
            logger.error(f"❌ Failed to register core routers: {e}")

    def _register_feature_routers(self, app: FastAPI) -> None:
        """Register additional feature routers."""
        # Service Discovery
        try:
            from src.core.service_manager.endpoints.discovery import router as discovery_router
            app.include_router(discovery_router)
            logger.info("✅ Discovery router registered")
        except ImportError:
            logger.warning("⚠️  Discovery endpoints not found")

        # Profile Management
        try:
            from src.core.service_manager.profile_endpoints import setup_profile_endpoints
            if self.facade:
                setup_profile_endpoints(app, self.facade)
            logger.info("✅ Profile endpoints registered")
        except ImportError:
            logger.warning("⚠️  Profile endpoints not found")

        # Benchmark
        try:
            from src.services.api_gateway.routers.benchmark import router as benchmark_router
            app.include_router(benchmark_router)
            logger.info("✅ Benchmark router registered")
        except ImportError:
            logger.warning("⚠️  Benchmark endpoints not found")

    def _register_system_routers(self, app: FastAPI) -> None:
        """Register system and monitoring routers."""
        # System Management
        try:
            from src.core.service_manager.system_endpoints import router as system_router, set_service_manager
            if self.facade:
                set_service_manager(self.facade)
            app.include_router(system_router)
            logger.info("✅ System management router registered")
        except ImportError:
            logger.warning("⚠️  System endpoints not found")

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def set_facade(self, facade: Any) -> None:
        """
        Set ServiceManagerFacade instance.

        Args:
            facade: ServiceManagerFacade instance
        """
        self.facade = facade
        logger.info("🔗 ServiceManagerFacade instance set")

    def get_app(self) -> Optional[FastAPI]:
        """
        Get FastAPI application instance.

        Returns:
            FastAPI app if created, None otherwise
        """
        return self.app

    def __repr__(self) -> str:
        """String representation."""
        app_status = "created" if self.app else "not created"
        routers_status = "registered" if self._routers_registered else "not registered"
        return f"HTTPServerManager(app={app_status}, routers={routers_status})"
