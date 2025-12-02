"""
Base Service (v5.3)
Abstract base class for all services in the Ultravox pipeline

Services can run either:
- Module: In-process within Service Manager (mounted as FastAPI routes, zero overhead)
- Internal: Separate process, same machine (IPC communication)
- External: Remote machine (HTTP/gRPC communication)

Dependency Injection (v5.3):
- Services receive unified ServiceContext with all dependencies
- ServiceContext provides: logger, comm, gpu, metrics, telemetry, config
- Logger now uses unified logging system (src.core.core_logging)
- Replaces old 3-layer system (GlobalContext → ProcessContext → ServiceContext)

Logging Changes (v5.3):
- self.logger uses unified logging system with consistent formatting
- Automatic log rotation, retention, and compression
- Separate error logs and JSON logs for aggregators
- Legacy mode (without context) uses standard logging for compatibility

The Service Manager creates ServiceContext and injects it into services automatically.
"""

from abc import ABC, abstractmethod
from fastapi import APIRouter
from typing import Dict, Any, Optional
import logging
import aiohttp
import asyncio
import os

logger = logging.getLogger(__name__)

# OpenTelemetry - Auto-instrumentation (global setup)
_INSTRUMENTATION_INITIALIZED = False

def _initialize_instrumentation():
    """Initialize OpenTelemetry auto-instrumentation once globally"""
    global _INSTRUMENTATION_INITIALIZED

    if _INSTRUMENTATION_INITIALIZED:
        return

    try:
        from .observability import instrument_all
        instrument_all()
        _INSTRUMENTATION_INITIALIZED = True
        logger.info("✅ OpenTelemetry auto-instrumentation enabled globally")
    except ImportError:
        logger.debug("⚠️  OpenTelemetry not available, skipping auto-instrumentation")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize OpenTelemetry auto-instrumentation: {e}")


class BaseService(ABC):
    """
    Abstract base class for all Ultravox services

    Each service implements business logic and exposes FastAPI routes.
    The Service Manager handles:
    - Loading the service class
    - Deciding execution mode (internal vs external)
    - Mounting routes (internal) or starting HTTP server (external)
    - URL resolution via Communication Manager

    Dependency Injection (REQUIRED):
    - ALL services MUST receive ServiceContext in constructor
    - ServiceContext provides: logger, comm, gpu, metrics, telemetry, settings
    - Legacy mode (context=None) has been REMOVED
    - This ensures proper dependency management and testability

    Example:
        class MyService(BaseService):
            def __init__(self, context: ServiceContext, config: Optional[Dict] = None):
                super().__init__(context=context, config=config)
                # self.logger, self.comm, etc. are already injected
    """

    def __init__(self, context: 'ServiceContext', config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the service with Dependency Injection (DRY v5.2 - Enhanced with auto-logging)

        Args:
            context: ServiceContext with injected dependencies (REQUIRED)
            config: Optional service configuration dictionary (overrides context.config)

        Raises:
            ValueError: If context is None (DI is now mandatory)
        """
        # Initialize OpenTelemetry auto-instrumentation (once globally)
        _initialize_instrumentation()

        # ServiceContext is now REQUIRED (no more legacy mode)
        if context is None:
            raise ValueError(
                f"{self.__class__.__name__} requires ServiceContext for dependency injection. "
                f"Legacy mode has been removed. Please provide a valid ServiceContext instance."
            )

        # ServiceContext injection (unified v4.0)
        self.context = context

        # Use dependencies from context (all injected automatically)
        self.config = config or context.config
        self.logger = context.logger
        self.comm = context.comm
        self.gpu = context.gpu
        self.metrics = context.metrics
        self.telemetry = context.telemetry
        self.settings = context.settings

        # Enhanced auto-logging (DRY v5.2 - eliminates manual logging in 10+ services)
        service_name = self.__class__.__name__
        self.logger.info(
            f"✅ {service_name} initialized with ServiceContext (DI v4.0)",
            di_enabled=True,
            settings_available=bool(self.settings),
            comm_available=bool(self.comm),
            gpu_available=bool(self.gpu),
            metrics_available=bool(self.metrics),
            telemetry_available=bool(self.telemetry)
        )

        self.router = APIRouter()
        self.initialized = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._grpc_server = None  # gRPC server instance
        self._heartbeat_task: Optional[asyncio.Task] = None  # Heartbeat background task
        self._setup_router()

    @abstractmethod
    def _setup_router(self) -> None:
        """
        Setup FastAPI routes for this service

        Subclasses must implement this to register their endpoints using:
        @self.router.get/post/put/delete(...)
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize service resources (databases, connections, etc.)

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for this service

        Returns:
            Dict containing health status information
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Cleanup resources on service shutdown
        """
        pass

    def get_router(self) -> APIRouter:
        """
        Get the FastAPI router for this service

        Used by:
        - Service Manager (internal mode): Mounts routes in main FastAPI app
        - HTTP Server Template (external mode): Creates standalone server

        Returns:
            APIRouter instance with all service endpoints
        """
        return self.router

    def load_config_from_settings(
        self,
        config_class,
        default_factory=None
    ):
        """
        Load configuration from SettingsService with automatic fallback (DRY v5.2)

        Eliminates duplicated try/except blocks found in 18+ services.

        Before (duplicated in every service.py):
            if self.settings:
                try:
                    self.llm_config = LLMConfig.from_settings(self.settings)
                    self.logger.info("✅ LLM Configuration loaded via SettingsService")
                except Exception as e:
                    self.logger.error(f"Failed to load LLM config: {e}")
                    self.llm_config = LLMConfig()

        After (one line):
            self.llm_config = self.load_config_from_settings(LLMConfig)

        Args:
            config_class: Pydantic model class with from_settings() method
            default_factory: Factory function to create default config (optional)

        Returns:
            Config instance (loaded from SettingsService or default)

        Example:
            # Basic usage
            self.llm_config = self.load_config_from_settings(LLMConfig)

            # With custom default
            self.db_config = self.load_config_from_settings(
                DatabaseConfig,
                default_factory=lambda: DatabaseConfig(host="localhost")
            )
        """
        # No SettingsService available - use defaults
        if not self.settings:
            if default_factory:
                return default_factory()
            return config_class()

        # Try to load from SettingsService
        try:
            config = config_class.from_settings(self.settings)
            self.logger.info(f"✅ {config_class.__name__} loaded via SettingsService")
            return config
        except Exception as e:
            self.logger.warning(
                f"Failed to load {config_class.__name__} from SettingsService: {e}. Using defaults."
            )
            if default_factory:
                return default_factory()
            return config_class()

    def get_standard_routes(self) -> APIRouter:
        """
        Get standard routes (/health, /info, /validate) for all services.

        This method creates a router with standard endpoints that are common
        across all services, eliminating duplication.

        Returns:
            APIRouter with standard endpoints

        Example:
            def _setup_router(self):
                # Include standard routes
                self.router.include_router(self.get_standard_routes())

                # Add service-specific routes
                from .routes import create_router
                self.router.include_router(create_router(self))
        """
        from src.core.route_helpers import get_standard_router

        service_name = self.config.get('name', self.__class__.__name__.replace('Service', '').lower())
        return get_standard_router(self, service_name)

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information with auto-discovery.

        Auto-generates service metadata including:
        - Service name and version
        - Description
        - Endpoints
        - Configuration

        Returns:
            Dict with service information
        """
        service_name = self.config.get('name', self.__class__.__name__.replace('Service', '').lower())

        # Auto-discover endpoints from router
        endpoints = {}
        for route in self.router.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                for method in route.methods:
                    endpoints.setdefault(method, []).append(route.path)

        return {
            "service": service_name,
            "version": self.config.get("version", "1.0.0"),
            "description": self.__doc__ or "No description",
            "status": "running" if self.initialized else "initializing",
            "endpoints": endpoints,
            "config": {
                "port": self.config.get("port"),
                "host": self.config.get("host", "0.0.0.0"),
            },
        }

    def create_app(self) -> Any:
        """
        Create a FastAPI app with telemetry middleware for standalone execution

        This method automatically:
        - Creates a FastAPI app instance
        - Adds telemetry middleware for request tracking
        - Includes the service's router
        - Adds telemetry JSON API endpoints

        Used by standalone service launchers (app.py, main.py files)

        Returns:
            FastAPI app instance ready to run
        """
        try:
            from fastapi import FastAPI
            from src.core.telemetry_middleware import add_telemetry_middleware
            from src.core.telemetry_router import create_telemetry_router

            # Get service name from config or class name
            service_name = self.config.get('name', self.__class__.__name__.replace('Service', '').lower())

            # Create FastAPI app
            app = FastAPI(
                title=f"{service_name.title()} Service",
                version="1.0.0"
            )

            # Add telemetry middleware
            add_telemetry_middleware(app, service_name)

            # Include telemetry API router (JSON endpoints)
            telemetry_router = create_telemetry_router()
            app.include_router(telemetry_router)

            # Include service router
            app.include_router(self.router)

            logger.info(f"✅ FastAPI app created for {service_name} with telemetry middleware + JSON API")

            return app

        except Exception as e:
            logger.error(f"❌ Failed to create FastAPI app: {e}")
            raise

    @property
    def app(self) -> Any:
        """
        Property to access FastAPI app (creates it if needed)

        This allows services to use service.app directly without calling create_app()
        The app is created on first access and cached for subsequent calls.
        """
        if not hasattr(self, '_app_instance'):
            self._app_instance = self.create_app()
        return self._app_instance

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information for display/monitoring

        Returns:
            Dict with service metadata
        """
        return {
            "name": self.__class__.__name__,
            "initialized": self.initialized,
            "config": self.config
        }

    async def start(self) -> bool:
        """
        Start the service (initialize and mark as ready)

        This automatically handles:
        1. Communication Manager initialization
        2. Proto generation & compilation (for external services)
        3. Service registration (for internal services)
        4. Service initialization (databases, models, etc.)
        5. gRPC server startup (for external services)

        Returns:
            bool: True if started successfully
        """
        try:
            # 1. Initialize Communication Manager
            await self._initialize_communication_manager()

            # 2. Register with Communication Manager (for internal services)
            if self.config.get('execution_mode') == 'internal':
                await self._register_with_communication_manager()

            # 3. Initialize service-specific resources
            success = await self.initialize()
            if success:
                self.initialized = True
                logger.info(f"✅ Service {self.__class__.__name__} started")

                # 4. Auto-register with Service Discovery (ONLY for external services)
                # Internal services are registered directly by Service Manager
                if self.config.get('execution_mode') == 'external':
                    await self._register_with_discovery()

                # 5. Start heartbeat task (ONLY for external services)
                # Internal services don't need heartbeat (they're in-process)
                if self.config.get('execution_mode') == 'external':
                    await self._start_heartbeat()

                # 6. Auto-start gRPC server (ONLY for external services)
                # NOTE: gRPC server runs but does NOT auto-generate proto files
                # If you need gRPC: write .proto manually and compile with grpc_tools
                if self.config.get('execution_mode') == 'external':
                    await self._start_grpc_server()
            else:
                logger.error(f"❌ Failed to initialize {self.__class__.__name__}")
            return success
        except Exception as e:
            logger.error(f"❌ Error starting {self.__class__.__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def stop(self) -> None:
        """Stop the service and cleanup resources"""
        try:
            # Stop heartbeat first
            await self._stop_heartbeat()

            # Stop gRPC server
            await self._stop_grpc_server()

            # Then stop service-specific resources
            await self.shutdown()

            # Finally cleanup Communication Manager
            await self._cleanup_communication_manager()

            self.initialized = False
            logger.info(f"🛑 Service {self.__class__.__name__} stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping {self.__class__.__name__}: {e}")

    async def _initialize_communication_manager(self) -> None:
        """
        Initialize Communication Manager for inter-service communication

        NOTE: With DI, Communication Manager should already be injected via ServiceContext.
        This method is kept for backward compatibility but does nothing if comm is already set.
        """
        # Communication Manager should already be injected via ServiceContext
        if self.comm is not None:
            logger.debug(
                f"✅ Communication Manager already injected via DI for {self.__class__.__name__}"
            )
            return

        # This should not happen with proper DI
        logger.warning(
            f"⚠️ Communication Manager not injected for {self.__class__.__name__}. "
            f"This indicates a problem with dependency injection setup."
        )
        raise RuntimeError(
            f"Communication Manager missing for {self.__class__.__name__}. "
            f"Ensure ServiceContext is properly configured with a Communication Manager instance."
        )

    async def _cleanup_communication_manager(self) -> None:
        """Cleanup Communication Manager resources"""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                logger.debug(f"🛑 aiohttp session closed for {self.__class__.__name__}")
        except Exception as e:
            logger.warning(f"⚠️ Error closing aiohttp session: {e}")

    async def _register_with_communication_manager(self) -> None:
        """
        Register this service with Communication Manager for direct calls

        Only runs for internal services. Enables zero-overhead direct function calls.
        """
        try:
            if self.comm:
                service_name = self.config.get('service_name', self.__class__.__name__.replace('Service', '').lower())
                self.comm.register_internal_service(service_name, self)
                logger.info(f"✅ Registered {service_name} for direct calls (zero overhead)")
        except Exception as e:
            logger.warning(f"⚠️  Failed to register with Communication Manager: {e}")

    def _get_grpc_port(self) -> int:
        """
        Get gRPC port for this service

        Calculates gRPC port from HTTP port using the formula:
        grpc_port = 50000 + (http_port % 1000)

        Returns:
            int: gRPC port number

        Examples:
            HTTP 8099 (STT) -> gRPC 50099
            HTTP 8100 (LLM) -> gRPC 50100
            HTTP 8101 (TTS) -> gRPC 50101
        """
        # Get HTTP port from config
        http_port = self.config.get('port', 8000)

        # Calculate gRPC port
        grpc_port = 50000 + (http_port % 1000)

        return grpc_port

    async def _start_grpc_server(self) -> None:
        """
        Automatically start gRPC server for this service

        Only starts if ENABLE_GRPC=true environment variable is set.
        Uses GenericGrpcServicer to route gRPC calls to FastAPI endpoints.

        This makes the service available via both HTTP and gRPC!
        """
        # Check if gRPC is enabled (default: true, HTTP is backup)
        if os.getenv("ENABLE_GRPC", "true").lower() != "true":
            logger.debug(f"⏭️  gRPC disabled for {self.__class__.__name__} (set ENABLE_GRPC=false to disable)")
            return

        try:
            from src.core.grpc_server.server import start_generic_grpc_server

            grpc_port = self._get_grpc_port()
            service_name = self.__class__.__name__

            logger.info(f"🚀 Starting gRPC server for {service_name} on port {grpc_port}...")

            # Start generic gRPC server
            self._grpc_server = await start_generic_grpc_server(
                service=self,
                port=grpc_port,
                service_name=service_name
            )

            if self._grpc_server:
                logger.info(f"✅ gRPC server started for {service_name} on port {grpc_port}")
                logger.info(f"   📡 HTTP port: {self.config.get('port', '?')}")
                logger.info(f"   📡 gRPC port: {grpc_port}")
            else:
                logger.warning(f"⚠️ Failed to start gRPC server for {service_name}")

        except ImportError as e:
            logger.warning(f"⚠️ gRPC dependencies not available: {e}")
        except Exception as e:
            logger.error(f"❌ Error starting gRPC server for {self.__class__.__name__}: {e}")

    async def _stop_grpc_server(self) -> None:
        """Stop gRPC server if running"""
        if self._grpc_server:
            try:
                from src.core.grpc_server.server import stop_generic_grpc_server

                service_name = self.__class__.__name__
                await stop_generic_grpc_server(service_name)
                self._grpc_server = None
                logger.debug(f"🛑 gRPC server stopped for {service_name}")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping gRPC server: {e}")

    async def _register_with_discovery(self) -> None:
        """Register this service with Service Discovery Registry"""
        try:
            # Extract service info
            service_name = self.config.get('name', self.__class__.__name__.lower())
            service_port = self.config.get('port', 8000)
            service_host = self.config.get('host', '127.0.0.1')

            # Get all endpoints from router
            endpoints = []
            for route in self.router.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    for method in route.methods:
                        endpoints.append({
                            'path': route.path,
                            'method': method,
                            'description': route.name or ''
                        })

            # Get supported protocols
            protocols = ['http']  # Default
            if self._grpc_server:
                protocols.append('grpc')

            # Get service metadata
            metadata = self.get_service_metadata() if hasattr(self, 'get_service_metadata') else {}

            # Register via HTTP (Service Manager should be running)
            service_manager_url = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{service_manager_url}/discovery/register",
                    json={
                        'name': service_name,
                        'host': service_host,
                        'port': service_port,
                        'endpoints': endpoints,
                        'protocols': protocols,
                        'metadata': metadata,
                        'version': '1.0.0'
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        logger.info(f"📝 Service registered with Discovery: {service_name} (id: {result.get('service_id')})")
                    else:
                        logger.warning(f"⚠️ Failed to register with Discovery: HTTP {response.status}")

        except Exception as e:
            logger.warning(f"⚠️ Could not register with Service Discovery: {e}")
            # Non-critical error, service can still function

    def get_service_metadata(self) -> dict:
        """
        Get service metadata for discovery registration
        Can be overridden by subclasses to provide custom metadata
        """
        return {
            'class': self.__class__.__name__,
            'execution_mode': self.config.get('execution_mode', 'unknown')
        }

    async def _start_heartbeat(self):
        """Start heartbeat background task"""
        heartbeat_interval = int(os.getenv('SERVICE_HEARTBEAT_INTERVAL', '15'))  # seconds
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(heartbeat_interval))
        logger.debug(f"💓 Heartbeat task started (interval: {heartbeat_interval}s)")

    async def _heartbeat_loop(self, interval: int):
        """Background task to send periodic heartbeats to Service Discovery"""
        service_name = self.config.get('name', self.__class__.__name__.lower())
        service_manager_url = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")

        while True:
            try:
                await asyncio.sleep(interval)

                # Determine current status
                health = await self.health_check()
                status = 'healthy' if health.get('status') == 'healthy' else 'degraded'

                # Send heartbeat
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{service_manager_url}/discovery/heartbeat",
                        json={'name': service_name, 'status': status},
                        timeout=aiohttp.ClientTimeout(total=3)
                    ) as response:
                        if response.status == 200:
                            logger.debug(f"💓 Heartbeat sent: {service_name} ({status})")
                        else:
                            logger.warning(f"⚠️ Heartbeat failed: HTTP {response.status}")

            except asyncio.CancelledError:
                logger.debug(f"💓 Heartbeat task cancelled for {service_name}")
                break
            except Exception as e:
                logger.debug(f"⚠️ Heartbeat error for {service_name}: {e}")
                # Continue loop despite errors

    async def _stop_heartbeat(self):
        """Stop heartbeat background task"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.debug("💓 Heartbeat task stopped")
