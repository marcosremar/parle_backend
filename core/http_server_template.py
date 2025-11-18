#!/usr/bin/env python3
"""
Generic HTTP Server Template
Creates standalone FastAPI server for any service running in external mode

Usage:
    python http_server_template.py --service session --port 8800
    python http_server_template.py --service orchestrator --port 8900
"""

import sys
import os
import importlib
import logging
import argparse
import socket
import subprocess
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import telemetry middleware (may not be available in all environments)
try:
    from src.core.telemetry_middleware import add_telemetry_middleware
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def force_cleanup_port(port: int) -> bool:
    """
    Force cleanup of zombie sockets on a port using ss -K command.

    Args:
        port: Port number to cleanup

    Returns:
        True if cleanup attempted, False otherwise
    """
    try:
        # Use ss -K to forcefully close any sockets on this port
        # This handles orphaned/zombie sockets that lsof/fuser can't see
        cmd = f"ss -K '( dport = :{port} or sport = :{port} )'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)

        if result.returncode == 0:
            logger.info(f"✅ Port {port} cleanup attempted via ss -K")
            return True
        else:
            logger.debug(f"Port {port} cleanup returned code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"⚠️  Port {port} cleanup timed out")
        return False
    except Exception as e:
        logger.debug(f"Port {port} cleanup exception: {e}")
        return False


class ReuseAddrServer(uvicorn.Server):
    """
    Custom Uvicorn Server that enables SO_REUSEADDR socket option.

    This allows immediate port reuse after process restart, preventing
    "Address already in use" errors caused by TIME_WAIT socket states.

    Based on solution from: https://github.com/encode/uvicorn/discussions/1854
    """

    def install_signal_handlers(self):
        """Override to prevent signal handler conflicts when run as subprocess"""
        pass  # Skip signal handlers - parent process handles signals

    async def startup(self, sockets=None):
        """Override startup to enable SO_REUSEADDR + SO_REUSEPORT BEFORE binding"""
        # Create socket with SO_REUSEADDR + SO_REUSEPORT if not provided
        if sockets is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Enable SO_REUSEADDR to reuse TIME_WAIT sockets
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Enable SO_REUSEPORT to handle IPv4/IPv6 dual stack conflicts
            # This allows multiple sockets to bind to same port (critical for zombie cleanup)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                logger.info(f"✅ SO_REUSEPORT enabled (zombie socket protection)")
            except (AttributeError, OSError) as e:
                # SO_REUSEPORT might not be available on all platforms
                logger.warning(f"⚠️  SO_REUSEPORT not available: {e}")

            # Get config from server
            host = self.config.host
            port = self.config.port

            # Bind and listen on socket
            sock.bind((host, port))
            sock.listen(128)  # Standard listen backlog
            logger.info(f"✅ SO_REUSEADDR enabled, bound and listening on {host}:{port}")

            sockets = [sock]

        # Pass pre-configured socket to uvicorn
        await super().startup(sockets=sockets)


def create_http_server(service_name: str, module_path: str, port: int, execution_mode: str = "external") -> FastAPI:
    """
    Create standalone HTTP server for a service

    Args:
        service_name: Service identifier (e.g., "session", "orchestrator")
        module_path: Python module path (e.g., "src.services.session.service")
        port: Port number to run on
        execution_mode: Execution mode ("external" or "internal"/"module")

    Returns:
        FastAPI application instance
    """
    # Normalize execution mode - "module" is the same as "internal"
    if execution_mode == "module":
        execution_mode = "internal"

    app = FastAPI(
        title=f"{service_name.title()} Service",
        description=f"Ultravox {service_name} service ({execution_mode} mode)",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Telemetry middleware (automatic for all services)
    if TELEMETRY_AVAILABLE:
        add_telemetry_middleware(app, service_name)
        logger.info(f"📊 Telemetry middleware enabled for {service_name}")

    logger.info(f"🔧 Loading service: {service_name} from {module_path} (mode: {execution_mode})")

    try:
        # Dynamically import the service module
        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            logger.error(f"❌ Failed to import module {module_path}: {e}")
            import traceback
            logger.error(f"Import traceback:\n{traceback.format_exc()}")
            raise

        # Find the service class (should end with 'Service')
        service_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                attr_name.endswith('Service') and
                attr_name != 'BaseService'):
                service_class = attr
                break

        if not service_class:
            raise ValueError(f"No service class found in module: {module_path}")

        logger.info(f"✅ Found service class: {service_class.__name__}")

        # Instantiate the service with ServiceContext (DI v4.0)
        # Create lightweight context with SettingsService for proper .env and settings.yaml loading
        try:
            from src.core.context import ServiceContext
            from src.core.settings_service import SettingsService
            from loguru import logger as loguru_logger

            # Create SettingsService (loads .env and settings.yaml)
            settings = SettingsService.get_instance()

            # Create logger for service
            service_logger = loguru_logger.bind(service=service_name)

            # Create lightweight ServiceContext with only settings (no comm, minimal DI)
            service_context = ServiceContext(
                service_name=service_name,
                config={"port": port, "execution_mode": execution_mode},
                logger=service_logger,  # Use loguru logger
                comm=None,  # No communication manager (standalone service)
                gpu=None,  # No GPU manager
                metrics=None,  # No metrics manager
                telemetry=None,  # No telemetry
                settings=settings  # ✅ This is the key - inject SettingsService for .env reading!
            )

            service_instance = service_class(
                config={
                    "service_name": service_name,
                    "port": port,
                    "execution_mode": execution_mode,
                    "name": service_name
                },
                context=service_context  # DI v4.0 - inject settings, logger, etc.
            )
        except Exception as e:
            logger.error(f"❌ Failed to instantiate {service_class.__name__}: {e}")
            import traceback
            logger.error(f"Instantiation traceback:\n{traceback.format_exc()}")
            raise

        # Mount the service router
        app.include_router(
            service_instance.get_router(),
            tags=[service_name]
        )

        logger.info(f"📌 Mounted routes for {service_name}")

        # Add startup event to initialize service
        @app.on_event("startup")
        async def startup_event():
            try:
                logger.info(f"🚀 Initializing {service_name} service...")
                success = await service_instance.start()
                if success:
                    logger.info(f"✅ {service_name} service ready")
                else:
                    logger.error(f"❌ {service_name} service initialization failed")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"❌ FATAL ERROR during {service_name} startup: {e}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                sys.exit(1)

        # Add shutdown event
        @app.on_event("shutdown")
        async def shutdown_event():
            logger.info(f"🛑 Shutting down {service_name} service...")
            await service_instance.stop()

        # Health endpoint
        @app.get("/health")
        async def health():
            return await service_instance.health_check()

        # Info endpoint
        @app.get("/info")
        async def info():
            return service_instance.get_service_info()

        # Check if service has custom get_app() (e.g., Socket.IO services)
        # This must be done AFTER all FastAPI event handlers are registered
        if hasattr(service_instance, 'get_app') and callable(service_instance.get_app):
            logger.info(f"🔌 Service has custom app (Socket.IO), wrapping FastAPI app")
            # Pass the FastAPI app to get_app() so Socket.IO can wrap it
            wrapped_app = service_instance.get_app(app)
            logger.info(f"✅ Using custom Socket.IO wrapped app for {service_name}")
            return wrapped_app

        return app

    except Exception as e:
        logger.error(f"❌ Failed to create server for {service_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for HTTP server template"""
    parser = argparse.ArgumentParser(description="Generic HTTP Server for Ultravox Services")
    parser.add_argument("--service", required=True, help="Service name (e.g., session, orchestrator)")
    parser.add_argument("--port", type=int, required=True, help="Port to run on")
    parser.add_argument("--module", help="Module path (auto-detected if not provided)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--execution-mode", default="external", choices=["external", "internal", "module"],
                       help="Execution mode (external=standalone process with discovery, internal/module=no discovery registration)")

    args = parser.parse_args()

    # Auto-detect module path if not provided
    module_path = args.module or f"src.services.{args.service}.service"

    logger.info("=" * 60)
    logger.info(f"🎛️  Ultravox HTTP Server")
    logger.info("=" * 60)
    logger.info(f"Service: {args.service}")
    logger.info(f"Module: {module_path}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Execution Mode: {args.execution_mode}")
    logger.info("=" * 60)

    # Force cleanup of zombie sockets before binding
    # This handles orphaned sockets from previous crashes
    force_cleanup_port(args.port)

    # Create FastAPI app with execution mode
    app = create_http_server(args.service, module_path, args.port, args.execution_mode)

    # Create config for custom server with SO_REUSEADDR
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

    # Run server with SO_REUSEADDR + SO_REUSEPORT enabled
    server = ReuseAddrServer(config)
    server.run()


if __name__ == "__main__":
    main()
