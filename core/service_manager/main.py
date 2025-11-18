#!/usr/bin/env python3
"""
Service Manager API - Refactored using ServiceManagerFacade

This is the simplified main entry point that orchestrates all service management
through the ServiceManagerFacade pattern.

BEFORE: 3,456 lines (God Class anti-pattern)
AFTER: ~500 lines (Clean Architecture)
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import asyncio
import atexit

# Authorization check - Service Manager must be started via official script
if __name__ == "__main__" and not os.getenv("ULTRAVOX_SM_AUTHORIZED"):
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    print(f"{RED}❌ ERROR: Service Manager must be started using the official script{RESET}")
    print(f"{YELLOW}➜  Use: ./start_service_manager.sh start{RESET}")
    print(f"{BLUE}   This ensures proper process management, logging, and PID tracking{RESET}")
    sys.exit(1)

# FastAPI and HTTP server
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration and logging
from src.core.configurations.config_manager import ConfigurationManager
_ = ConfigurationManager()  # Load .env file

# Import unified logging system (v5.3)
from src.core.core_logging import setup_logging

# Import ServiceManagerFacade (the star of the show!)
from src.core.service_manager.managers.service_manager_facade import ServiceManagerFacade

# Import routers (already extracted in previous refactoring)
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
    set_config_manager,
)

# Setup logging
logger = setup_logging("service-manager", level="INFO")

# Colors for console output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

# Global facade instance
facade: ServiceManagerFacade = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle with systemd integration.

    Startup:
    - Initialize ServiceManagerFacade
    - Run configuration validation
    - Start systemd watchdog
    - Auto-start internal services

    Shutdown:
    - Stop all services gracefully
    - Clean up resources
    """
    # ============================================================================
    # STARTUP
    # ============================================================================

    # Import systemd watchdog
    from src.core.service_manager.systemd_watchdog import get_watchdog
    watchdog = get_watchdog()
    watchdog.initialize()

    port = int(os.getenv("SERVICE_MANAGER_PORT", "8888"))
    watchdog.notify_status("Starting Service Manager...")

    logger.info(f"🚀 Starting Service Manager on port {port} (Facade Pattern)")

    # Start watchdog heartbeat
    await watchdog.start()

    # GPU Memory Pre-flight Check
    logger.info("🔍 Performing GPU memory pre-flight checks...")
    global facade
    try:
        gpu_memory = facade.check_gpu_memory()
        if gpu_memory:
            logger.info(f"📊 GPU Status: {gpu_memory} MB free")

            # Optional: Clean GPU memory before starting services
            try:
                from src.core.gpu_memory_manager import get_gpu_manager
                gpu_manager = get_gpu_manager()
                cleanup_success = gpu_manager.cleanup_gpu_memory(level="soft")
                if cleanup_success:
                    logger.info("✅ GPU cleanup completed")
            except Exception as e:
                logger.warning(f"⚠️  GPU cleanup failed: {e}")
        else:
            logger.info("ℹ️  No GPU detected or CUDA not available")
    except Exception as e:
        logger.warning(f"⚠️  GPU pre-flight check failed: {e}")

    # Load active profile
    logger.info("🎯 Loading active profile...")
    from src.core.managers.profile_manager import get_profile_manager
    profile_manager = get_profile_manager()
    active_profile = profile_manager.get_active_profile()

    if active_profile:
        logger.info(f"✅ Active profile: {active_profile.name} - {active_profile.description}")
        logger.info(f"   Enabled services: {len(active_profile.enabled_services)}")
    else:
        logger.warning("⚠️  No active profile, using default configuration")

    # Initialize DI Registry (if available)
    try:
        from src.core.service_manager.di_registry import ServiceRegistryWithDI
        services_config_path = project_root / "services_config.yaml"
        if services_config_path.exists():
            di_registry = ServiceRegistryWithDI(services_config_path)
            profile_name = active_profile.name if active_profile else "development"
            await di_registry.initialize(profile_name=profile_name)
            logger.info("✅ DI Registry initialized with GlobalContext + ProcessContext")
    except Exception as e:
        logger.warning(f"⚠️  DI Registry initialization failed: {e}")

    # Auto-start internal services (if configured)
    logger.info("🔧 Auto-start internal services...")
    # TODO: Implement auto-start logic using facade.start_all_services()

    # Store app reference for internal service router registration
    facade.app = app

    logger.info("✅ Service Manager started successfully")
    watchdog.notify_ready()

    yield

    # ============================================================================
    # SHUTDOWN
    # ============================================================================

    logger.info("🛑 Shutting down Service Manager...")
    watchdog.notify_stopping()

    # Stop all services gracefully
    try:
        stop_result = await facade.stop_all_services()
        logger.info(f"✅ Stopped {stop_result.get('stopped', 0)} services")
    except Exception as e:
        logger.error(f"❌ Error stopping services: {e}")

    # Clean up resources
    facade.cleanup()

    watchdog.stop()
    logger.info("✅ Service Manager shutdown complete")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Ultravox Service Manager API",
    description="API para gerenciar todos os serviços do Ultravox Pipeline (Facade Pattern)",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# Basic Endpoints
# ============================================================================

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "service": "Ultravox Service Manager API",
        "version": "2.0.0",
        "architecture": "Facade Pattern (Refactored)",
        "endpoints": {
            "GET /health": "Health check",
            "GET /services": "List all services status",
            "GET /services/{service_id}": "Get service status",
            "POST /services/{service_id}/start": "Start a service",
            "POST /services/{service_id}/stop": "Stop a service",
            "POST /services/{service_id}/restart": "Restart a service",
            "POST /services/start-all": "Start all services",
            "POST /services/stop-all": "Stop all services",
            "GET /system": "Get system information"
        }
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.

    Returns detailed health information including:
    - Service Manager readiness state
    - Service status summary
    - System uptime and memory usage
    - Configuration validation status

    This endpoint can be used by:
    - Load balancers for health checking
    - Monitoring systems (Prometheus, Grafana)
    - CLI tools (./main.sh) to verify Service Manager is ready
    - Integration tests
    """
    try:
        # Calculate uptime
        uptime_seconds = 0
        if hasattr(facade, 'start_time'):
            uptime_seconds = (datetime.now() - facade.start_time).total_seconds()

        # Check if startup is complete
        startup_complete = not os.getenv("STARTUP_MODE")

        # Get service status summary
        all_status = facade.get_all_status()
        service_summary = {
            "total": len(all_status),
            "running": sum(1 for s in all_status.values() if s.get("status") == "running"),
            "healthy": sum(1 for s in all_status.values() if s.get("healthy", False)),
            "stopped": sum(1 for s in all_status.values() if s.get("status") == "stopped"),
        }

        # Get system memory info
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
        except Exception:
            memory_mb = 0

        # Determine overall status
        if not startup_complete:
            overall_status = "starting"
        elif service_summary["healthy"] >= service_summary["total"] * 0.8:
            overall_status = "healthy"
        elif service_summary["running"] > 0:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return {
            "status": overall_status,
            "startup_complete": startup_complete,
            "uptime_seconds": int(uptime_seconds),
            "services": service_summary,
            "memory_mb": round(memory_mb, 2),
            "timestamp": datetime.now().isoformat(),
            "architecture": "Facade Pattern"
        }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Main Function
# ============================================================================

def main() -> Dict[str, Any]:
    """
    Main entry point with singleton pattern and PID file management.

    Returns:
        Dict with startup information
    """
    import fcntl

    # PID file location
    pid_dir = Path.home() / ".cache" / "ultravox-pipeline"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = str(pid_dir / "service_manager.pid")

    # PID file management (singleton pattern)
    try:
        # Try to acquire exclusive lock on PID file
        pid_fd = os.open(pid_file, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o644)

        # Write current PID
        os.write(pid_fd, str(os.getpid()).encode())
        os.close(pid_fd)

        # Register cleanup function
        def cleanup_pid_file():
            try:
                os.unlink(pid_file)
                logger.info("Cleaned up PID file")
            except Exception as e:
                logger.debug(f"Could not cleanup PID file: {e}")

        atexit.register(cleanup_pid_file)

        logger.info(f"Service Manager starting with PID {os.getpid()}")

    except OSError:
        # PID file exists, check if process is still running
        try:
            with open(pid_file, 'r') as f:
                old_pid = int(f.read().strip())

            # Check if process is still alive
            try:
                os.kill(old_pid, 0)
                print(f"{RED}ERROR: Service Manager already running with PID {old_pid}{RESET}")
                print(f"{YELLOW}Please stop the existing instance first with: kill -TERM {old_pid}{RESET}")
                sys.exit(1)
            except ProcessLookupError:
                # Process is dead, remove stale PID file
                print(f"{YELLOW}Removing stale PID file (old PID {old_pid} not running){RESET}")
                os.unlink(pid_file)

                # Retry creating PID file
                pid_fd = os.open(pid_file, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o644)
                os.write(pid_fd, str(os.getpid()).encode())
                os.close(pid_fd)

                def cleanup_pid_file():
                    try:
                        os.unlink(pid_file)
                        logger.info("Cleaned up PID file")
                    except Exception as e:
                        logger.debug(f"Could not cleanup PID file: {e}")

                atexit.register(cleanup_pid_file)

        except Exception as e:
            print(f"{RED}Error checking PID file: {e}{RESET}")
            sys.exit(1)

    # ============================================================================
    # Initialize ServiceManagerFacade
    # ============================================================================

    global facade
    logger.info("🏗️  Initializing Service Manager Facade...")
    facade = ServiceManagerFacade()
    logger.info(f"✅ Facade initialized - managing {len(facade.get_all_status())} services")

    # Run configuration validation
    logger.info("🔍 Validating configuration...")
    validation_result = facade.validate_configuration_consistency()
    if validation_result["status"] == "ok":
        logger.info("✅ Configuration validation passed")
    else:
        logger.warning(f"⚠️  Configuration has {len(validation_result['issues'])} issues")

    # ============================================================================
    # Register Routers
    # ============================================================================

    # Configure extracted endpoints with facade instance
    set_manager(facade)
    set_info_manager(facade)
    set_pipeline_manager(facade)
    set_config_manager(facade)

    # Create venv router and register it
    venv_router = create_venv_router(facade)

    # Include all routers
    logger.info("📱 Registering API routers...")

    try:
        app.include_router(services_router, prefix="/services", tags=["Services"])
        logger.info("  ✅ Services router registered")
    except Exception as e:
        logger.warning(f"  ⚠️  Services router failed: {e}")

    try:
        app.include_router(gpu_router, prefix="/gpu", tags=["GPU"])
        logger.info("  ✅ GPU router registered")
    except Exception as e:
        logger.warning(f"  ⚠️  GPU router failed: {e}")

    try:
        app.include_router(info_router, prefix="/info", tags=["Info"])
        logger.info("  ✅ Info router registered")
    except Exception as e:
        logger.warning(f"  ⚠️  Info router failed: {e}")

    try:
        app.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])
        logger.info("  ✅ Pipeline router registered")
    except Exception as e:
        logger.warning(f"  ⚠️  Pipeline router failed: {e}")

    try:
        app.include_router(config_router, prefix="/config", tags=["Config"])
        logger.info("  ✅ Config router registered")
    except Exception as e:
        logger.warning(f"  ⚠️  Config router failed: {e}")

    try:
        app.include_router(venv_router, prefix="/venv", tags=["Virtual Environments"])
        logger.info("  ✅ Venv router registered")
    except Exception as e:
        logger.warning(f"  ⚠️  Venv router failed: {e}")

    # Additional routers (optional)
    try:
        from src.core.service_manager.system_endpoints import router as system_router, set_service_manager
        set_service_manager(facade)
        app.include_router(system_router, prefix="/system", tags=["System"])
        logger.info("  ✅ System router registered")
    except ImportError:
        logger.debug("  ℹ️  System router not available")
    except Exception as e:
        logger.warning(f"  ⚠️  System router failed: {e}")

    try:
        from src.services.api_gateway.routers.profiles import router as profiles_router
        app.include_router(profiles_router, prefix="/profiles", tags=["Profiles"])
        logger.info("  ✅ Profiles router registered")
    except ImportError:
        logger.debug("  ℹ️  Profiles router not available")
    except Exception as e:
        logger.warning(f"  ⚠️  Profiles router failed: {e}")

    # ============================================================================
    # Setup Signal Handlers
    # ============================================================================

    import signal

    def handle_sigterm(signum, frame):
        """Handle SIGTERM signal for graceful shutdown."""
        logger.info("Received SIGTERM, shutting down gracefully...")
        facade.cleanup()
        sys.exit(0)

    def handle_sighup(signum, frame):
        """Handle SIGHUP signal for configuration reload."""
        logger.info("Received SIGHUP, reloading configuration...")
        # TODO: Implement configuration reload

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGHUP, handle_sighup)

    # ============================================================================
    # Start Server
    # ============================================================================

    port = int(os.getenv("SERVICE_MANAGER_PORT", "8888"))
    host = os.getenv("SERVICE_MANAGER_HOST", "0.0.0.0")

    logger.info(f"🌐 Starting HTTP server on {host}:{port}")
    logger.info("━" * 60)
    logger.info(f"   {GREEN}Service Manager Ready!{RESET}")
    logger.info(f"   API: http://localhost:{port}")
    logger.info(f"   Docs: http://localhost:{port}/docs")
    logger.info(f"   Health: http://localhost:{port}/health")
    logger.info("━" * 60)

    # Store start time for uptime calculation
    facade.start_time = datetime.now()

    # Start uvicorn server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False  # Disable uvicorn access logs (we have our own)
    )

    return {
        "status": "started",
        "port": port,
        "pid": os.getpid()
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
