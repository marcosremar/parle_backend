"""
Example: How to integrate systemd watchdog and lifecycle management into Service Manager

This file shows the integration points needed in main.py
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .systemd_watchdog import get_watchdog
from .systemd_manager import SystemdServiceManager

#=============================================================================
# 1. LIFESPAN INTEGRATION (replaces existing lifespan)
#=============================================================================

@asynccontextmanager
async def lifespan_with_systemd(app: FastAPI):
    """
    Enhanced lifespan with systemd integration

    Add this to replace the existing lifespan in main.py
    """
    # Get watchdog instance
    watchdog = get_watchdog()

    # Initialize watchdog from environment
    watchdog.initialize()

    # STARTUP
    logger.info("🚀 Service Manager starting...")

    # Update status
    watchdog.notify_status("Starting internal services...")

    # Initialize your services here (existing code)
    # manager = ServiceManager()
    # await manager.initialize()

    watchdog.notify_status("Starting external services...")

    # Start external services if needed
    # await manager.start_all_services()

    # Start watchdog heartbeat
    await watchdog.start()

    # Notify systemd we're ready
    watchdog.notify_ready()
    watchdog.notify_status("Service Manager ready - managing all services")

    logger.info("✅ Service Manager ready")

    yield  # Application runs here

    # SHUTDOWN
    logger.info("🛑 Service Manager shutting down...")

    # Notify systemd we're stopping
    watchdog.notify_stopping()
    watchdog.notify_status("Stopping all services...")

    # Stop watchdog
    await watchdog.stop()

    # Stop services gracefully (existing code)
    # await manager.stop_all_services()
    # await manager.shutdown()

    logger.info("✅ Service Manager stopped cleanly")


#=============================================================================
# 2. SYSTEM MANAGEMENT ENDPOINTS (add these to main.py)
#=============================================================================

def add_system_endpoints(app: FastAPI, manager):
    """Add system management endpoints"""

    watchdog = get_watchdog()

    @app.get("/system/health")
    async def system_health():
        """
        Comprehensive system health check

        Returns:
            - Service Manager status
            - Internal services status
            - External services status
            - Watchdog status
        """
        # Check internal services
        internal_health = {
            "session": "healthy",  # manager.check_internal_service("session")
            "scenarios": "healthy",
            # ... other internal services
        }

        # Check external services (via systemd if enabled)
        external_health = {}
        if hasattr(manager, 'systemd_manager'):
            for service_id in manager.services.keys():
                if manager.is_external(service_id):
                    status = manager.systemd_manager.get_service_status(service_id)
                    external_health[service_id] = {
                        "state": status.state.value,
                        "pid": status.pid
                    }

        return {
            "status": "healthy",
            "service_manager": "running",
            "internal_services": internal_health,
            "external_services": external_health,
            "watchdog": watchdog.get_status(),
            "mode": "systemd" if os.getenv('SERVICE_MANAGER_MODE') == 'systemd' else 'subprocess'
        }

    @app.post("/system/reload")
    async def reload_configuration():
        """
        Reload configuration without restarting

        Steps:
        1. Notify systemd we're reloading
        2. Reload YAML configuration
        3. Apply changes to services
        4. Notify systemd we're ready
        """
        watchdog.notify_reloading()
        watchdog.notify_status("Reloading configuration...")

        try:
            # Reload configuration
            # manager.reload_config()

            # Apply changes
            # await manager.apply_config_changes()

            watchdog.notify_ready()
            watchdog.notify_status("Configuration reloaded successfully")

            return {
                "success": True,
                "message": "Configuration reloaded successfully"
            }
        except Exception as e:
            watchdog.notify_status(f"Reload failed: {str(e)}")
            watchdog.notify_ready()  # Back to ready state
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/system/restart")
    async def restart_service_manager():
        """
        Graceful self-restart via systemd

        Only works if running under systemd.
        Systemd will automatically restart the service.
        """
        if os.getenv('SERVICE_MANAGER_MODE') != 'systemd':
            raise HTTPException(
                status_code=400,
                detail="Self-restart only available in systemd mode"
            )

        watchdog.notify_stopping()
        watchdog.notify_status("Restarting Service Manager...")

        # Schedule restart
        async def do_restart():
            await asyncio.sleep(1)  # Let response be sent
            # Systemd will restart us when we exit
            os._exit(0)

        asyncio.create_task(do_restart())

        return {
            "success": True,
            "message": "Service Manager restarting..."
        }

    @app.get("/system/watchdog")
    async def get_watchdog_status():
        """Get watchdog status"""
        return watchdog.get_status()


#=============================================================================
# 3. MAIN ENTRY POINT (modify existing main.py)
#=============================================================================

def main():
    """
    Main entry point with systemd integration

    Modify the existing main() function in main.py:
    """
    import uvicorn

    # Check if running in systemd mode
    systemd_mode = os.getenv('SERVICE_MANAGER_MODE') == 'systemd'

    if systemd_mode:
        logger.info("🔧 Running in SYSTEMD mode")
    else:
        logger.info("🔧 Running in SUBPROCESS mode")

    # Create FastAPI app with systemd lifespan
    app = FastAPI(lifespan=lifespan_with_systemd)

    # Initialize Service Manager
    # manager = ServiceManager()

    # Add system endpoints
    # add_system_endpoints(app, manager)

    # Add existing endpoints
    # ... existing endpoint registrations

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8888,
        log_config=None  # Use existing logging
    )


#=============================================================================
# 4. GRACEFUL SHUTDOWN SIGNAL HANDLER (optional but recommended)
#=============================================================================

import signal

def setup_signal_handlers(app: FastAPI):
    """Setup signal handlers for graceful shutdown"""

    watchdog = get_watchdog()

    def handle_sigterm(signum, frame):
        """Handle SIGTERM from systemd"""
        logger.info("Received SIGTERM, shutting down gracefully...")
        watchdog.notify_stopping()
        # FastAPI will handle the actual shutdown via lifespan

    def handle_sighup(signum, frame):
        """Handle SIGHUP (reload signal)"""
        logger.info("Received SIGHUP, reloading configuration...")
        watchdog.notify_reloading()
        # Trigger reload here
        # asyncio.create_task(reload_config())
        watchdog.notify_ready()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGHUP, handle_sighup)

    logger.info("✅ Signal handlers registered")


#=============================================================================
# INTEGRATION CHECKLIST
#=============================================================================

"""
To integrate into main.py:

1. ✅ Import systemd_watchdog:
   from .systemd_watchdog import get_watchdog

2. ✅ Replace lifespan function:
   - Use lifespan_with_systemd instead of current lifespan
   - Add watchdog.initialize() at startup
   - Add watchdog.start() after services initialized
   - Add watchdog.notify_ready() when ready
   - Add watchdog.stop() in shutdown

3. ✅ Add system endpoints:
   - /system/health - comprehensive health check
   - /system/reload - reload configuration
   - /system/restart - self-restart via systemd
   - /system/watchdog - watchdog status

4. ✅ Setup signal handlers (optional):
   - SIGTERM for graceful shutdown
   - SIGHUP for reload

5. ✅ Update systemd service file:
   - Type=notify (already configured)
   - WatchdogSec=30 (already configured)
   - NotifyAccess=main (already configured)

6. ✅ Install systemd Python package:
   pip install systemd-python

That's it! Service Manager will now:
- Send heartbeats to systemd
- Notify systemd of lifecycle events
- Support graceful reload and restart
- Provide system health endpoints
"""
