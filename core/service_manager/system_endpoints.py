"""
System Management Endpoints for Service Manager
Provides health, reload, restart, and watchdog endpoints
"""

import os
import asyncio
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from .systemd_watchdog import get_watchdog

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/system", tags=["System Management"])

# Global reference to service manager (set during initialization)
_service_manager = None


def set_service_manager(manager):
    """Set global service manager reference"""
    global _service_manager
    _service_manager = manager


@router.get("/health")
async def system_health() -> Dict[str, Any]:
    """
    Comprehensive system health check

    Returns status of:
    - Service Manager itself
    - Watchdog
    - Internal services
    - External services (if systemd mode)
    """
    watchdog = get_watchdog()

    health_data = {
        "status": "healthy",
        "service_manager": {
            "mode": os.getenv('SERVICE_MANAGER_MODE', 'subprocess'),
            "running": True,
            "pid": os.getpid()
        },
        "watchdog": watchdog.get_status(),
        "timestamp": asyncio.get_event_loop().time()
    }

    # Add service manager specific health if available
    if _service_manager and hasattr(_service_manager, 'get_all_status'):
        try:
            services_status = _service_manager.get_all_status()
            health_data["services"] = services_status
        except Exception as e:
            logger.error(f"Failed to get services status: {e}")
            health_data["services"] = {"error": str(e)}

    return health_data


@router.post("/reload")
async def reload_configuration() -> Dict[str, Any]:
    """
    Reload configuration without restarting

    Steps:
    1. Notify systemd (if in systemd mode)
    2. Reload configuration files
    3. Apply changes to running services
    4. Notify systemd ready
    """
    watchdog = get_watchdog()
    watchdog.notify_reloading()
    watchdog.notify_status("Reloading configuration...")

    try:
        logger.info("🔄 Reloading configuration...")

        # Reload service execution config
        from src.config.service_execution_config import get_service_execution_config
        config = get_service_execution_config(reload=True)

        logger.info(f"✅ Configuration reloaded: {len(config.services)} services configured")

        # If service manager has reload method, call it
        if _service_manager and hasattr(_service_manager, 'reload_config'):
            await _service_manager.reload_config()

        watchdog.notify_ready()
        watchdog.notify_status("Configuration reloaded successfully")

        return {
            "success": True,
            "message": "Configuration reloaded successfully",
            "services_count": len(config.services),
            "timestamp": asyncio.get_event_loop().time()
        }

    except Exception as e:
        logger.error(f"❌ Failed to reload configuration: {e}")
        watchdog.notify_status(f"Reload failed: {str(e)}")
        watchdog.notify_ready()  # Back to ready state
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart")
async def restart_service_manager() -> Dict[str, Any]:
    """
    Graceful self-restart via systemd

    Only works if running under systemd.
    Systemd will automatically restart the service after exit.
    """
    systemd_mode = os.getenv('SERVICE_MANAGER_MODE') == 'systemd'

    if not systemd_mode:
        raise HTTPException(
            status_code=400,
            detail="Self-restart only available in systemd mode. Set SERVICE_MANAGER_MODE=systemd"
        )

    watchdog = get_watchdog()
    watchdog.notify_stopping()
    watchdog.notify_status("Service Manager restarting...")

    logger.info("🔄 Service Manager restarting via systemd...")

    # Schedule restart (let response be sent first)
    async def do_restart():
        await asyncio.sleep(1)
        logger.info("👋 Exiting for systemd restart...")
        # Systemd will restart us automatically
        os._exit(0)

    # Track background task for proper cleanup
    task = asyncio.create_task(do_restart())
    if _service_manager and hasattr(_service_manager, 'background_tasks'):
        _service_manager.background_tasks.append(task)

    return {
        "success": True,
        "message": "Service Manager restarting... systemd will restart automatically",
        "timestamp": asyncio.get_event_loop().time()
    }


@router.get("/watchdog")
async def get_watchdog_status() -> Dict[str, Any]:
    """
    Get systemd watchdog status

    Shows:
    - Whether watchdog is enabled
    - Last heartbeat time
    - Interval settings
    """
    watchdog = get_watchdog()
    return watchdog.get_status()


@router.get("/info")
async def system_info() -> Dict[str, Any]:
    """
    Get system information

    Returns:
    - Runtime mode (systemd/subprocess)
    - PID
    - Environment
    - Watchdog status
    """
    import psutil

    process = psutil.Process(os.getpid())

    return {
        "mode": os.getenv('SERVICE_MANAGER_MODE', 'subprocess'),
        "pid": os.getpid(),
        "uptime_seconds": asyncio.get_event_loop().time() - process.create_time(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "watchdog_enabled": get_watchdog().enabled,
        "environment": {
            "WATCHDOG_USEC": os.getenv('WATCHDOG_USEC'),
            "WATCHDOG_PID": os.getenv('WATCHDOG_PID'),
            "SERVICE_MANAGER_MODE": os.getenv('SERVICE_MANAGER_MODE'),
        }
    }


# Optional: Add status update endpoint
@router.post("/status")
async def update_status(message: str) -> Dict[str, Any]:
    """
    Update systemd status message

    Visible in `systemctl status ultravox-main`
    """
    watchdog = get_watchdog()
    watchdog.notify_status(message)

    return {
        "success": True,
        "status_message": message
    }
