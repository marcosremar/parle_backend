"""
Orchestrator Service Task Integration
Handles async tasks for pipeline orchestration
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
import httpx
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.task_manager import TaskManager, Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)

# Global task manager instance
task_manager: Optional[TaskManager] = None

# Global Communication Manager (set by orchestrator service)
comm_manager: Optional['ServiceCommunicationManager'] = None

# Service health cache
service_health_cache = {}
last_health_check = None


def set_comm_manager(cm):
    """Set Communication Manager instance from orchestrator service"""
    global comm_manager
    comm_manager = cm


def setup_task_integration():
    """Setup task integration for Orchestrator service"""
    global task_manager

    try:
        task_manager = TaskManager(service_name="orchestrator")

        # Register task handlers
        task_manager.register_handler(TaskType.VALIDATE, handle_validate)
        task_manager.register_handler(TaskType.HEALTH_CHECK, handle_health_check)
        task_manager.register_handler("CHECK_SERVICES", handle_check_services)

        # Start background health monitoring
        asyncio.create_task(background_health_monitor())

        logger.info("✅ Task integration setup completed for Orchestrator")
        return task_manager

    except Exception as e:
        logger.error(f"Failed to setup task integration: {e}")
        return None


async def handle_validate(task: Task) -> Dict[str, Any]:
    """Handle validation task"""
    try:
        # Check all dependent services (environment variables with defaults)
        services_to_check = {
            "llm": os.getenv('LLM_SERVICE_URL', 'http://localhost:8100') + '/health',
            "tts": os.getenv('TTS_SERVICE_URL', 'http://localhost:8101') + '/health',
            "stt": os.getenv('STT_SERVICE_URL', 'http://localhost:8099') + '/health',
            "webrtc": os.getenv('WEBRTC_SERVICE_URL', 'http://localhost:8010') + '/health',
            "websocket": os.getenv('WEBSOCKET_SERVICE_URL', 'http://localhost:8020') + '/health'
        }

        validation_results = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for service_name, url in services_to_check.items():
                try:
                    response = await client.get(url)
                    validation_results[service_name] = {
                        "reachable": True,
                        "healthy": response.status_code == 200,
                        "status": response.json().get("status", "unknown") if response.status_code == 200 else "error"
                    }
                except Exception as e:
                    validation_results[service_name] = {
                        "reachable": False,
                        "healthy": False,
                        "error": str(e)[:100]
                    }

        # Check if critical services are healthy
        critical_services = ["llm", "tts", "stt"]
        critical_healthy = all(
            validation_results.get(s, {}).get("healthy", False)
            for s in critical_services
        )

        return {
            "success": True,
            "valid": critical_healthy,
            "services": validation_results,
            "status": "healthy" if critical_healthy else "degraded"
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def handle_health_check(task: Task) -> Dict[str, Any]:
    """Handle health check task"""
    try:
        # Use cached health data if recent
        global last_health_check, service_health_cache

        if last_health_check and (datetime.now() - last_health_check).seconds < 30:
            return {
                "success": True,
                "cached": True,
                "services": service_health_cache,
                "last_check": last_health_check.isoformat()
            }

        # Perform new health check
        result = await handle_check_services(task)
        return result

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def handle_check_services(task: Task) -> Dict[str, Any]:
    """Check health of all dependent services"""
    global last_health_check, service_health_cache

    try:
        services_status = {}
        services_to_check = {
            "llm": "http://localhost:8100/health",
            "tts": "http://localhost:8101/health",
            "stt": "http://localhost:8099/health",
            "webrtc": "http://localhost:8500/health",
            "websocket": "http://localhost:8302/health",
            "api_gateway": "http://localhost:8020/health"
        }

        # Use fallback to httpx for health checks (GET requests)
        # Communication Manager is for POST service calls
        async with httpx.AsyncClient(timeout=3.0) as client:
            for service_name, url in services_to_check.items():
                try:
                    response = await client.get(url)
                    service_data = response.json() if response.status_code == 200 else {}

                    services_status[service_name] = {
                        "healthy": response.status_code == 200,
                        "status": service_data.get("status", "unknown"),
                        "response_time_ms": int(response.elapsed.total_seconds() * 1000)
                    }

                    # Check for initialization status
                    if "initialization" in service_data:
                        services_status[service_name]["initialization"] = service_data["initialization"]

                except Exception as e:
                    services_status[service_name] = {
                        "healthy": False,
                        "status": "offline",
                        "error": str(e)[:100]
                    }

        # Update cache
        service_health_cache = services_status
        last_health_check = datetime.now()

        # Determine overall health
        critical_services = ["llm", "tts", "stt"]
        critical_healthy = all(
            services_status.get(s, {}).get("healthy", False)
            for s in critical_services
        )

        return {
            "success": True,
            "services": services_status,
            "overall_status": "healthy" if critical_healthy else "degraded",
            "timestamp": last_health_check.isoformat()
        }

    except Exception as e:
        logger.error(f"Service check failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def background_health_monitor():
    """Background task to monitor service health"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds

            # Create a health check task
            health_task = Task(
                id=f"health_{datetime.now().timestamp()}",
                type="CHECK_SERVICES",
                status=TaskStatus.PENDING,
                metadata={}
            )

            # Run health check
            result = await handle_check_services(health_task)

            if not result["success"]:
                logger.warning(f"Health monitor detected issues: {result}")

        except Exception as e:
            logger.error(f"Background health monitor error: {e}")
            await asyncio.sleep(60)  # Wait longer on error


def get_service_health() -> Dict[str, Any]:
    """Get current service health status"""
    return {
        "services": service_health_cache,
        "last_check": last_health_check.isoformat() if last_health_check else None,
        "cache_age_seconds": (datetime.now() - last_health_check).seconds if last_health_check else None
    }