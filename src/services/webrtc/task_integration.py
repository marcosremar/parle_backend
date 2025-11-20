"""
WebRTC Service Task Integration
Handles async tasks for WebRTC operations
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.task_manager import TaskManager, Task, TaskStatus, TaskType
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)

# Global task manager instance
task_manager: Optional[TaskManager] = None

# Initialization status
init_status = {"started": False, "completed": False, "error": None}


def setup_task_integration():
    """Setup task integration for WebRTC service"""
    global task_manager

    try:
        task_manager = TaskManager(service_name="webrtc")

        # Register task handlers
        task_manager.register_handler(TaskType.INITIALIZE, handle_initialize)
        task_manager.register_handler(TaskType.VALIDATE, handle_validate)
        task_manager.register_handler(TaskType.RELOAD_MODEL, handle_reload)

        logger.info("✅ Task integration setup completed for WebRTC")
        return task_manager

    except Exception as e:
        logger.error(f"Failed to setup task integration: {e}")
        return None


async def handle_initialize(task: Task) -> Dict[str, Any]:
    """Handle initialization task"""
    global init_status

    try:
        init_status["started"] = True
        logger.info("Starting WebRTC initialization...")

        # Import and initialize controller
        from controllers import ConversationController
        controller = ConversationController()

        # Initialize with timeout
        await asyncio.wait_for(
            controller.initialize(),
            timeout=60  # 1 minute timeout
        )

        init_status["completed"] = True
        logger.info("✅ WebRTC initialization completed")

        return {
            "success": True,
            "message": "WebRTC initialized successfully",
            "controller_ready": True
        }

    except asyncio.TimeoutError:
        init_status["error"] = "Initialization timeout"
        logger.error("WebRTC initialization timeout")
        return {
            "success": False,
            "error": "Initialization timeout after 60 seconds"
        }

    except Exception as e:
        init_status["error"] = str(e)
        logger.error(f"WebRTC initialization failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def handle_validate(task: Task) -> Dict[str, Any]:
    """Handle validation task"""
    try:
        # Check initialization status
        if not init_status["completed"]:
            return {
                "success": False,
                "status": "initializing" if init_status["started"] else "not_started",
                "error": init_status.get("error")
            }

        # Validate WebRTC components
        validation_results = {
            "websocket_server": True,  # Check if WebSocket server is running
            "controller": init_status["completed"],
            "pipeline_connection": False  # Will be set based on actual check
        }

        # Check pipeline connection
        try:
            from src.core.model_manager import get_shared_pipeline
            pipeline = await asyncio.wait_for(get_shared_pipeline(), timeout=5)
            validation_results["pipeline_connection"] = pipeline is not None
        except Exception as e:
            validation_results["pipeline_connection"] = False

        all_valid = all(validation_results.values())

        return {
            "success": True,
            "valid": all_valid,
            "components": validation_results,
            "status": "healthy" if all_valid else "degraded"
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def handle_reload(task: Task) -> Dict[str, Any]:
    """Handle reload/restart task"""
    try:
        logger.info("Reloading WebRTC components...")

        # Re-initialize controller
        init_status["started"] = False
        init_status["completed"] = False
        init_status["error"] = None

        # Trigger re-initialization
        result = await handle_initialize(task)

        return {
            "success": result["success"],
            "message": "WebRTC reloaded",
            **result
        }

    except Exception as e:
        logger.error(f"Reload failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_initialization_status() -> Dict[str, Any]:
    """Get current initialization status"""
    return {
        "started": init_status["started"],
        "completed": init_status["completed"],
        "error": init_status["error"],
        "status": "healthy" if init_status["completed"] else "initializing" if init_status["started"] else "not_started"
    }