"""
Task Manager Module
Unified system for managing background tasks across all services

This module provides a simple interface for task management with support for
both in-memory and Redis Queue (RQ) backends.

Usage:
    from src.core.task_manager import get_task_manager, TaskType, TaskPriority

    # Get task manager instance (auto-selects backend based on config)
    tm = get_task_manager('my-service')

    # Register handler
    tm.register_handler(TaskType.CUSTOM, my_handler_function)

    # Create task
    task_id = tm.create_task(TaskType.CUSTOM, priority=TaskPriority.HIGH)

    # Monitor task
    status = tm.get_task_status(task_id)

Environment Variables:
    USE_RQ_TASKS=1|true|yes - Enable RQ backend (requires Redis)
    REDIS_URL - Redis connection URL (default: redis://localhost:6379/0)
"""

# Export main interface
from .factory import get_task_manager

# Export backends for direct instantiation (if needed)
from .backend import BackgroundTaskManager
# Alias for backward compatibility
TaskManager = BackgroundTaskManager

# Export models for convenience
from .models import (
    Task,
    TaskStatus,
    TaskPriority,
    TaskType,
    TaskProgress,
    TaskMetadata,
    TaskFilter,
    TaskStatistics,
)

# Define public API
__all__ = [
    # Factory
    'get_task_manager',

    # Backends
    'BackgroundTaskManager',
    'TaskManager',  # Alias for backward compatibility

    # Models
    'Task',
    'TaskStatus',
    'TaskPriority',
    'TaskType',
    'TaskProgress',
    'TaskMetadata',
    'TaskFilter',
    'TaskStatistics',
]

# Version
__version__ = '2.0.0'  # v2.0 - Modular architecture with RQ support
