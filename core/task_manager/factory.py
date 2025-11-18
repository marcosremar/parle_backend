"""
Task Manager Factory
Creates appropriate task manager backend based on configuration
"""

import os
import logging
from typing import Dict, Union

# Import backends
from .backend import BackgroundTaskManager
try:
    from .backend_rq import RQTaskManager, get_rq_task_manager
    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False
    RQTaskManager = None


# Singleton instances per service
_task_managers: Dict[str, Union[BackgroundTaskManager, 'RQTaskManager']] = {}


def get_task_manager(
    service_name: str,
    max_workers: int = 5,
    use_rq: bool = None,
    **kwargs
) -> Union[BackgroundTaskManager, 'RQTaskManager']:
    """
    Get or create a task manager for a service

    This factory function automatically selects the appropriate backend:
    - RQ (Redis Queue): Persistent, distributed, production-ready
    - In-memory: Simple, no dependencies, development-friendly

    Args:
        service_name: Service name
        max_workers: Maximum worker threads/processes
        use_rq: Use RQ (Redis Queue) instead of in-memory implementation
                If None, checks environment variable USE_RQ_TASKS
        **kwargs: Additional arguments passed to backend

    Returns:
        Task manager instance (either BackgroundTaskManager or RQTaskManager)

    Environment Variables:
        USE_RQ_TASKS=1|true|yes - Enable RQ globally
        REDIS_URL - Redis connection URL (default: redis://localhost:6379/0)

    Examples:
        # Default (in-memory)
        tm = get_task_manager('my-service')

        # Explicit RQ
        tm = get_task_manager('my-service', use_rq=True)

        # Via environment variable
        os.environ['USE_RQ_TASKS'] = '1'
        tm = get_task_manager('my-service')
    """
    logger = logging.getLogger(__name__)

    # Determine if should use RQ
    if use_rq is None:
        use_rq_env = os.getenv('USE_RQ_TASKS', '').lower()
        use_rq = use_rq_env in ('1', 'true', 'yes', 'on')

    if use_rq:
        if not RQ_AVAILABLE:
            logger.warning(
                "RQ backend requested but not available. "
                "Install with: pip install redis rq. "
                "Falling back to in-memory implementation"
            )
        else:
            # Use RQ-based implementation
            try:
                logger.info(f"Using RQ (Redis Queue) for task manager: {service_name}")

                return get_rq_task_manager(
                    service_name=service_name,
                    max_workers=max_workers,
                    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                    **kwargs
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize RQ task manager: {e}. "
                    f"Falling back to in-memory implementation"
                )
                # Fall through to in-memory implementation

    # Use in-memory implementation (original)
    if service_name not in _task_managers:
        logger.info(f"Using in-memory task manager for: {service_name}")
        _task_managers[service_name] = BackgroundTaskManager(
            service_name=service_name,
            max_workers=max_workers,
            **kwargs
        )
    return _task_managers[service_name]
