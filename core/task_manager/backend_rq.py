"""
RQ-based Task Queue Implementation
Drop-in replacement for BackgroundTaskManager using Redis Queue (RQ)

Key Benefits over custom implementation:
- Persistent tasks (survive restart)
- Distributed workers
- Built-in retry with exponential backoff
- Web dashboard (rq-dashboard)
- Battle-tested (10+ years in production)
- Eliminates 350 lines of threading code
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from redis import Redis
from rq import Queue, Worker, Retry
from rq.job import Job
from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry

from .models import (
    Task, TaskStatus, TaskPriority, TaskType,
    TaskProgress, TaskMetadata, TaskFilter, TaskStatistics
)

# Setup logging
logger = logging.getLogger(__name__)

# Global task handlers registry (module-level for pickling)
_global_task_handlers: Dict[str, Dict[TaskType, Callable]] = {}


def execute_rq_task(service_name: str, task_id: str, redis_url: str):
    """
    Global function to execute RQ tasks
    Must be at module level for pickle serialization

    Args:
        service_name: Service name
        task_id: Task ID
        redis_url: Redis connection URL
    """
    # Connect to Redis
    redis_conn = Redis.from_url(redis_url)

    # Get task metadata
    task_key = f"task:{service_name}:{task_id}"
    task_data = redis_conn.get(task_key)
    if not task_data:
        raise ValueError(f"Task {task_id} not found")

    task = Task.from_json(task_data.decode('utf-8'))

    # Update status to RUNNING
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.now()
    task.updated_at = datetime.now()

    # Save updated task
    redis_conn.setex(task_key, timedelta(hours=24), task.to_json())

    try:
        # Get handler from global registry
        handlers = _global_task_handlers.get(service_name, {})
        handler = handlers.get(task.type)

        if not handler:
            raise ValueError(f"No handler registered for task type {task.type.value}")

        # Execute handler
        result = handler(task)

        # Mark as completed
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.updated_at = datetime.now()
        task.result = result if isinstance(result, dict) else {"result": result}
        task.progress.update(
            current=task.progress.total_steps,
            percentage=100.0,
            message="Completed successfully"
        )

        # Save final task state
        redis_conn.setex(task_key, timedelta(hours=24), task.to_json())

        return result

    except Exception as e:
        # Mark as failed
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.updated_at = datetime.now()
        task.error = str(e)

        # Save failed state
        redis_conn.setex(task_key, timedelta(hours=24), task.to_json())

        raise


class RQTaskManager:
    """
    Redis Queue-based task manager that provides the same API
    as BackgroundTaskManager but with much better robustness
    """

    def __init__(
        self,
        service_name: str,
        max_workers: int = 5,
        max_queue_size: int = 1000,
        default_timeout: int = 300,
        enable_persistence: bool = True,  # Always true with RQ
        cleanup_after_hours: int = 24,
        logger: Optional[logging.Logger] = None,
        redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ):
        self.service_name = service_name
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.cleanup_after_hours = cleanup_after_hours
        self.logger = logger or logging.getLogger(__name__)

        # Connect to Redis
        try:
            self.redis_conn = Redis.from_url(redis_url)
            self.redis_conn.ping()  # Test connection
            self.logger.info(f"✅ Connected to Redis: {redis_url}")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Redis: {e}")
            raise RuntimeError(f"Redis connection failed: {e}")

        # Create RQ queues for different priorities
        self.queues = {
            TaskPriority.CRITICAL: Queue('critical', connection=self.redis_conn),
            TaskPriority.HIGH: Queue('high', connection=self.redis_conn),
            TaskPriority.NORMAL: Queue('normal', connection=self.redis_conn),
            TaskPriority.LOW: Queue('low', connection=self.redis_conn),
        }

        # Task metadata storage (Task object → Job mapping)
        # We store full Task objects in Redis to maintain compatibility
        self._task_prefix = f"task:{service_name}:"
        self.redis_url = redis_url

        # Initialize global handlers registry for this service
        if service_name not in _global_task_handlers:
            _global_task_handlers[service_name] = {}

        # Statistics (cached, updated periodically)
        self.stats = TaskStatistics()

        self.logger.info(
            f"RQTaskManager initialized for {service_name} "
            f"(queues: critical/high/normal/low)"
        )

    def _get_queue(self, priority: TaskPriority) -> Queue:
        """Get RQ queue for given priority"""
        return self.queues.get(priority, self.queues[TaskPriority.NORMAL])

    def _task_key(self, task_id: str) -> str:
        """Get Redis key for task metadata"""
        return f"{self._task_prefix}{task_id}"

    def _store_task(self, task: Task):
        """Store task metadata in Redis"""
        key = self._task_key(task.id)
        self.redis_conn.setex(
            key,
            timedelta(hours=self.cleanup_after_hours),
            task.to_json()
        )

    def _get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task metadata from Redis"""
        key = self._task_key(task_id)
        data = self.redis_conn.get(key)
        if data:
            return Task.from_json(data.decode('utf-8'))
        return None

    def _update_task(self, task: Task):
        """Update task metadata in Redis"""
        self._store_task(task)

    def _delete_task(self, task_id: str):
        """Delete task metadata from Redis"""
        key = self._task_key(task_id)
        self.redis_conn.delete(key)

    # Public API - Compatible with BackgroundTaskManager

    def create_task(
        self,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None
    ) -> str:
        """
        Create and queue a new task

        Compatible with BackgroundTaskManager.create_task()
        """
        # Create task object
        task = Task(
            type=task_type,
            priority=priority,
            parent_task_id=parent_task_id
        )

        # Set metadata
        task.metadata.service_name = self.service_name
        if metadata:
            for key, value in metadata.items():
                if hasattr(task.metadata, key):
                    setattr(task.metadata, key, value)
                else:
                    task.metadata.custom_data[key] = value

        # Store task metadata
        self._store_task(task)

        # Enqueue in RQ
        queue = self._get_queue(priority)
        timeout = task.metadata.timeout_seconds or self.default_timeout

        try:
            # Enqueue with retry configuration
            retry_config = None
            if task.metadata.max_retries > 0:
                retry_config = Retry(max=task.metadata.max_retries, interval=[60, 120, 300])

            rq_job = queue.enqueue(
                execute_rq_task,
                self.service_name,
                task.id,
                self.redis_url,
                job_id=task.id,  # Use task.id as RQ job_id for easy mapping
                timeout=timeout,
                result_ttl=self.cleanup_after_hours * 3600,
                failure_ttl=self.cleanup_after_hours * 3600,
                retry=retry_config,
            )

            self.logger.info(
                f"Task {task.id} enqueued in '{queue.name}' queue "
                f"(RQ job: {rq_job.id})"
            )

            return task.id

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"Failed to enqueue: {e}"
            self._update_task(task)
            raise RuntimeError(f"Failed to enqueue task: {e}")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID - compatible with BackgroundTaskManager"""
        return self._get_task(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status as dictionary"""
        task = self.get_task(task_id)
        if task:
            # Also sync with RQ job status
            try:
                rq_job = Job.fetch(task_id, connection=self.redis_conn)

                # Sync status from RQ if different
                if rq_job.is_queued and task.status == TaskStatus.PENDING:
                    pass  # Already correct
                elif rq_job.is_started and task.status != TaskStatus.RUNNING:
                    task.status = TaskStatus.RUNNING
                elif rq_job.is_finished and task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.COMPLETED
                    task.result = rq_job.result
                elif rq_job.is_failed and task.status != TaskStatus.FAILED:
                    task.status = TaskStatus.FAILED
                    task.error = str(rq_job.exc_info)

                self._update_task(task)
            except (KeyError, ValueError) as e:
                logger.debug(f"Job might not exist yet or already cleaned up for task {task_id}: {e}")

            return task.to_dict()
        return None

    def list_tasks(self, filter: Optional[TaskFilter] = None) -> List[Dict[str, Any]]:
        """List tasks with optional filtering"""
        # Get all task keys
        pattern = f"{self._task_prefix}*"
        task_keys = self.redis_conn.keys(pattern)

        tasks = []
        for key in task_keys:
            data = self.redis_conn.get(key)
            if data:
                try:
                    task = Task.from_json(data.decode('utf-8'))
                    tasks.append(task)
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to deserialize task from key {key}: {e}")
                    continue

        # Apply filters
        if filter:
            if filter.service_name:
                tasks = [t for t in tasks if t.metadata.service_name == filter.service_name]
            if filter.status:
                tasks = [t for t in tasks if t.status == filter.status]
            if filter.type:
                tasks = [t for t in tasks if t.type == filter.type]
            if filter.priority:
                tasks = [t for t in tasks if t.priority == filter.priority]
            if filter.created_after:
                tasks = [t for t in tasks if t.created_at >= filter.created_after]
            if filter.created_before:
                tasks = [t for t in tasks if t.created_at <= filter.created_before]
            if not filter.include_completed:
                tasks = [t for t in tasks if t.status != TaskStatus.COMPLETED]
            if not filter.include_failed:
                tasks = [t for t in tasks if t.status != TaskStatus.FAILED]

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        # Apply limit and offset
        if filter:
            tasks = tasks[filter.offset:filter.offset + filter.limit]

        return [t.to_dict() for t in tasks]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        task = self.get_task(task_id)
        if not task:
            return False

        if not task.can_cancel():
            return False

        try:
            # Cancel RQ job
            rq_job = Job.fetch(task_id, connection=self.redis_conn)
            rq_job.cancel()

            # Update task status
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            task.updated_at = datetime.now()
            task.error = "Cancelled by user"
            self._update_task(task)

            self.logger.info(f"Task {task_id} cancelled")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    def update_task_progress(
        self,
        task_id: str,
        current: int = None,
        total: int = None,
        percentage: float = None,
        message: str = None,
        details: Dict[str, Any] = None
    ):
        """Update task progress"""
        task = self.get_task(task_id)
        if task:
            # Use the TaskProgress.update method to handle percentage calculation
            task.progress.update(
                current=current,
                total=total,
                message=message,
                percentage=percentage
            )
            if details:
                task.progress.details.update(details)
            task.updated_at = datetime.now()
            self._update_task(task)

    def register_handler(self, task_type: TaskType, handler: Callable):
        """Register a handler function for a task type"""
        _global_task_handlers[self.service_name][task_type] = handler
        self.logger.info(f"Registered handler for task type {task_type.value}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current task statistics"""
        # Get all tasks
        pattern = f"{self._task_prefix}*"
        task_keys = self.redis_conn.keys(pattern)

        total = len(task_keys)
        pending = 0
        running = 0
        completed = 0
        failed = 0
        cancelled = 0
        timeout = 0

        durations = []

        for key in task_keys:
            data = self.redis_conn.get(key)
            if data:
                try:
                    task = Task.from_json(data.decode('utf-8'))

                    if task.status == TaskStatus.PENDING:
                        pending += 1
                    elif task.status == TaskStatus.RUNNING:
                        running += 1
                    elif task.status == TaskStatus.COMPLETED:
                        completed += 1
                    elif task.status == TaskStatus.FAILED:
                        failed += 1
                    elif task.status == TaskStatus.CANCELLED:
                        cancelled += 1
                    elif task.status == TaskStatus.TIMEOUT:
                        timeout += 1

                    if task.is_terminal() and task.duration_seconds():
                        durations.append(task.duration_seconds())
                except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
                    logger.warning(f"Failed to process task statistics for key {key}: {e}")
                    continue

        # Calculate stats
        avg_duration = sum(durations) / len(durations) if durations else 0
        terminal_tasks = completed + failed + timeout
        success_rate = (completed / terminal_tasks * 100) if terminal_tasks > 0 else 0

        self.stats.total_tasks = total
        self.stats.pending_tasks = pending
        self.stats.running_tasks = running
        self.stats.completed_tasks = completed
        self.stats.failed_tasks = failed
        self.stats.cancelled_tasks = cancelled
        self.stats.timeout_tasks = timeout
        self.stats.avg_duration_seconds = avg_duration
        self.stats.success_rate = success_rate

        return self.stats.to_dict()

    def shutdown(self, wait: bool = True):
        """Shutdown the task manager"""
        self.logger.info("RQTaskManager shutdown - tasks persist in Redis")
        # RQ workers are separate processes, nothing to shutdown here
        # Tasks will continue in Redis and can be processed by workers


# Singleton instance per service
_rq_task_managers: Dict[str, RQTaskManager] = {}


def get_rq_task_manager(
    service_name: str,
    max_workers: int = 5,
    **kwargs
) -> RQTaskManager:
    """Get or create an RQ task manager for a service"""
    if service_name not in _rq_task_managers:
        _rq_task_managers[service_name] = RQTaskManager(
            service_name=service_name,
            max_workers=max_workers,
            **kwargs
        )
    return _rq_task_managers[service_name]
