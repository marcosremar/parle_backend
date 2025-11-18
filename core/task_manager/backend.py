"""
Background Task Manager
Unified system for managing background tasks across all services
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import PriorityQueue, Queue, Empty
import time
import traceback

from .models import (
    Task, TaskStatus, TaskPriority, TaskType,
    TaskProgress, TaskMetadata, TaskFilter, TaskStatistics
)


class BackgroundTaskManager:
    """
    Manages background task execution with priority queue,
    worker pool, and comprehensive tracking
    """

    def __init__(
        self,
        service_name: str,
        max_workers: int = 5,
        max_queue_size: int = 1000,
        default_timeout: int = 300,
        enable_persistence: bool = False,
        cleanup_after_hours: int = 24,
        logger: Optional[logging.Logger] = None
    ):
        self.service_name = service_name
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.enable_persistence = enable_persistence
        self.cleanup_after_hours = cleanup_after_hours
        self.logger = logger or logging.getLogger(__name__)

        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.task_lock = threading.Lock()

        # Priority queue for pending tasks
        self.task_queue = PriorityQueue(maxsize=max_queue_size)

        # Worker management
        self.workers: List[threading.Thread] = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.shutdown_event = threading.Event()

        # Task callbacks
        self.task_handlers: Dict[TaskType, Callable] = {}

        # Statistics
        self.stats = TaskStatistics()
        self.stats_lock = threading.Lock()

        # Start workers
        self._start_workers()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        self.logger.info(f"TaskManager initialized for {service_name} with {max_workers} workers")

    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"{self.service_name}-worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def _worker_loop(self):
        """Main worker loop that processes tasks from the queue"""
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                priority, task_id = self.task_queue.get(timeout=1)

                # Get task details
                with self.task_lock:
                    task = self.tasks.get(task_id)

                if not task:
                    self.logger.warning(f"Task {task_id} not found in storage")
                    continue

                # Only process tasks that are pending or retrying
                if task.status not in [TaskStatus.PENDING, TaskStatus.RETRYING]:
                    self.logger.debug(f"Skipping task {task_id} with status {task.status}")
                    continue

                # Execute task
                self._execute_task(task)

                # Mark queue task as done
                self.task_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {str(e)}\n{traceback.format_exc()}")

    def _execute_task(self, task: Task):
        """Execute a single task"""
        try:
            # Update task status
            with self.task_lock:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                task.updated_at = datetime.now()
                self._update_stats()

            self.logger.info(f"Starting task {task.id} of type {task.type.value}")

            # Get handler for task type
            handler = self.task_handlers.get(task.type)
            if not handler:
                raise ValueError(f"No handler registered for task type {task.type.value}")

            # Set up timeout
            timeout = task.metadata.timeout_seconds or self.default_timeout

            # Execute with timeout
            future = self.executor.submit(handler, task)
            try:
                result = future.result(timeout=timeout)
                self._complete_task(task, result)
            except TimeoutError:
                self._timeout_task(task)
            except Exception as e:
                self._fail_task(task, str(e), traceback.format_exc())

        except Exception as e:
            self.logger.error(f"Task execution error: {str(e)}\n{traceback.format_exc()}")
            self._fail_task(task, str(e), traceback.format_exc())

    def _complete_task(self, task: Task, result: Any):
        """Mark task as completed"""
        with self.task_lock:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.updated_at = datetime.now()
            task.result = result if isinstance(result, dict) else {"result": result}
            task.progress.update(
                current=task.progress.total_steps,
                percentage=100.0,
                message="Completed successfully"
            )
            self._update_stats()

        self.logger.info(f"Task {task.id} completed successfully")
        self._trigger_webhook(task)

    def _fail_task(self, task: Task, error: str, details: str = None):
        """Mark task as failed and potentially retry"""
        with self.task_lock:
            # Check if we should retry
            if task.metadata.retry_count < task.metadata.max_retries:
                task.status = TaskStatus.RETRYING
                task.metadata.retry_count += 1
                task.error = f"Attempt {task.metadata.retry_count} failed: {error}"
                task.updated_at = datetime.now()

                # Re-queue with lower priority
                priority = task.priority.value + 1
                self.task_queue.put((priority, task.id))

                self.logger.warning(f"Task {task.id} failed, retrying ({task.metadata.retry_count}/{task.metadata.max_retries})")
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.updated_at = datetime.now()
                task.error = error
                task.error_details = {"traceback": details} if details else None
                self._update_stats()

                self.logger.error(f"Task {task.id} failed after {task.metadata.retry_count} retries: {error}")
                self._trigger_webhook(task)

    def _timeout_task(self, task: Task):
        """Mark task as timed out"""
        with self.task_lock:
            task.status = TaskStatus.TIMEOUT
            task.completed_at = datetime.now()
            task.updated_at = datetime.now()
            task.error = f"Task exceeded timeout of {task.metadata.timeout_seconds} seconds"
            self._update_stats()

        self.logger.error(f"Task {task.id} timed out after {task.metadata.timeout_seconds} seconds")
        self._trigger_webhook(task)

    def _update_stats(self):
        """Update task statistics"""
        with self.stats_lock:
            self.stats.total_tasks = len(self.tasks)
            self.stats.pending_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
            self.stats.running_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
            self.stats.completed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            self.stats.failed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
            self.stats.cancelled_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.CANCELLED)
            self.stats.timeout_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.TIMEOUT)

            # Calculate average duration
            durations = []
            for task in self.tasks.values():
                if task.is_terminal() and task.duration_seconds():
                    durations.append(task.duration_seconds())

            if durations:
                self.stats.avg_duration_seconds = sum(durations) / len(durations)

            # Calculate success rate
            terminal_tasks = self.stats.completed_tasks + self.stats.failed_tasks + self.stats.timeout_tasks
            if terminal_tasks > 0:
                self.stats.success_rate = (self.stats.completed_tasks / terminal_tasks) * 100

    def _trigger_webhook(self, task: Task):
        """Trigger webhook if configured"""
        if task.metadata.webhook_url:
            # Implement webhook notification (async)
            self.executor.submit(self._send_webhook, task)

    def _send_webhook(self, task: Task):
        """Send webhook notification"""
        try:
            import requests
            response = requests.post(
                task.metadata.webhook_url,
                json=task.to_dict(),
                timeout=10
            )
            self.logger.debug(f"Webhook sent for task {task.id}: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to send webhook for task {task.id}: {str(e)}")

    def _cleanup_loop(self):
        """Periodically clean up old completed tasks"""
        while not self.shutdown_event.is_set():
            try:
                self.shutdown_event.wait(timeout=3600)  # Check every hour
                if self.shutdown_event.is_set():
                    break

                cutoff = datetime.now() - timedelta(hours=self.cleanup_after_hours)
                removed_count = 0

                with self.task_lock:
                    tasks_to_remove = []
                    for task_id, task in self.tasks.items():
                        if task.is_terminal() and task.completed_at and task.completed_at < cutoff:
                            tasks_to_remove.append(task_id)

                    for task_id in tasks_to_remove:
                        del self.tasks[task_id]
                        removed_count += 1

                if removed_count > 0:
                    self.logger.info(f"Cleaned up {removed_count} old tasks")

            except Exception as e:
                self.logger.error(f"Cleanup error: {str(e)}")

    # Public API methods

    def create_task(
        self,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None
    ) -> str:
        """Create and queue a new task"""
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

        # Store task
        with self.task_lock:
            self.tasks[task.id] = task
            self._update_stats()

        # Queue task
        try:
            self.task_queue.put_nowait((priority.value, task.id))
            self.logger.info(f"Task {task.id} created and queued with priority {priority.value}")
        except Exception as e:
            with self.task_lock:
                task.status = TaskStatus.FAILED
                task.error = "Queue is full"
            raise RuntimeError(f"Task queue is full (max {self.max_queue_size})")

        return task.id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        with self.task_lock:
            return self.tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status as dictionary"""
        task = self.get_task(task_id)
        return task.to_dict() if task else None

    def list_tasks(self, filter: Optional[TaskFilter] = None) -> List[Dict[str, Any]]:
        """List tasks with optional filtering"""
        with self.task_lock:
            tasks = list(self.tasks.values())

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
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            if not task.can_cancel():
                return False

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            task.updated_at = datetime.now()
            task.error = "Cancelled by user"
            self._update_stats()

        self.logger.info(f"Task {task_id} cancelled")
        return True

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
        with self.task_lock:
            task = self.tasks.get(task_id)
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

    def register_handler(self, task_type: TaskType, handler: Callable):
        """Register a handler function for a task type"""
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type {task_type.value}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current task statistics"""
        with self.stats_lock:
            return self.stats.to_dict()

    def shutdown(self, wait: bool = True):
        """Shutdown the task manager"""
        self.logger.info("Shutting down TaskManager...")
        self.shutdown_event.set()

        if wait:
            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=5)

            # Shutdown executor
            self.executor.shutdown(wait=True)

        self.logger.info("TaskManager shutdown complete")