"""
Task Management Models and Enums
Unified system for background task management across all services
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid


class TaskStatus(Enum):
    """Task lifecycle states"""
    PENDING = "pending"          # In queue, not started
    RUNNING = "running"          # Currently executing
    COMPLETED = "completed"      # Successfully finished
    FAILED = "failed"           # Failed with error
    CANCELLED = "cancelled"      # Cancelled by user
    TIMEOUT = "timeout"         # Exceeded time limit
    RETRYING = "retrying"       # Failed but retrying


class TaskPriority(Enum):
    """Task priority levels (lower number = higher priority)"""
    CRITICAL = 1    # System critical operations
    HIGH = 2        # Service start/stop/restart
    NORMAL = 3      # Warmup, reload operations
    LOW = 4         # Validation operations
    BACKGROUND = 5  # Cleanup, metrics, etc


class TaskType(Enum):
    """Standard task types across services"""
    # Service lifecycle
    START = "start"
    STOP = "stop"
    RESTART = "restart"

    # Health and validation
    VALIDATE = "validate"
    HEALTH_CHECK = "health_check"

    # Model operations
    WARMUP = "warmup"
    RELOAD_MODEL = "reload_model"

    # Batch operations
    START_ALL = "start_all"
    STOP_ALL = "stop_all"
    RESTART_ALL = "restart_all"
    VALIDATE_ALL = "validate_all"

    # Custom operations
    CUSTOM = "custom"
    CLEANUP = "cleanup"
    BACKUP = "backup"


@dataclass
class TaskProgress:
    """Track task execution progress"""
    current_step: int = 0
    total_steps: int = 0
    percentage: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def update(self, current: int = None, total: int = None, message: str = None, percentage: float = None):
        """Update progress tracking"""
        if current is not None:
            self.current_step = current
        if total is not None:
            self.total_steps = total
        if message is not None:
            self.message = message

        # Allow explicit percentage override or calculate from steps
        if percentage is not None:
            self.percentage = percentage
        elif self.total_steps > 0:
            self.percentage = (self.current_step / self.total_steps) * 100


@dataclass
class TaskMetadata:
    """Additional task metadata"""
    service_name: str = ""
    service_port: Optional[int] = None
    target: Optional[str] = None  # Target service or resource
    user: Optional[str] = None
    ip_address: Optional[str] = None
    webhook_url: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    tags: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Universal task representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType = TaskType.CUSTOM
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)

    # Progress and results
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: TaskMetadata = field(default_factory=TaskMetadata)

    # Relationships
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API responses"""
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat(),
            "progress": {
                "current_step": self.progress.current_step,
                "total_steps": self.progress.total_steps,
                "percentage": self.progress.percentage,
                "message": self.progress.message,
                "details": self.progress.details
            },
            "result": self.result,
            "error": self.error,
            "error_details": self.error_details,
            "metadata": {
                "service_name": self.metadata.service_name,
                "service_port": self.metadata.service_port,
                "target": self.metadata.target,
                "user": self.metadata.user,
                "retry_count": self.metadata.retry_count,
                "max_retries": self.metadata.max_retries,
                "timeout_seconds": self.metadata.timeout_seconds,
                "tags": self.metadata.tags,
                "custom_data": self.metadata.custom_data
            },
            "parent_task_id": self.parent_task_id,
            "subtask_ids": self.subtask_ids
        }

    def is_terminal(self) -> bool:
        """Check if task is in a terminal state"""
        return self.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT
        ]

    def can_cancel(self) -> bool:
        """Check if task can be cancelled"""
        return self.status in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.RETRYING]

    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None

    def to_json(self) -> str:
        """Serialize task to JSON string"""
        import json
        data = self.to_dict()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Task':
        """Deserialize task from JSON string"""
        import json
        from datetime import datetime

        data = json.loads(json_str)

        # Reconstruct task
        task = cls(
            id=data['id'],
            type=TaskType(data['type']),
            status=TaskStatus(data['status']),
            priority=TaskPriority(data['priority']),
        )

        # Timestamps
        task.created_at = datetime.fromisoformat(data['created_at'])
        task.started_at = datetime.fromisoformat(data['started_at']) if data['started_at'] else None
        task.completed_at = datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None
        task.updated_at = datetime.fromisoformat(data['updated_at'])

        # Progress
        progress_data = data['progress']
        task.progress = TaskProgress(
            current_step=progress_data['current_step'],
            total_steps=progress_data['total_steps'],
            percentage=progress_data['percentage'],
            message=progress_data['message'],
            details=progress_data['details']
        )

        # Result and error
        task.result = data.get('result')
        task.error = data.get('error')
        task.error_details = data.get('error_details')

        # Metadata
        meta_data = data['metadata']
        task.metadata = TaskMetadata(
            service_name=meta_data.get('service_name'),
            service_port=meta_data.get('service_port'),
            target=meta_data.get('target'),
            user=meta_data.get('user'),
            retry_count=meta_data.get('retry_count', 0),
            max_retries=meta_data.get('max_retries', 3),
            timeout_seconds=meta_data.get('timeout_seconds'),
            tags=meta_data.get('tags', []),
            custom_data=meta_data.get('custom_data', {})
        )

        # Relationships
        task.parent_task_id = data.get('parent_task_id')
        task.subtask_ids = data.get('subtask_ids', [])

        return task


@dataclass
class TaskFilter:
    """Filters for querying tasks"""
    service_name: Optional[str] = None
    status: Optional[TaskStatus] = None
    type: Optional[TaskType] = None
    priority: Optional[TaskPriority] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[List[str]] = None
    limit: int = 100
    offset: int = 0
    include_completed: bool = True
    include_failed: bool = True


@dataclass
class TaskStatistics:
    """Task execution statistics"""
    total_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    timeout_tasks: int = 0
    avg_duration_seconds: float = 0.0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            "total": self.total_tasks,
            "pending": self.pending_tasks,
            "running": self.running_tasks,
            "completed": self.completed_tasks,
            "failed": self.failed_tasks,
            "cancelled": self.cancelled_tasks,
            "timeout": self.timeout_tasks,
            "avg_duration_seconds": self.avg_duration_seconds,
            "success_rate": self.success_rate
        }