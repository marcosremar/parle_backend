"""Configuration classes for process management"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List


@dataclass
class ServiceConfig:
    """Configuration for a managed service"""

    # Required
    name: str
    command: List[str]

    # Optional
    port: Optional[int] = None
    working_dir: Optional[str] = None

    # Resource Limits
    memory_mb: int = 150  # RAM limit
    cpu_percent: int = 25  # CPU quota (systemd only)
    cpu_nice: int = 10  # CPU priority for subprocess mode
    max_fds: int = 1024  # File descriptor limit

    # Behavior
    auto_restart: bool = True  # Restart on crash
    restart_delay_seconds: int = 5  # Wait before restart
    max_restarts: int = 3  # Max restart attempts
    kill_timeout: int = 30  # Seconds to wait for graceful shutdown

    # Environment
    env: Optional[Dict[str, str]] = None
    user: Optional[str] = None  # Run as different user (systemd only)

    # Healthcheck
    healthcheck_url: Optional[str] = None  # HTTP health endpoint
    healthcheck_interval: int = 30  # Health check frequency
    healthcheck_timeout: int = 5  # Health check timeout

    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ProcessStatus:
    """Status information for a process"""

    name: str
    pid: Optional[int]
    state: str  # 'running', 'stopped', 'failed', 'restarting', 'unknown'

    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0

    # Timing
    started_at: Optional[datetime] = None
    uptime_seconds: float = 0.0

    # Health
    is_healthy: bool = False
    last_health_check: Optional[datetime] = None
    restart_count: int = 0

    # Limits
    memory_limit_mb: int = 0
    cpu_limit_percent: int = 0

    # Additional info
    exit_code: Optional[int] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "pid": self.pid,
            "state": self.state,
            "cpu_percent": self.cpu_percent,
            "memory_mb": round(self.memory_mb, 2),
            "memory_percent": round(self.memory_percent, 2),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check.isoformat()
            if self.last_health_check
            else None,
            "restart_count": self.restart_count,
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_limit_percent": self.cpu_limit_percent,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
        }


@dataclass
class OrphanInfo:
    """Information about an orphan process"""

    pid: int
    name: str
    cmdline: str
    age_seconds: float
    memory_mb: float
    cpu_percent: float = 0.0


@dataclass
class ZombieInfo:
    """Information about a zombie process"""

    pid: int
    name: str
    ppid: int  # Parent PID
    parent_name: Optional[str] = None
