"""
PyProcessd - Modern Python Process Manager
Combines systemd integration, psutil monitoring, and resource limits
"""

from .config import ServiceConfig, ProcessStatus
from .manager import ProcessManager
from .exceptions import (
    ProcessManagementError,
    SystemdNotAvailableError,
    ServiceNotFoundError,
    ServiceAlreadyRunningError,
)

__version__ = "0.1.0"

__all__ = [
    "ProcessManager",
    "ServiceConfig",
    "ProcessStatus",
    "ProcessManagementError",
    "SystemdNotAvailableError",
    "ServiceNotFoundError",
    "ServiceAlreadyRunningError",
]
