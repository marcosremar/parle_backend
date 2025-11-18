"""Base launcher interface"""

from abc import ABC, abstractmethod
from typing import Optional
from ..config import ServiceConfig, ProcessStatus


class BaseLauncher(ABC):
    """Abstract base class for process launchers"""

    @abstractmethod
    def launch(self, config: ServiceConfig) -> int:
        """
        Launch a process

        Args:
            config: Service configuration

        Returns:
            PID of launched process

        Raises:
            ServiceStartFailedError: If launch fails
        """
        pass

    @abstractmethod
    def stop(self, name: str, pid: int, timeout: int = 30) -> bool:
        """
        Stop a process gracefully

        Args:
            name: Service name
            pid: Process ID
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_running(self, pid: int) -> bool:
        """
        Check if process is running

        Args:
            pid: Process ID

        Returns:
            True if running, False otherwise
        """
        pass

    @abstractmethod
    def get_status(self, name: str, pid: int, config: ServiceConfig) -> ProcessStatus:
        """
        Get detailed process status

        Args:
            name: Service name
            pid: Process ID
            config: Service configuration

        Returns:
            ProcessStatus with current information
        """
        pass

    @abstractmethod
    def get_launcher_type(self) -> str:
        """
        Get launcher type identifier

        Returns:
            'systemd' or 'subprocess'
        """
        pass
