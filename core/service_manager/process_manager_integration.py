"""
Integration layer between ProcessManager (PyProcessd) and existing Service Manager
Provides backward compatibility while migrating to new process management
"""

import logging
from typing import Dict, Optional, List
from pathlib import Path

from ..process_management import ProcessManager, ServiceConfig
from ..process_management.config import ProcessStatus as PMProcessStatus

logger = logging.getLogger(__name__)


class ProcessManagerAdapter:
    """
    Adapter that wraps ProcessManager and provides interface
    compatible with existing service manager
    """

    def __init__(
        self,
        mode: str = 'auto',
        user_mode: bool = True,
        enable_monitoring: bool = True
    ):
        """
        Initialize adapter

        Args:
            mode: Launcher mode ('auto', 'systemd', 'subprocess')
            user_mode: Use systemd --user
            enable_monitoring: Enable background monitoring
        """
        self.pm = ProcessManager(
            mode=mode,
            user_mode=user_mode,
            registry_path='os.path.expanduser("~/.cache/ultravox-pipeline/")pids.json',
            enable_monitoring=enable_monitoring,
            monitoring_interval=30,
            log_level='INFO'
        )

        logger.info(
            f"ProcessManagerAdapter initialized "
            f"(mode={mode}, launcher={self.pm.launcher.get_launcher_type()})"
        )

    def launch_service(
        self,
        service_name: str,
        command: List[str],
        port: Optional[int] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        memory_mb: int = 150,
        cpu_percent: int = 25,
        auto_restart: bool = True,
    ) -> bool:
        """
        Launch a service (compatible with old service_launcher API)

        Args:
            service_name: Service name
            command: Command to execute
            port: Port number
            working_dir: Working directory
            env: Environment variables
            memory_mb: Memory limit
            cpu_percent: CPU limit
            auto_restart: Enable auto-restart

        Returns:
            True if launched successfully
        """
        try:
            # Register if not already registered
            if service_name not in self.pm.services:
                config = ServiceConfig(
                    name=service_name,
                    command=command,
                    port=port,
                    working_dir=working_dir or str(Path.cwd()),
                    memory_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    cpu_nice=10,  # For subprocess mode
                    auto_restart=auto_restart,
                    env=env,
                    description=f'Ultravox {service_name} service'
                )
                self.pm.register(config)

            # Start
            return self.pm.start(service_name)

        except Exception as e:
            logger.error(f"Failed to launch {service_name}: {e}")
            return False

    def stop_service(self, service_name: str, timeout: int = 30) -> bool:
        """
        Stop a service

        Args:
            service_name: Service name
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully
        """
        try:
            return self.pm.stop(service_name, timeout)
        except Exception as e:
            logger.error(f"Failed to stop {service_name}: {e}")
            return False

    def restart_service(self, service_name: str) -> bool:
        """
        Restart a service

        Args:
            service_name: Service name

        Returns:
            True if restarted successfully
        """
        try:
            return self.pm.restart(service_name)
        except Exception as e:
            logger.error(f"Failed to restart {service_name}: {e}")
            return False

    def get_service_status(self, service_name: str) -> Dict:
        """
        Get service status (compatible with old API)

        Args:
            service_name: Service name

        Returns:
            Status dictionary
        """
        try:
            status = self.pm.status(service_name)
            return self._convert_status_to_dict(status)
        except Exception as e:
            logger.error(f"Failed to get status for {service_name}: {e}")
            return {
                'name': service_name,
                'state': 'unknown',
                'error': str(e)
            }

    def get_all_services_status(self) -> Dict[str, Dict]:
        """
        Get status of all services

        Returns:
            Dictionary of service_name -> status dict
        """
        statuses = self.pm.status_all()
        return {
            name: self._convert_status_to_dict(status)
            for name, status in statuses.items()
        }

    def is_service_running(self, service_name: str) -> bool:
        """
        Check if service is running

        Args:
            service_name: Service name

        Returns:
            True if running
        """
        return self.pm.pid_registry.is_running(service_name)

    def cleanup_orphans(self, dry_run: bool = False) -> int:
        """
        Cleanup orphan processes

        Args:
            dry_run: If True, only log what would be done

        Returns:
            Number of orphans cleaned
        """
        return self.pm.cleanup_orphans(dry_run=dry_run)

    def get_zombies(self) -> List[Dict]:
        """
        Get list of zombie processes

        Returns:
            List of zombie info dicts
        """
        zombies = self.pm.get_zombies()
        return [
            {
                'pid': z.pid,
                'name': z.name,
                'ppid': z.ppid,
                'parent_name': z.parent_name
            }
            for z in zombies
        ]

    def stop_all(self, timeout: int = 30) -> None:
        """Stop all services"""
        self.pm.stop_all(timeout)

    def _convert_status_to_dict(self, status: PMProcessStatus) -> Dict:
        """Convert ProcessStatus to dict (backward compatibility)"""
        return {
            'name': status.name,
            'pid': status.pid,
            'state': status.state,
            'cpu_percent': status.cpu_percent,
            'memory_mb': round(status.memory_mb, 2),
            'memory_percent': round(status.memory_percent, 2),
            'uptime_seconds': round(status.uptime_seconds, 2),
            'is_healthy': status.is_healthy,
            'restart_count': status.restart_count,
            'memory_limit_mb': status.memory_limit_mb,
            'cpu_limit_percent': status.cpu_limit_percent,
        }


# Global instance (singleton pattern for compatibility)
_process_manager_adapter: Optional[ProcessManagerAdapter] = None


def get_process_manager(
    mode: str = 'auto',
    user_mode: bool = True,
    enable_monitoring: bool = True
) -> ProcessManagerAdapter:
    """
    Get global ProcessManagerAdapter instance (singleton)

    Args:
        mode: Launcher mode
        user_mode: Use systemd --user
        enable_monitoring: Enable monitoring

    Returns:
        ProcessManagerAdapter instance
    """
    global _process_manager_adapter

    if _process_manager_adapter is None:
        _process_manager_adapter = ProcessManagerAdapter(
            mode=mode,
            user_mode=user_mode,
            enable_monitoring=enable_monitoring
        )

    return _process_manager_adapter


def reset_process_manager() -> None:
    """Reset global instance (for testing)"""
    global _process_manager_adapter
    _process_manager_adapter = None
