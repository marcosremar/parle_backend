"""ProcessManager - High-level API for process management"""

import asyncio
import logging
import os
from typing import Dict, List, Optional
from pathlib import Path

# Load environment configuration

from .config import ServiceConfig, ProcessStatus, OrphanInfo, ZombieInfo
from .launchers import BaseLauncher, SystemdLauncher, SubprocessLauncher
from .monitoring import PIDRegistry, ZombieMonitor, OrphanCleaner
from .utils.systemd_detect import has_systemd
from .exceptions import (
    ServiceNotFoundError,
    ServiceAlreadyRunningError,
    ServiceStartFailedError,
    SystemdNotAvailableError,
)

logger = logging.getLogger(__name__)


class ProcessManager:
    """
    High-level process manager
    Combines systemd/subprocess launching with monitoring
    """

    def __init__(
        self,
        mode: str = 'auto',
        user_mode: bool = True,
        registry_path: Optional[str] = None,
        log_level: str = 'INFO',
        enable_monitoring: bool = True,
        monitoring_interval: int = 30,
    ):
        """
        Initialize process manager

        Args:
            mode: Launcher mode ('auto', 'systemd', 'subprocess')
            user_mode: Use systemd --user (no root required)
            registry_path: Path to PID registry file (reads from PID_REGISTRY_PATH env var if not provided)
            log_level: Logging level
            enable_monitoring: Enable background monitoring
            monitoring_interval: Monitoring check interval in seconds
        """
        # Use registry_path from environment if not provided
        if registry_path is None:
            registry_path = os.getenv("PID_REGISTRY_PATH")
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.mode = mode
        self.user_mode = user_mode
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval

        # Initialize PID registry
        self.pid_registry = PIDRegistry(registry_path)

        # Select and initialize launcher
        self.launcher = self._select_launcher(mode, user_mode)

        # Service configurations
        self.services: Dict[str, ServiceConfig] = {}

        # Monitoring components
        self.zombie_monitor: Optional[ZombieMonitor] = None
        self.orphan_cleaner: Optional[OrphanCleaner] = None
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info(
            f"ProcessManager initialized "
            f"(mode={mode}, launcher={self.launcher.get_launcher_type()})"
        )

    def _select_launcher(self, mode: str, user_mode: bool) -> BaseLauncher:
        """Select appropriate launcher based on mode and availability"""
        if mode == 'systemd':
            if not has_systemd(user_mode):
                raise SystemdNotAvailableError(
                    f"Systemd not available (user_mode={user_mode})"
                )
            logger.info("Using SystemdLauncher (forced)")
            return SystemdLauncher(user_mode=user_mode)

        elif mode == 'subprocess':
            logger.info("Using SubprocessLauncher (forced)")
            return SubprocessLauncher()

        elif mode == 'auto':
            if has_systemd(user_mode):
                logger.info("Using SystemdLauncher (auto-detected)")
                return SystemdLauncher(user_mode=user_mode)
            else:
                logger.info("Using SubprocessLauncher (systemd not available)")
                return SubprocessLauncher()

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def register(self, config: ServiceConfig) -> None:
        """
        Register a service (doesn't start it)

        Args:
            config: Service configuration

        Raises:
            ValueError: If service name already registered
        """
        if config.name in self.services:
            raise ValueError(f"Service {config.name} already registered")

        self.services[config.name] = config
        logger.info(f"📝 Registered service: {config.name}")

    def start(self, name: str) -> bool:
        """
        Start a specific service

        Args:
            name: Service name

        Returns:
            True if started successfully

        Raises:
            ServiceNotFoundError: If service not registered
            ServiceAlreadyRunningError: If service already running
            ServiceStartFailedError: If launch fails
        """
        if name not in self.services:
            raise ServiceNotFoundError(f"Service {name} not registered")

        config = self.services[name]

        # Check if already running
        if self.pid_registry.is_running(name):
            existing_pid = self.pid_registry.get_pid(name)
            raise ServiceAlreadyRunningError(
                f"Service {name} already running (PID: {existing_pid})"
            )

        logger.info(f"🚀 Starting {name}...")

        try:
            # Launch process
            pid = self.launcher.launch(config)

            # Register in PID registry
            self.pid_registry.register(
                name=name,
                pid=pid,
                port=config.port,
                command=' '.join(config.command),
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=config.cpu_percent,
                launcher_type=self.launcher.get_launcher_type(),
                systemd_unit=(
                    f"ultravox-{name}.service"
                    if self.launcher.get_launcher_type() == 'systemd'
                    else None
                )
            )

            logger.info(f"✅ {name} started successfully (PID: {pid})")
            return True

        except ServiceStartFailedError:
            raise
        except Exception as e:
            raise ServiceStartFailedError(f"Failed to start {name}: {e}")

    def stop(self, name: str, timeout: int = 30) -> bool:
        """
        Stop a specific service gracefully

        Args:
            name: Service name
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully

        Raises:
            ServiceNotFoundError: If service not registered or not running
        """
        if name not in self.services:
            raise ServiceNotFoundError(f"Service {name} not registered")

        info = self.pid_registry.get(name)
        if not info:
            raise ServiceNotFoundError(f"Service {name} not running")

        logger.info(f"🛑 Stopping {name}...")

        config = self.services[name]
        success = self.launcher.stop(name, info.pid, timeout)

        if success:
            # Unregister from PID registry
            self.pid_registry.unregister(name)
            logger.info(f"✅ {name} stopped")
        else:
            logger.error(f"❌ Failed to stop {name}")

        return success

    def restart(self, name: str) -> bool:
        """
        Restart a service

        Args:
            name: Service name

        Returns:
            True if restarted successfully
        """
        logger.info(f"🔄 Restarting {name}...")

        # Stop if running
        if self.pid_registry.is_running(name):
            self.stop(name)

        # Wait a moment
        import time
        time.sleep(1)

        # Start again
        return self.start(name)

    def start_all(self) -> Dict[str, bool]:
        """
        Start all registered services

        Returns:
            Dictionary of service_name -> success
        """
        logger.info("🚀 Starting all services...")

        results = {}
        for name in self.services.keys():
            try:
                success = self.start(name)
                results[name] = success
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")
                results[name] = False

        success_count = sum(1 for v in results.values() if v)
        total = len(results)

        logger.info(f"📊 Started {success_count}/{total} services")
        return results

    def stop_all(self, timeout: int = 30) -> None:
        """
        Stop all services gracefully

        Args:
            timeout: Timeout per service
        """
        logger.info("🛑 Stopping all services...")

        # Stop monitoring first
        if self.monitoring_task:
            self.monitoring_task.cancel()

        # Get all running services from registry
        all_running = self.pid_registry.get_all()

        for name in all_running.keys():
            try:
                self.stop(name, timeout)
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        logger.info("✅ All services stopped")

    def status(self, name: str) -> ProcessStatus:
        """
        Get status of a service

        Args:
            name: Service name

        Returns:
            ProcessStatus

        Raises:
            ServiceNotFoundError: If service not registered
        """
        if name not in self.services:
            raise ServiceNotFoundError(f"Service {name} not registered")

        config = self.services[name]
        info = self.pid_registry.get(name)

        if not info:
            return ProcessStatus(
                name=name,
                pid=None,
                state='stopped'
            )

        return self.launcher.get_status(name, info.pid, config)

    def status_all(self) -> Dict[str, ProcessStatus]:
        """
        Get status of all services

        Returns:
            Dictionary of service_name -> ProcessStatus
        """
        statuses = {}

        for name in self.services.keys():
            try:
                statuses[name] = self.status(name)
            except Exception as e:
                logger.error(f"Error getting status for {name}: {e}")
                statuses[name] = ProcessStatus(
                    name=name,
                    pid=None,
                    state='unknown',
                    error_message=str(e)
                )

        return statuses

    def cleanup_orphans(self, dry_run: bool = False) -> int:
        """
        Find and terminate orphan processes

        Args:
            dry_run: If True, only log what would be done

        Returns:
            Number of orphans terminated
        """
        if not self.orphan_cleaner:
            self.orphan_cleaner = OrphanCleaner(self.pid_registry)

        return self.orphan_cleaner.cleanup_orphans(dry_run=dry_run)

    def get_orphans(self) -> List[OrphanInfo]:
        """
        Get list of orphan processes

        Returns:
            List of OrphanInfo objects
        """
        if not self.orphan_cleaner:
            self.orphan_cleaner = OrphanCleaner(self.pid_registry)

        return self.orphan_cleaner.find_orphans()

    def get_zombies(self) -> List[ZombieInfo]:
        """
        Get list of zombie processes

        Returns:
            List of ZombieInfo objects
        """
        if not self.zombie_monitor:
            self.zombie_monitor = ZombieMonitor(self.pid_registry)

        return self.zombie_monitor.find_zombies()

    async def start_monitoring(self) -> None:
        """Start background monitoring (async)"""
        if not self.enable_monitoring:
            logger.warning("Monitoring is disabled")
            return

        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Monitoring already running")
            return

        # Initialize monitoring components
        if not self.zombie_monitor:
            self.zombie_monitor = ZombieMonitor(
                self.pid_registry,
                check_interval=self.monitoring_interval
            )

        if not self.orphan_cleaner:
            self.orphan_cleaner = OrphanCleaner(self.pid_registry)

        # Start zombie monitor
        await self.zombie_monitor.start()

        logger.info("✅ Background monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        if self.zombie_monitor:
            await self.zombie_monitor.stop()

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Monitoring stopped")

    def __enter__(self):
        """Context manager: start all services"""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: stop all services"""
        self.stop_all()
        return False
