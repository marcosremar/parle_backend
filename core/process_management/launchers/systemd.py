"""Systemd launcher with cgroups v2 resource limits"""

import subprocess
import time
import logging
import psutil
from datetime import datetime
from typing import Optional

from .base import BaseLauncher
from ..config import ServiceConfig, ProcessStatus
from ..exceptions import ServiceStartFailedError, SystemdNotAvailableError
from ..utils.systemd_detect import has_systemd, is_systemd_unit_active

logger = logging.getLogger(__name__)


class SystemdLauncher(BaseLauncher):
    """
    Launch processes as systemd transient units with resource limits
    Uses systemd-run to create units with cgroups v2 constraints
    """

    def __init__(self, user_mode: bool = True):
        """
        Initialize systemd launcher

        Args:
            user_mode: Use systemd --user (no root required)

        Raises:
            SystemdNotAvailableError: If systemd is not available
        """
        if not has_systemd(user_mode):
            raise SystemdNotAvailableError(
                f"Systemd is not available (user_mode={user_mode})"
            )

        self.user_mode = user_mode
        logger.info(f"SystemdLauncher initialized (user_mode={user_mode})")

    def launch(self, config: ServiceConfig) -> int:
        """
        Launch process as systemd transient unit

        Args:
            config: Service configuration

        Returns:
            PID of launched process

        Raises:
            ServiceStartFailedError: If launch fails
        """
        unit_name = f"ultravox-{config.name}.service"

        # Build systemd-run command
        cmd = ["systemd-run"]

        if self.user_mode:
            cmd.append("--user")

        cmd.extend([
            f"--unit={unit_name}",
            f"--property=MemoryMax={config.memory_mb}M",
            f"--property=CPUQuota={config.cpu_percent}%",
            "--property=Restart=on-failure" if config.auto_restart else "--property=Restart=no",
            f"--property=RestartSec={config.restart_delay_seconds}",
            "--property=KillMode=mixed",  # Graceful shutdown
            f"--property=TimeoutStopSec={config.kill_timeout}",
        ])

        # Add description if provided
        if config.description:
            cmd.append(f"--property=Description={config.description}")

        # Add working directory if provided
        if config.working_dir:
            cmd.append(f"--property=WorkingDirectory={config.working_dir}")

        # Add user if provided (and not in user mode)
        if config.user and not self.user_mode:
            cmd.append(f"--property=User={config.user}")

        # Add environment variables
        if config.env:
            for key, value in config.env.items():
                cmd.append(f"--setenv={key}={value}")

        # Add the actual command
        cmd.append("--")
        cmd.extend(config.command)

        logger.info(f"Launching {config.name} via systemd-run")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise ServiceStartFailedError(
                    f"systemd-run failed: {result.stderr}"
                )

            logger.info(f"✅ {config.name} launched as {unit_name}")
            logger.debug(f"systemd-run output: {result.stdout}")

            # Wait a moment for unit to start
            time.sleep(0.5)

            # Get PID from systemd
            pid = self._get_unit_pid(unit_name)
            if pid is None:
                raise ServiceStartFailedError(
                    f"Failed to get PID for {unit_name}"
                )

            logger.info(f"✅ {config.name} running with PID {pid}")
            return pid

        except subprocess.TimeoutExpired:
            raise ServiceStartFailedError(
                f"systemd-run timed out for {config.name}"
            )
        except Exception as e:
            raise ServiceStartFailedError(
                f"Failed to launch {config.name}: {e}"
            )

    def stop(self, name: str, pid: int, timeout: int = 30) -> bool:
        """
        Stop systemd unit gracefully

        Args:
            name: Service name
            pid: Process ID (for fallback)
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully
        """
        unit_name = f"ultravox-{name}.service"

        logger.info(f"Stopping {unit_name}")

        cmd = ["systemctl"]
        if self.user_mode:
            cmd.append("--user")
        cmd.extend(["stop", unit_name])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 5  # Add buffer to systemctl timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ {unit_name} stopped")
                return True
            else:
                logger.warning(f"systemctl stop failed: {result.stderr}")

                # Fallback: try killing by PID
                if self.is_running(pid):
                    logger.info(f"Fallback: killing PID {pid} directly")
                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                        proc.wait(timeout=timeout)
                        return True
                    except Exception as e:
                        logger.error(f"Failed to kill PID {pid}: {e}")
                        return False

                return False

        except subprocess.TimeoutExpired:
            logger.error(f"systemctl stop timed out for {unit_name}")
            return False
        except Exception as e:
            logger.error(f"Error stopping {unit_name}: {e}")
            return False

    def is_running(self, pid: int) -> bool:
        """Check if process is running"""
        try:
            proc = psutil.Process(pid)
            return proc.is_running()
        except psutil.NoSuchProcess:
            return False
        except (psutil.AccessDenied, psutil.ZombieProcess, OSError):
            # Cannot determine process status due to access or OS errors
            return False

    def get_status(self, name: str, pid: int, config: ServiceConfig) -> ProcessStatus:
        """Get process status from systemd and psutil"""
        unit_name = f"ultravox-{name}.service"

        # Check if unit is active
        is_active = is_systemd_unit_active(unit_name, self.user_mode)

        if not is_active or not self.is_running(pid):
            return ProcessStatus(
                name=name,
                pid=pid,
                state="stopped",
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=config.cpu_percent,
            )

        # Get resource usage from psutil
        try:
            proc = psutil.Process(pid)
            mem_info = proc.memory_info()
            cpu_percent = proc.cpu_percent(interval=0.1)
            create_time = proc.create_time()

            return ProcessStatus(
                name=name,
                pid=pid,
                state="running",
                cpu_percent=cpu_percent,
                memory_mb=mem_info.rss / (1024 * 1024),
                memory_percent=proc.memory_percent(),
                started_at=datetime.fromtimestamp(create_time),
                uptime_seconds=time.time() - create_time,
                is_healthy=True,  # TODO: Implement health check
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=config.cpu_percent,
            )

        except psutil.NoSuchProcess:
            return ProcessStatus(
                name=name,
                pid=pid,
                state="stopped",
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=config.cpu_percent,
            )
        except Exception as e:
            logger.error(f"Error getting status for {name}: {e}")
            return ProcessStatus(
                name=name,
                pid=pid,
                state="unknown",
                error_message=str(e),
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=config.cpu_percent,
            )

    def get_launcher_type(self) -> str:
        """Get launcher type"""
        return "systemd"

    def _get_unit_pid(self, unit_name: str) -> Optional[int]:
        """
        Get PID of a systemd unit

        Args:
            unit_name: Systemd unit name

        Returns:
            PID or None if not found
        """
        cmd = ["systemctl"]
        if self.user_mode:
            cmd.append("--user")
        cmd.extend(["show", unit_name, "-p", "MainPID", "--value"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                pid_str = result.stdout.strip()
                if pid_str and pid_str != "0":
                    return int(pid_str)

            return None

        except Exception as e:
            logger.error(f"Error getting PID for {unit_name}: {e}")
            return None

    def cleanup_units(self) -> int:
        """
        Stop all ultravox systemd units

        Returns:
            Number of units stopped
        """
        logger.info("Cleaning up all ultravox systemd units")

        # List all units
        cmd = ["systemctl"]
        if self.user_mode:
            cmd.append("--user")
        cmd.extend(["list-units", "--all", "--no-pager", "--plain", "ultravox-*"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.warning(f"Failed to list units: {result.stderr}")
                return 0

            # Parse output and stop each unit
            count = 0
            for line in result.stdout.split("\n"):
                if "ultravox-" in line and ".service" in line:
                    parts = line.split()
                    if parts:
                        unit_name = parts[0]
                        logger.info(f"Stopping {unit_name}")

                        stop_cmd = ["systemctl"]
                        if self.user_mode:
                            stop_cmd.append("--user")
                        stop_cmd.extend(["stop", unit_name])

                        subprocess.run(stop_cmd, capture_output=True, timeout=30)
                        count += 1

            logger.info(f"✅ Stopped {count} systemd units")
            return count

        except Exception as e:
            logger.error(f"Error cleaning up units: {e}")
            return 0
