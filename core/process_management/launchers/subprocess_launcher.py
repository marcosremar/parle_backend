"""Subprocess launcher with resource limits via resource module"""

import subprocess
import logging
import psutil
import time
from datetime import datetime
from typing import Dict, Optional

from .base import BaseLauncher
from ..config import ServiceConfig, ProcessStatus
from ..exceptions import ServiceStartFailedError
from ..utils.resource_limits import create_preexec_fn

logger = logging.getLogger(__name__)


class SubprocessLauncher(BaseLauncher):
    """
    Launch processes as subprocesses with resource limits
    Uses resource.setrlimit() to apply memory/CPU constraints
    """

    def __init__(self):
        """Initialize subprocess launcher"""
        self.processes: Dict[str, subprocess.Popen] = {}
        logger.info("SubprocessLauncher initialized")

    def launch(self, config: ServiceConfig) -> int:
        """
        Launch process as subprocess with resource limits

        Args:
            config: Service configuration

        Returns:
            PID of launched process

        Raises:
            ServiceStartFailedError: If launch fails
        """
        logger.info(f"Launching {config.name} as subprocess")
        logger.debug(f"Command: {' '.join(config.command)}")

        # Create preexec function with resource limits
        preexec_fn = create_preexec_fn(
            memory_mb=config.memory_mb,
            cpu_nice=config.cpu_nice,
            max_fds=config.max_fds
        )

        # Prepare environment
        env = None
        if config.env:
            import os
            env = os.environ.copy()
            env.update(config.env)

        try:
            # Launch subprocess
            proc = subprocess.Popen(
                config.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec_fn,
                cwd=config.working_dir,
                env=env,
            )

            # Wait a moment to check if it started successfully
            time.sleep(0.2)

            if proc.poll() is not None:
                # Process died immediately
                stdout, stderr = proc.communicate()
                raise ServiceStartFailedError(
                    f"{config.name} died immediately. "
                    f"Exit code: {proc.returncode}. "
                    f"Stderr: {stderr.decode()[:500]}"
                )

            # Store process reference
            self.processes[config.name] = proc

            logger.info(f"✅ {config.name} launched with PID {proc.pid}")
            return proc.pid

        except FileNotFoundError:
            raise ServiceStartFailedError(
                f"Command not found: {config.command[0]}"
            )
        except Exception as e:
            raise ServiceStartFailedError(
                f"Failed to launch {config.name}: {e}"
            )

    def stop(self, name: str, pid: int, timeout: int = 30) -> bool:
        """
        Stop subprocess gracefully

        Args:
            name: Service name
            pid: Process ID
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully
        """
        logger.info(f"Stopping {name} (PID: {pid})")

        try:
            proc = psutil.Process(pid)

            # Try graceful shutdown first (SIGTERM)
            proc.terminate()

            try:
                proc.wait(timeout=timeout)
                logger.info(f"✅ {name} stopped gracefully")

                # Remove from tracked processes
                if name in self.processes:
                    del self.processes[name]

                return True

            except psutil.TimeoutExpired:
                # Force kill (SIGKILL)
                logger.warning(f"⚠️  {name} didn't stop, force killing")
                proc.kill()

                try:
                    proc.wait(timeout=5)
                    logger.info(f"✅ {name} force killed")

                    # Remove from tracked processes
                    if name in self.processes:
                        del self.processes[name]

                    return True

                except psutil.TimeoutExpired:
                    logger.error(f"❌ Failed to kill {name}")
                    return False

        except psutil.NoSuchProcess:
            logger.info(f"{name} already stopped")

            # Remove from tracked processes
            if name in self.processes:
                del self.processes[name]

            return True

        except Exception as e:
            logger.error(f"Error stopping {name}: {e}")
            return False

    def is_running(self, pid: int) -> bool:
        """Check if process is running"""
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except (psutil.AccessDenied, psutil.ZombieProcess, OSError):
            # Cannot determine process status due to access or OS errors
            return False

    def get_status(self, name: str, pid: int, config: ServiceConfig) -> ProcessStatus:
        """Get process status from psutil"""
        if not self.is_running(pid):
            return ProcessStatus(
                name=name,
                pid=pid,
                state="stopped",
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=0,  # No CPU quota in subprocess mode
            )

        try:
            proc = psutil.Process(pid)
            mem_info = proc.memory_info()
            cpu_percent = proc.cpu_percent(interval=0.1)
            create_time = proc.create_time()

            # Check if exceeding memory limit
            mem_mb = mem_info.rss / (1024 * 1024)
            if mem_mb > config.memory_mb * 1.1:  # 10% tolerance
                logger.warning(
                    f"⚠️  {name} exceeding memory limit: "
                    f"{mem_mb:.1f}MB > {config.memory_mb}MB"
                )

            return ProcessStatus(
                name=name,
                pid=pid,
                state="running",
                cpu_percent=cpu_percent,
                memory_mb=mem_mb,
                memory_percent=proc.memory_percent(),
                started_at=datetime.fromtimestamp(create_time),
                uptime_seconds=time.time() - create_time,
                is_healthy=True,  # TODO: Implement health check
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=0,
            )

        except psutil.NoSuchProcess:
            return ProcessStatus(
                name=name,
                pid=pid,
                state="stopped",
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=0,
            )
        except Exception as e:
            logger.error(f"Error getting status for {name}: {e}")
            return ProcessStatus(
                name=name,
                pid=pid,
                state="unknown",
                error_message=str(e),
                memory_limit_mb=config.memory_mb,
                cpu_limit_percent=0,
            )

    def get_launcher_type(self) -> str:
        """Get launcher type"""
        return "subprocess"

    def cleanup_all(self) -> int:
        """
        Stop all tracked subprocesses

        Returns:
            Number of processes stopped
        """
        logger.info("Cleaning up all tracked subprocesses")

        count = 0
        for name, proc in list(self.processes.items()):
            try:
                if proc.poll() is None:  # Still running
                    logger.info(f"Stopping {name} (PID: {proc.pid})")
                    self.stop(name, proc.pid)
                    count += 1
                else:
                    # Already stopped
                    del self.processes[name]

            except Exception as e:
                logger.error(f"Error cleaning up {name}: {e}")

        logger.info(f"✅ Cleaned up {count} subprocesses")
        return count
