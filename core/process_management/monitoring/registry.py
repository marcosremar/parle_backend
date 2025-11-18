"""PID Registry - Centralized process tracking with file locking"""

import json
import fcntl
import os
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

from ..exceptions import RegistryLockError

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Information about a registered process"""

    name: str
    pid: int
    port: Optional[int]
    started_at: str  # ISO format
    command: str
    memory_limit_mb: int
    cpu_limit_percent: int
    launcher_type: str  # 'systemd' or 'subprocess'
    systemd_unit: Optional[str] = None  # For systemd launcher


class PIDRegistry:
    """
    Centralized PID registry with file locking
    Prevents race conditions and tracks all managed processes
    """

    def __init__(self, registry_path: str = str(Path.home() / ".cache" / "ultravox-pipeline" / "ultravox/pids.json"):
        """
        Initialize PID registry

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.lock_path = Path(str(registry_path) + ".lock")

        # Ensure directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize registry file if it doesn't exist
        if not self.registry_path.exists():
            self._write_registry({})

        logger.info(f"PID Registry initialized: {self.registry_path}")

    def _acquire_lock(self, timeout: float = 5.0) -> int:
        """
        Acquire exclusive lock on registry file

        Args:
            timeout: Timeout in seconds

        Returns:
            File descriptor of lock file

        Raises:
            RegistryLockError: If lock cannot be acquired
        """
        lock_fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)

        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.debug("Registry lock acquired")
                return lock_fd
            except BlockingIOError:
                if time.time() - start_time > timeout:
                    os.close(lock_fd)
                    raise RegistryLockError(
                        f"Failed to acquire registry lock after {timeout}s"
                    )
                time.sleep(0.1)

    def _release_lock(self, lock_fd: int) -> None:
        """Release lock and close file descriptor"""
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            logger.debug("Registry lock released")
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")

    def _read_registry(self) -> Dict[str, dict]:
        """Read registry from file (assumes lock is held)"""
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                return data.get("processes", {})
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Registry file corrupted or missing, resetting")
            return {}

    def _write_registry(self, processes: Dict[str, dict]) -> None:
        """Write registry to file (assumes lock is held)"""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "processes": processes,
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        pid: int,
        port: Optional[int],
        command: str,
        memory_limit_mb: int,
        cpu_limit_percent: int,
        launcher_type: str,
        systemd_unit: Optional[str] = None,
    ) -> bool:
        """
        Register a process in the registry

        Args:
            name: Service name
            pid: Process ID
            port: Port number (if applicable)
            command: Command line
            memory_limit_mb: Memory limit
            cpu_limit_percent: CPU limit
            launcher_type: 'systemd' or 'subprocess'
            systemd_unit: Systemd unit name (if using systemd)

        Returns:
            True if registered successfully, False if already registered
        """
        lock_fd = self._acquire_lock()
        try:
            processes = self._read_registry()

            # Check if already registered
            if name in processes:
                existing_pid = processes[name].get("pid")
                if self._is_pid_alive(existing_pid):
                    logger.warning(
                        f"Service {name} already registered with PID {existing_pid}"
                    )
                    return False

            # Register new process
            info = ProcessInfo(
                name=name,
                pid=pid,
                port=port,
                started_at=datetime.now().isoformat(),
                command=command,
                memory_limit_mb=memory_limit_mb,
                cpu_limit_percent=cpu_limit_percent,
                launcher_type=launcher_type,
                systemd_unit=systemd_unit,
            )

            processes[name] = asdict(info)
            self._write_registry(processes)

            logger.info(f"✅ Registered {name} (PID: {pid})")
            return True

        finally:
            self._release_lock(lock_fd)

    def unregister(self, name: str) -> bool:
        """
        Unregister a process from the registry

        Args:
            name: Service name

        Returns:
            True if unregistered, False if not found
        """
        lock_fd = self._acquire_lock()
        try:
            processes = self._read_registry()

            if name in processes:
                del processes[name]
                self._write_registry(processes)
                logger.info(f"Unregistered {name}")
                return True
            else:
                logger.warning(f"Service {name} not in registry")
                return False

        finally:
            self._release_lock(lock_fd)

    def get(self, name: str) -> Optional[ProcessInfo]:
        """
        Get process info by name

        Args:
            name: Service name

        Returns:
            ProcessInfo or None if not found
        """
        lock_fd = self._acquire_lock()
        try:
            processes = self._read_registry()
            if name in processes:
                return ProcessInfo(**processes[name])
            return None

        finally:
            self._release_lock(lock_fd)

    def get_pid(self, name: str) -> Optional[int]:
        """Get PID for a service"""
        info = self.get(name)
        return info.pid if info else None

    def is_running(self, name: str) -> bool:
        """
        Check if a service is running

        Args:
            name: Service name

        Returns:
            True if service is registered and process is alive
        """
        info = self.get(name)
        if not info:
            return False

        return self._is_pid_alive(info.pid)

    def get_all(self) -> Dict[str, ProcessInfo]:
        """
        Get all registered processes

        Returns:
            Dictionary of service_name -> ProcessInfo
        """
        lock_fd = self._acquire_lock()
        try:
            processes = self._read_registry()
            return {
                name: ProcessInfo(**data) for name, data in processes.items()
            }

        finally:
            self._release_lock(lock_fd)

    def get_all_pids(self) -> List[int]:
        """Get list of all registered PIDs"""
        all_processes = self.get_all()
        return [info.pid for info in all_processes.values()]

    def cleanup_stale(self) -> List[str]:
        """
        Remove stale entries (processes that no longer exist)

        Returns:
            List of service names that were cleaned up
        """
        lock_fd = self._acquire_lock()
        try:
            processes = self._read_registry()
            stale = []

            for name, data in list(processes.items()):
                pid = data.get("pid")
                if not self._is_pid_alive(pid):
                    stale.append(name)
                    del processes[name]
                    logger.info(f"Cleaned up stale entry: {name} (PID: {pid})")

            if stale:
                self._write_registry(processes)

            return stale

        finally:
            self._release_lock(lock_fd)

    def terminate_all(self, timeout: int = 10) -> int:
        """
        Terminate all registered processes

        Args:
            timeout: Timeout for graceful shutdown per process

        Returns:
            Number of processes terminated
        """
        all_processes = self.get_all()
        count = 0

        for name, info in all_processes.items():
            try:
                if self._is_pid_alive(info.pid):
                    logger.info(f"Terminating {name} (PID: {info.pid})")
                    proc = psutil.Process(info.pid)

                    # Try graceful shutdown first
                    proc.terminate()
                    try:
                        proc.wait(timeout=timeout)
                        logger.info(f"✅ {name} terminated gracefully")
                    except psutil.TimeoutExpired:
                        # Force kill
                        logger.warning(f"⚠️  {name} didn't stop, force killing")
                        proc.kill()
                        proc.wait(timeout=5)
                        logger.info(f"✅ {name} force killed")

                    count += 1

            except psutil.NoSuchProcess:
                logger.debug(f"{name} already dead")
            except Exception as e:
                logger.error(f"Error terminating {name}: {e}")

        # Clear registry after terminating
        lock_fd = self._acquire_lock()
        try:
            self._write_registry({})
        finally:
            self._release_lock(lock_fd)

        return count

    def _is_pid_alive(self, pid: Optional[int]) -> bool:
        """Check if a PID is alive"""
        if pid is None:
            return False

        try:
            proc = psutil.Process(pid)
            return proc.is_running()
        except psutil.NoSuchProcess:
            return False
        except Exception as e:
            logger.debug(f"Error checking PID {pid}: {e}")
            return False

    def clear(self) -> None:
        """Clear all entries from registry (dangerous!)"""
        lock_fd = self._acquire_lock()
        try:
            self._write_registry({})
            logger.warning("Registry cleared")
        finally:
            self._release_lock(lock_fd)
