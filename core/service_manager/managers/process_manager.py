"""
Process Manager - Handles service process lifecycle and monitoring.

Part of Service Manager refactoring (Phase 3).
Manages process PIDs, monitoring, and cleanup.
"""

from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path
from datetime import datetime
import asyncio
import psutil
import signal
import threading
import time
import subprocess
import requests
from loguru import logger


class ProcessManager:
    """
    Manages service processes - PID tracking, monitoring, cleanup.

    Responsibilities:
    - Track process PIDs for services
    - Monitor service startup/shutdown
    - Kill processes on ports
    - Check port availability
    - Clean up zombie processes

    SOLID Principles:
    - Single Responsibility: Only handles process management
    - Open/Closed: Easy to add new monitoring strategies
    """

    def __init__(self):
        """Initialize process manager."""
        self.monitor_threads: List[threading.Thread] = []
        self.pid_cache: Dict[str, int] = {}  # service_id -> PID
        logger.info("🔧 Process Manager initialized")

    def get_service_pid(self, script_name: str) -> Optional[int]:
        """
        Get PID of a service by script name.

        Args:
            script_name: Name of the service script (e.g., "orchestrator/service.py")

        Returns:
            PID if found, None otherwise
        """
        try:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any(script_name in arg for arg in cmdline):
                    pid = proc.info['pid']
                    logger.debug(f"Found PID {pid} for {script_name}")
                    return pid
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return None

    def kill_process_on_port(self, port: int) -> bool:
        """
        Kill process using a specific port.

        Args:
            port: Port number

        Returns:
            True if process was killed, False if no process found
        """
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    try:
                        proc = psutil.Process(conn.pid)
                        logger.warning(f"Killing process {conn.pid} on port {port}")
                        proc.terminate()
                        proc.wait(timeout=5)
                        return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except Exception as e:
            logger.error(f"Error killing process on port {port}: {e}")
        return False

    def check_port(self, port: int) -> bool:
        """
        Check if a port is available.

        Args:
            port: Port number to check

        Returns:
            True if port is available, False if in use
        """
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                result = s.connect_ex(('localhost', port))
                return result != 0  # Port is available if connect fails
        except Exception:
            return True

    def monitor_service_startup(
        self,
        service_id: str,
        port: int,
        timeout: int = 30,
        callback: Optional[Callable] = None
    ):
        """
        Monitor service startup in background thread.

        Args:
            service_id: Service identifier
            port: Port to monitor
            timeout: Max time to wait (seconds)
            callback: Optional callback when service is ready
        """
        def _monitor():
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not self.check_port(port):
                    logger.info(f"✅ {service_id} is ready on port {port}")
                    if callback:
                        callback(service_id, True)
                    return
                time.sleep(1)

            logger.warning(f"⏱️  {service_id} startup timeout after {timeout}s")
            if callback:
                callback(service_id, False)

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        self.monitor_threads.append(thread)

    def monitor_service_stop(
        self,
        service_id: str,
        pid: int,
        timeout: int = 10,
        callback: Optional[Callable] = None
    ):
        """
        Monitor service shutdown in background thread.

        Args:
            service_id: Service identifier
            pid: Process ID to monitor
            timeout: Max time to wait (seconds)
            callback: Optional callback when service stops
        """
        def _monitor():
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    proc = psutil.Process(pid)
                    if not proc.is_running():
                        logger.info(f"✅ {service_id} (PID {pid}) stopped gracefully")
                        if callback:
                            callback(service_id, True)
                        return
                except psutil.NoSuchProcess:
                    logger.info(f"✅ {service_id} (PID {pid}) stopped")
                    if callback:
                        callback(service_id, True)
                    return
                time.sleep(0.5)

            # Force kill after timeout
            try:
                proc = psutil.Process(pid)
                logger.warning(f"⚠️  Force killing {service_id} (PID {pid})")
                proc.kill()
            except psutil.NoSuchProcess:
                pass

            if callback:
                callback(service_id, False)

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        self.monitor_threads.append(thread)

    def cleanup(self):
        """Clean up monitor threads."""
        for thread in self.monitor_threads:
            if thread.is_alive():
                thread.join(timeout=1)
        self.monitor_threads.clear()
        logger.debug("Process monitor threads cleaned up")

    def check_gpu_memory(self) -> Optional[int]:
        """
        Check available GPU memory in MB.

        Returns:
            Free GPU memory in MB, or None if no GPU available
        """
        try:
            # Run nvidia-smi to get GPU memory info
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse the output (returns free memory in MB)
                memory_free = int(result.stdout.strip())
                logger.debug(f"GPU free memory: {memory_free} MB")
                return memory_free
            else:
                logger.warning("Could not check GPU memory: nvidia-smi returned error")
                return None

        except FileNotFoundError:
            logger.debug("nvidia-smi not found - GPU may not be available")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("GPU memory check timed out")
            return None
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return None

    def check_service_health(
        self,
        service_id: str,
        port: int,
        is_module: bool = False,
        timeout: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Check health status of a service.

        Args:
            service_id: Service identifier
            port: Service port
            is_module: True if service runs in-process (MODULE mode)
            timeout: HTTP request timeout in seconds

        Returns:
            Tuple of (status, details) where status is "healthy", "unhealthy", or "unknown"
        """
        # MODULE services (in-process) are always healthy if loaded
        if is_module:
            return ("healthy", {
                "type": "module",
                "message": "In-process service (always healthy)"
            })

        # For external services, try HTTP health endpoint
        try:
            health_url = f"http://localhost:{port}/health"
            response = requests.get(health_url, timeout=timeout)

            if response.status_code == 200:
                try:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    return (status, data)
                except Exception:
                    # Response is not JSON, but 200 OK means healthy
                    return ("healthy", {"message": "Service responding (non-JSON)"})
            else:
                return ("unhealthy", {
                    "message": f"HTTP {response.status_code}",
                    "url": health_url
                })

        except requests.exceptions.ConnectionError:
            return ("unhealthy", {
                "message": "Connection refused",
                "url": f"http://localhost:{port}/health"
            })
        except requests.exceptions.Timeout:
            return ("unhealthy", {
                "message": f"Timeout after {timeout}s",
                "url": f"http://localhost:{port}/health"
            })
        except Exception as e:
            return ("unknown", {
                "message": f"Error: {str(e)}",
                "url": f"http://localhost:{port}/health"
            })
