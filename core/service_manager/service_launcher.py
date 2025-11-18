"""
Service Launcher
Manages launching services in external mode (as separate HTTP processes)

Supports two launch methods:
1. Systemd: Delegates to systemd for production deployment
2. Subprocess: Directly spawns Python process for development
"""

import asyncio
import subprocess
import logging
import os
import pwd
import grp
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import psutil
import aiohttp

from src.utils.venv_manager import VenvManager
from src.core.service_manager.core import HEALTH_CHECK_TIMEOUT
from src.core.utils.port_pool import get_port_pool
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)


def get_user_preexec_fn() -> Optional[Callable]:
    """
    Create preexec_fn to preserve current user when launching subprocesses.

    This ensures that when Service Manager starts services, they run as the
    CURRENT USER (e.g., 'marcos') instead of inheriting the Service Manager's
    user (e.g., 'ubuntu').

    Returns:
        Callable to pass to subprocess.Popen(preexec_fn=...) or None if not available
    """
    try:
        # Get current user info
        current_uid = os.getuid()
        current_gid = os.getgid()

        # Get username for logging
        try:
            username = pwd.getpwuid(current_uid).pw_name
            logger.info(f"🔐 Subprocess will run as user: {username} (UID: {current_uid}, GID: {current_gid})")
        except KeyError:
            logger.warning(f"⚠️  Could not resolve UID {current_uid} to username")

        # Create preexec function that sets user/group
        def preexec_fn():
            """Set process user and group to current user"""
            try:
                # Set group first (must be done before setuid)
                os.setgid(current_gid)
                # Then set user
                os.setuid(current_uid)
            except Exception as e:
                logger.error(f"❌ Failed to set user/group in subprocess: {e}")
                raise

        return preexec_fn

    except Exception as e:
        logger.warning(f"⚠️  Could not create user preexec function: {e}")
        return None


class ServiceLauncher:
    """Launches and manages external service processes"""

    def __init__(self, use_port_pool: bool = True):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.processes_lock = asyncio.Lock()  # Protect processes dict from concurrent access
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.venv_manager = VenvManager()

        # Port Pool Manager (optional for backward compatibility)
        self.use_port_pool = use_port_pool
        self.port_pool = get_port_pool() if use_port_pool else None

        # Get preexec_fn for preserving current user
        self.user_preexec_fn = get_user_preexec_fn()

    async def launch_external_service(
        self,
        service_name: str,
        port: int,
        module_path: str,
        use_systemd: bool = False,
        venv_path: Optional[str] = None,
        execution_mode: str = "external"
    ) -> bool:
        """
        Launch a service in external mode

        Args:
            service_name: Service identifier (e.g., "session")
            port: Port to run on
            module_path: Python module path (e.g., "src.services.session.service")
            use_systemd: If True, use systemd; otherwise use subprocess
            venv_path: Path to virtual environment (if any)
            execution_mode: Execution mode (external, internal, module) - default "external"

        Returns:
            bool: True if launch successful
        """
        logger.info(f"🚀 Launching external service: {service_name} on port {port} (mode: {execution_mode})")

        if use_systemd:
            return await self.launch_via_systemd(service_name)
        else:
            return await self.launch_via_subprocess(service_name, port, module_path, venv_path, execution_mode)

    async def launch_via_systemd(self, service_name: str) -> bool:
        """
        Launch service via systemd

        Args:
            service_name: Service name

        Returns:
            bool: True if successfully started
        """
        try:
            # Start systemd service
            result = subprocess.run(
                ["systemctl", "--user", "start", f"ultravox-{service_name}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info(f"✅ Started {service_name} via systemd")
                return True
            else:
                logger.error(f"❌ Systemd start failed for {service_name}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Systemd start timeout for {service_name}")
            return False
        except FileNotFoundError:
            logger.error("❌ systemctl not found - systemd not available")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to start {service_name} via systemd: {e}")
            return False

    def _find_available_port(self, preferred_port: int, service_name: str) -> Optional[int]:
        """
        Find an available port, trying preferred port first, then fallbacks.

        If Port Pool Manager is enabled (use_port_pool=True):
          - Uses centralized port allocation from port pool
          - Preserves port history for service restarts
          - Automatic reallocation on crash/recovery

        Otherwise (legacy mode):
          - Auto-fallback strategy: port → port+100 → port+200 → port+1000

        Args:
            preferred_port: The desired port
            service_name: Service name for logging

        Returns:
            Available port or None if all attempts fail
        """
        # Use Port Pool Manager if enabled (Phase 2 improvement)
        if self.use_port_pool and self.port_pool:
            try:
                port = self.port_pool.allocate_port(
                    service_name=service_name,
                    preferred_port=preferred_port,
                    allow_reuse=True  # Reuse existing allocation if service already has one
                )
                logger.info(f"🎱 Port Pool allocated port {port} to {service_name}")
                return port
            except RuntimeError as e:
                logger.error(f"❌ Port Pool exhausted for {service_name}: {e}")
                return None

        # Legacy port fallback (backward compatibility)
        import socket as sock_module

        fallback_offsets = [0, 100, 200, 1000]

        for offset in fallback_offsets:
            candidate_port = preferred_port + offset

            # Check if port is available
            try:
                # Try to bind to port
                test_socket = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
                test_socket.setsockopt(sock_module.SOL_SOCKET, sock_module.SO_REUSEADDR, 1)
                test_socket.bind(('0.0.0.0', candidate_port))
                test_socket.close()

                if offset == 0:
                    logger.info(f"✅ Port {candidate_port} is available for {service_name}")
                else:
                    logger.warning(f"⚠️  Port {preferred_port} unavailable, using fallback port {candidate_port} for {service_name}")

                return candidate_port

            except OSError:
                if offset == 0:
                    logger.debug(f"Port {candidate_port} unavailable, trying fallbacks...")
                else:
                    logger.debug(f"Fallback port {candidate_port} also unavailable...")
                continue

        logger.error(f"❌ No available port found for {service_name} (tried {preferred_port}, {preferred_port+100}, {preferred_port+200}, {preferred_port+1000})")
        return None

    async def launch_via_subprocess(
        self,
        service_name: str,
        port: int,
        module_path: str,
        venv_path: Optional[str] = None,
        execution_mode: str = "external"
    ) -> bool:
        """
        Launch service as subprocess using HTTP server template

        Features:
        - Automatic port fallback if preferred port is unavailable
        - Tries: port, port+100, port+200, port+1000
        - Handles zombie sockets gracefully

        Args:
            service_name: Service name
            port: Preferred port to run on
            module_path: Python module path
            venv_path: Path to virtual environment (optional)
            execution_mode: Execution mode (external, internal, module) - default "external"

        Returns:
            bool: True if successfully started
        """
        try:
            # Auto-fallback: Find available port
            available_port = self._find_available_port(port, service_name)

            if available_port is None:
                logger.error(f"❌ Could not find available port for {service_name}")
                return False

            # Use the available port (might be fallback)
            actual_port = available_port

            # Simple port cleanup (no excessive delays)
            self._kill_process_on_port(actual_port, max_retries=1)

            # Determine Python executable
            if venv_path:
                # Ensure venv exists
                if not self.venv_manager.venv_exists(service_name):
                    logger.info(f"🔨 Creating virtual environment for {service_name}...")
                    if not self.venv_manager.create_venv(service_name):
                        logger.error(f"❌ Failed to create venv for {service_name}")
                        return False

                # Get venv Python executable
                python_executable = self.venv_manager.get_python_executable(service_name)
                if not python_executable:
                    logger.error(f"❌ Could not find Python executable in venv for {service_name}")
                    return False

                logger.info(f"🐍 Using virtual environment Python: {python_executable}")
            else:
                python_executable = "python3"
                logger.debug(f"🐍 Using system Python: {python_executable}")

            # Build command
            http_server_script = self.project_root / "src" / "core" / "http_server_template.py"

            cmd = [
                str(python_executable),
                str(http_server_script),
                "--service", service_name,
                "--port", str(actual_port),  # Use actual port (might be fallback)
                "--module", module_path,
                "--execution-mode", execution_mode
            ]

            # Set up log file (using portable cache directory)
            log_dir = Path.home() / ".cache" / "ultravox-pipeline" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{service_name}_external.log"

            logger.info(f"📝 Service logs: {log_file}")
            logger.info(f"🔧 Command: {' '.join(cmd)}")

            # ✅ Phase 3b: Set port environment variable for services that read it
            # This ensures the service can access its allocated port via environment variable
            service_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

            # Map service names to their port environment variables
            port_env_vars = {
                "api_gateway": "API_GATEWAY_PORT",
                "webrtc": "WEBRTC_PORT",
                "webrtc_signaling": "WEBRTC_SIGNALING_PORT",
                "websocket": "WEBSOCKET_PORT",
                "llm": "LLM_PORT",
                "tts": "TTS_PORT",
                "stt": "STT_PORT",
                "conversation_store": "CONVERSATION_STORE_PORT",
                "orchestrator": "ORCHESTRATOR_PORT",
                "user": "USER_PORT",
                "file_storage": "FILE_STORAGE_PORT",
                "session": "SESSION_PORT",
                "scenarios": "SCENARIOS_PORT",
                "rest_polling": "REST_POLLING_PORT",
                "external_llm": "EXTERNAL_LLM_PORT",
                "external_stt": "EXTERNAL_STT_PORT",
                "external_ultravox": "EXTERNAL_ULTRAVOX_PORT",
            }

            # Set port environment variable if this service has one defined
            if service_name in port_env_vars:
                service_env[port_env_vars[service_name]] = str(actual_port)
                logger.info(f"🌍 Set {port_env_vars[service_name]}={actual_port}")

            # Start process with current user preserved
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.project_root),
                    env=service_env,
                    preexec_fn=self.user_preexec_fn  # Preserve current user
                )

            self.processes[service_name] = process

            # Wait for service to be ready
            # Increased timeout to 40s for heavy services (session, scenarios, user, file_storage)
            # These services load database storage and may take longer during parallel startup
            ready = await self._wait_for_service(actual_port, timeout=40)

            if ready:
                if actual_port != port:
                    logger.info(f"✅ Service {service_name} started (PID: {process.pid}, Port: {actual_port} [fallback from {port}])")
                else:
                    logger.info(f"✅ Service {service_name} started (PID: {process.pid}, Port: {actual_port})")
                return True
            else:
                logger.error(f"❌ Service {service_name} failed to start")
                process.kill()
                del self.processes[service_name]
                return False

        except Exception as e:
            logger.error(f"❌ Failed to launch {service_name} via subprocess: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _wait_for_service(self, port: int, timeout: float = 10) -> bool:
        """
        Wait for service to be ready on given port using exponential backoff

        Retry strategy:
        - Use SHORT HTTP timeout (1s) to allow many quick attempts
        - Start with 0.2s delay between attempts
        - Increase to 0.5s → 1s → 2s for later attempts
        - This allows fast services to start quickly while giving slow services time

        Args:
            port: Port to check
            timeout: Max seconds to wait (overall timeout, not per-request)

        Returns:
            bool: True if service is responding
        """
        import aiohttp

        start_time = time.time()
        attempt = 0
        delay = 0.2  # Start with 200ms

        # Use SHORT timeout per HTTP request (1s) to allow many attempts
        # This is different from the overall timeout parameter
        http_timeout = aiohttp.ClientTimeout(total=1.0)

        while time.time() - start_time < timeout:
            attempt += 1
            try:
                async with aiohttp.ClientSession(timeout=http_timeout) as session:
                    async with session.get(f"http://localhost:{port}/health") as resp:
                        if resp.status == 200:
                            elapsed = time.time() - start_time
                            logger.debug(f"✅ Health check passed for port {port} after {elapsed:.1f}s ({attempt} attempts)")
                            return True
            except Exception as e:
                # Service not ready yet, will retry
                logger.debug(f"Health check failed for port {port} (attempt {attempt}): {e}")

            # Exponential backoff: 0.2s → 0.5s → 1s → 2s (max)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)

        logger.warning(f"⚠️  Health check timeout for port {port} after {timeout}s ({attempt} attempts)")
        return False

    def _kill_process_on_port(self, port: int, max_retries: int = 3):
        """
        Kill any process using the given port with retry logic

        Improvements:
        - Increased timeout from 3s to 10s for graceful shutdown
        - Added retry logic (up to 3 attempts)
        - Enhanced logging with process details (name, cmdline)
        - Post-cleanup verification to ensure port is free

        Args:
            port: Port number
            max_retries: Maximum number of retry attempts (default: 3)
        """
        # Protected ports - DO NOT KILL processes on these ports
        PROTECTED_PORTS = {
            22,     # SSH
            80,     # HTTP
            443,    # HTTPS
            3306,   # MySQL
            5432,   # PostgreSQL
            6379,   # Redis
            27017,  # MongoDB
        }

        # Also protect all system ports (< 1024) except our service range
        if port < 1024 or port in PROTECTED_PORTS:
            logger.warning(f"⚠️  Port {port} is protected - will not kill processes on this port")
            return

        for attempt in range(1, max_retries + 1):
            try:
                process_found = False

                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        connections = proc.connections(kind='tcp')
                        for conn in connections:
                            if conn.laddr.port == port and conn.status == 'LISTEN':
                                process_found = True

                                # Enhanced logging with process details
                                try:
                                    cmdline = ' '.join(proc.cmdline()[:3]) if proc.cmdline() else 'N/A'
                                    logger.warning(
                                        f"⚠️  Attempt {attempt}/{max_retries}: "
                                        f"Killing process {proc.pid} ({proc.name()}) on port {port}"
                                    )
                                    logger.debug(f"   Process cmdline: {cmdline}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    logger.warning(f"⚠️  Attempt {attempt}/{max_retries}: Killing process {proc.pid} on port {port}")

                                # Try graceful termination with moderate timeout
                                proc.terminate()
                                try:
                                    proc.wait(timeout=3)
                                    logger.info(f"✅ Process {proc.pid} terminated gracefully")
                                except psutil.TimeoutExpired:
                                    # Force kill if graceful fails
                                    logger.warning(f"⚠️  Process {proc.pid} did not terminate gracefully, forcing kill")
                                    try:
                                        proc.kill()
                                        proc.wait(timeout=5)
                                        logger.info(f"✅ Process {proc.pid} force killed")
                                    except psutil.TimeoutExpired:
                                        # Last resort: SIGKILL again and don't wait
                                        logger.error(f"❌ Process {proc.pid} extremely stubborn, sending final SIGKILL")
                                        try:
                                            proc.kill()
                                            # Don't wait - let OS handle it
                                        except Exception as e:
                                            pass

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Post-cleanup verification: Check if port is now free
                if process_found:
                    # Wait for OS to fully release port (kernel socket cleanup)
                    # This is CRITICAL for Python/uvicorn processes
                    time.sleep(2.0)
                    port_free = self._verify_port_free(port)

                    if port_free:
                        logger.info(f"✅ Port {port} successfully cleaned up (attempt {attempt}/{max_retries})")
                        return
                    else:
                        if attempt < max_retries:
                            logger.warning(f"⚠️  Port {port} still occupied after cleanup, retrying...")
                            time.sleep(1)
                        else:
                            logger.error(f"❌ Failed to clean up port {port} after {max_retries} attempts")
                else:
                    # No process found on port
                    logger.debug(f"ℹ️  No process found on port {port}")
                    return

            except Exception as e:
                logger.warning(f"⚠️  Error checking port {port} (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(1)

    def _verify_port_free(self, port: int) -> bool:
        """
        Verify that a port is free (no process listening)

        Args:
            port: Port number to check

        Returns:
            bool: True if port is free, False if occupied
        """
        try:
            for proc in psutil.process_iter(['pid']):
                try:
                    connections = proc.connections(kind='tcp')
                    for conn in connections:
                        if conn.laddr.port == port and conn.status == 'LISTEN':
                            logger.debug(f"Port {port} still occupied by PID {proc.pid}")
                            return False
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return True
        except Exception as e:
            logger.warning(f"⚠️  Error verifying port {port}: {e}")
            return False

    async def stop_service(self, service_name: str) -> bool:
        """
        Stop an external service

        Args:
            service_name: Service to stop

        Returns:
            bool: True if stopped successfully
        """
        if service_name not in self.processes:
            logger.warning(f"⚠️  Service {service_name} not found in managed processes")
            return False

        try:
            process = self.processes[service_name]

            logger.info(f"🛑 Stopping service {service_name} (PID: {process.pid})")

            # Try graceful termination first
            process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful fails
                logger.warning(f"⚠️  Force killing {service_name}")
                process.kill()
                process.wait()

            del self.processes[service_name]
            logger.info(f"✅ Service {service_name} stopped")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to stop {service_name}: {e}")
            return False

    async def stop_all(self):
        """Stop all managed external services"""
        services = list(self.processes.keys())
        for service_name in services:
            await self.stop_service(service_name)

    def get_process_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an external service process

        Args:
            service_name: Service name

        Returns:
            Dict with process info or None if not running
        """
        if service_name not in self.processes:
            return None

        process = self.processes[service_name]

        try:
            proc_info = psutil.Process(process.pid)
            return {
                "pid": process.pid,
                "status": proc_info.status(),
                "cpu_percent": proc_info.cpu_percent(),
                "memory_mb": proc_info.memory_info().rss / 1024 / 1024,
                "create_time": proc_info.create_time()
            }
        except psutil.NoSuchProcess:
            # Process died
            del self.processes[service_name]
            return None

    # ============================================================================
    # Phase 2: Auto-Recovery and Port Change Notification
    # ============================================================================

    async def auto_recover_service(
        self,
        service_name: str,
        module_path: str,
        venv_path: Optional[str] = None,
        preferred_port: Optional[int] = None,
        max_attempts: int = 3
    ) -> Optional[int]:
        """
        Auto-recover service with new port when it crashes

        Workflow:
        1. Release previous port from Port Pool
        2. Allocate new port (tries preferred → previous → next available)
        3. Launch service on new port
        4. Update Service Registry with new port
        5. Notify other services of port change

        Args:
            service_name: Name of the service
            module_path: Python module path
            venv_path: Path to virtual environment (optional)
            preferred_port: Preferred port to try (optional)
            max_attempts: Maximum recovery attempts

        Returns:
            New port number if successful, None if failed
        """
        if not self.use_port_pool or not self.port_pool:
            logger.warning(
                f"⚠️  Auto-recovery for {service_name} requires Port Pool Manager. "
                "Falling back to manual recovery."
            )
            return None

        logger.info(f"🔄 Starting auto-recovery for {service_name} (max {max_attempts} attempts)")

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"🔄 Recovery attempt {attempt}/{max_attempts} for {service_name}")

                # 1. Release old port (if any)
                self.port_pool.release_port(service_name)

                # 2. Allocate new port
                new_port = self.port_pool.allocate_port(
                    service_name=service_name,
                    preferred_port=preferred_port,
                    allow_reuse=False  # Force new allocation
                )

                logger.info(f"🎱 Allocated new port {new_port} for {service_name} recovery")

                # 3. Launch service on new port
                success = await self.launch_via_subprocess(
                    service_name=service_name,
                    port=new_port,
                    module_path=module_path,
                    venv_path=venv_path
                )

                if success:
                    logger.info(f"✅ Service {service_name} recovered on port {new_port}")

                    # 4. Update Service Registry (asynchronous, don't block on failure)
                    await self._update_service_registry(service_name, new_port)

                    # 5. Notify other services of port change
                    await self._notify_port_change(service_name, new_port)

                    return new_port
                else:
                    logger.warning(f"⚠️  Recovery attempt {attempt} failed for {service_name}")
                    # Wait before retry (exponential backoff)
                    if attempt < max_attempts:
                        await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"❌ Recovery attempt {attempt} error for {service_name}: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** attempt)

        logger.error(f"❌ Failed to recover {service_name} after {max_attempts} attempts")
        return None

    async def _update_service_registry(self, service_name: str, new_port: int):
        """
        Update Service Registry with new port

        Args:
            service_name: Service name
            new_port: New port number
        """
        try:
            from src.core.service_manager.registry.service_registry import get_registry

            registry = get_registry()

            # Re-register service with new port
            await registry.register(
                name=service_name,
                host="localhost",
                port=new_port,
                metadata={
                    "recovered": True,
                    "recovery_timestamp": time.time()
                }
            )

            logger.info(f"📝 Updated Service Registry: {service_name} → localhost:{new_port}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to update Service Registry for {service_name}: {e}")

    async def _notify_port_change(self, service_name: str, new_port: int):
        """
        Notify other services about port change

        Sends notifications to:
        1. API Gateway (to update routing)
        2. Communication Manager (to invalidate cache)
        3. WebSocket (to update connections) - optional
        4. Other interested services

        Args:
            service_name: Service name
            new_port: New port number
        """
        notification_payload = {
            "service": service_name,
            "new_port": new_port,
            "event": "port_changed",
            "timestamp": time.time()
        }

        # Notify API Gateway (cache update)
        await self._notify_service(
            target="api_gateway",
            endpoint="/internal/routes/update",
            payload=notification_payload
        )

        # Invalidate Communication Manager cache (Phase 4)
        try:
            from src.core.managers.communication_manager import get_communication_manager
            comm = get_communication_manager()
            old_port = comm.invalidate_service_port(service_name, new_port)
            logger.info(
                f"✅ Communication Manager cache invalidated: {service_name} "
                f"({old_port} → {new_port})"
            )
        except Exception as e:
            logger.warning(
                f"⚠️  Failed to invalidate Communication Manager cache for {service_name}: {e}"
            )

        # Notify WebSocket (if needed)
        # await self._notify_service(
        #     target="websocket",
        #     endpoint="/internal/service/update",
        #     payload=notification_payload
        # )

        logger.info(f"📢 Sent port change notifications for {service_name} → {new_port}")

    async def _notify_service(
        self,
        target: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: int = 5
    ):
        """
        Send HTTP notification to a service

        Args:
            target: Target service name
            endpoint: Endpoint path
            payload: Notification payload
            timeout: Request timeout in seconds
        """
        try:
            # Get target service URL from Service Registry
            from src.core.service_manager.registry.service_registry import get_registry

            registry = get_registry()
            service_info = registry.get_service(target)

            if not service_info:
                logger.debug(f"Service {target} not found in registry, skipping notification")
                return

            url = f"{service_info.get_base_url()}{endpoint}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        logger.debug(f"✅ Notified {target}: {endpoint}")
                    else:
                        logger.warning(f"⚠️  Notification to {target} returned {response.status}")

        except asyncio.TimeoutError:
            logger.debug(f"⏱️  Notification to {target} timed out")
        except Exception as e:
            logger.debug(f"⚠️  Failed to notify {target}: {e}")
