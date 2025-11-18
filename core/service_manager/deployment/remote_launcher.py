"""
Remote Service Launcher
Handles launching services on remote RunPod Pods
"""

from pathlib import Path
import logging
import asyncio
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.config.service_execution_config import ServiceExecutionInfo, ExecutionMode
from src.services.runpod_llm.remote_service_manager import RemoteServiceManager
from src.services.runpod_llm.config import load_runpod_config
from src.core.git_manager import get_git_manager

logger = logging.getLogger(__name__)


@dataclass
class RemoteLaunchResult:
    """Result of remote service launch"""
    success: bool
    service_id: str
    service_url: Optional[str] = None
    error_message: Optional[str] = None
    pod_started: bool = False
    command_executed: bool = False


class RemoteServiceLauncher:
    """
    Launches services on remote RunPod Pods

    Features:
    - Ensure Pod is running
    - Execute remote setup commands
    - Start service processes
    - Verify service health
    """

    def __init__(self):
        """Initialize Remote Service Launcher"""
        self.runpod_config = load_runpod_config()
        self.remote_manager: Optional[RemoteServiceManager] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the launcher"""
        if self._initialized:
            logger.warning("RemoteServiceLauncher already initialized")
            return

        logger.info("🚀 Initializing Remote Service Launcher...")

        # Create remote service manager
        self.remote_manager = RemoteServiceManager(self.runpod_config)
        await self.remote_manager.initialize()

        self._initialized = True
        logger.info("✅ Remote Service Launcher initialized")

    async def shutdown(self):
        """Shutdown the launcher"""
        if not self._initialized:
            return

        logger.info("🚀 Shutting down Remote Service Launcher...")

        if self.remote_manager:
            await self.remote_manager.shutdown()

        self._initialized = False
        logger.info("✅ Remote Service Launcher shutdown complete")

    async def _sync_code_on_pod(self, service_id: str) -> bool:
        """
        Synchronize code on Pod via git

        Steps:
        1. Commit local changes (via GitManager)
        2. SSH to Pod
        3. git pull to update code

        Args:
            service_id: Service identifier

        Returns:
            True if sync successful, False otherwise
        """
        logger.info(f"🔄 [{service_id}] Syncing code on Pod...")

        try:
            # Step 1: Commit and push local changes
            git_manager = get_git_manager()

            logger.info("📝 Committing local changes...")
            success, result = git_manager.auto_commit_and_push(
                message="auto-deploy: sync before remote launch"
            )

            if not success:
                logger.error(f"❌ Failed to commit/push local changes: {result}")
                return False

            commit_hash = result
            logger.info(f"✅ Local changes committed: {commit_hash}")

            # Step 2: Pull on Pod
            logger.info("📥 Pulling latest code on Pod...")

            pull_command = (
                "cd ~/ultravox-pipeline && "
                "git fetch --all && "
                "git pull origin main"
            )

            result = await self.remote_manager.execute_remote_command(
                service_id=service_id,
                command=pull_command,
                timeout=60
            )

            if not result or not result.get("success"):
                error_msg = result.get("stderr", "Unknown error") if result else "No result"
                logger.error(f"❌ Git pull failed: {error_msg}")
                return False

            logger.info(f"✅ Code synced successfully on Pod")
            logger.debug(f"Git pull output: {result.get('stdout', '')}")

            return True

        except Exception as e:
            logger.error(f"❌ Error syncing code: {e}", exc_info=True)
            return False

    async def launch_service(
        self,
        service_info: ServiceExecutionInfo
    ) -> RemoteLaunchResult:
        """
        Launch a service on remote Pod

        Args:
            service_info: Service execution information

        Returns:
            RemoteLaunchResult with launch status
        """
        if not self._initialized:
            return RemoteLaunchResult(
                success=False,
                service_id=service_info.service_id,
                error_message="RemoteServiceLauncher not initialized"
            )

        service_id = service_info.service_id

        logger.info(f"🚀 [{service_id}] Launching remote service...")

        try:
            # Register service with remote manager
            self.remote_manager.register_service(
                service_id=service_id,
                remote_host=service_info.remote_host or "localhost",
                remote_port=service_info.remote_port or 8000,
                auto_scale=service_info.auto_scale,
                idle_timeout_seconds=service_info.idle_timeout_seconds
            )

            # Ensure Pod is running
            logger.info(f"🔍 [{service_id}] Ensuring Pod is running...")
            pod_started = await self.remote_manager.ensure_pod_running(service_id)

            if not pod_started:
                return RemoteLaunchResult(
                    success=False,
                    service_id=service_id,
                    error_message="Failed to start Pod",
                    pod_started=False
                )

            # Wait for Pod runtime info (SSH ports to be exposed)
            logger.info(f"⏳ [{service_id}] Waiting for Pod runtime information...")
            runtime_ready = self.remote_manager.pod_controller.wait_for_runtime(timeout=60)

            if not runtime_ready:
                return RemoteLaunchResult(
                    success=False,
                    service_id=service_id,
                    error_message="Pod runtime info not available after 60s",
                    pod_started=True
                )

            logger.info(f"✅ [{service_id}] Pod runtime info ready")

            # Sync code on Pod (commit local + git pull remote)
            logger.info(f"📦 [{service_id}] Syncing code on Pod...")
            sync_success = await self._sync_code_on_pod(service_id)

            if not sync_success:
                logger.warning(f"⚠️  [{service_id}] Code sync failed, but continuing...")

            # Get service URL
            service_url = self.remote_manager.get_service_url(service_id)

            # Execute setup script if configured
            setup_commands = self._get_setup_commands(service_info)
            if setup_commands:
                logger.info(
                    f"🔧 [{service_id}] Executing {len(setup_commands)} setup commands..."
                )

                for i, command in enumerate(setup_commands, 1):
                    logger.info(f"   [{i}/{len(setup_commands)}] {command[:60]}...")

                    result = await self.remote_manager.execute_remote_command(
                        service_id=service_id,
                        command=command,
                        timeout=120
                    )

                    if not result or not result.get("success"):
                        error_msg = result.get("stderr", "Unknown error") if result else "No result"
                        logger.error(
                            f"❌ [{service_id}] Setup command failed: {error_msg}"
                        )
                        return RemoteLaunchResult(
                            success=False,
                            service_id=service_id,
                            error_message=f"Setup command failed: {error_msg}",
                            pod_started=True,
                            command_executed=False
                        )

                logger.info(f"✅ [{service_id}] Setup commands completed")

            # Health check
            logger.info(f"🏥 [{service_id}] Running health check...")
            is_healthy = await self.remote_manager.health_check(service_id)

            if not is_healthy:
                logger.warning(
                    f"⚠️  [{service_id}] Health check failed, but continuing..."
                )

            # Run auto-benchmark if configured
            await self._run_post_start_benchmark(service_id, service_info)

            logger.info(
                f"✅ [{service_id}] Remote service launched successfully "
                f"(URL: {service_url})"
            )

            return RemoteLaunchResult(
                success=True,
                service_id=service_id,
                service_url=service_url,
                pod_started=True,
                command_executed=True
            )

        except Exception as e:
            logger.error(
                f"❌ [{service_id}] Error launching remote service: {e}",
                exc_info=True
            )
            return RemoteLaunchResult(
                success=False,
                service_id=service_id,
                error_message=str(e)
            )

    def _get_setup_commands(self, service_info: ServiceExecutionInfo) -> list[str]:
        """
        Generate setup commands for a service

        Args:
            service_info: Service execution information

        Returns:
            List of setup commands
        """
        commands = []
        service_id = service_info.service_id

        # Base service name (remove _remote suffix)
        base_service_id = service_id.replace('_remote', '')

        # Base setup commands (git pull already done in _sync_code_on_pod)
        commands.extend([
            # Create logs directory
            "cd ~/ultravox-pipeline && mkdir -p logs",

            # Create virtual environment if needed
            "cd ~/ultravox-pipeline && "
            "if [ ! -d venv ]; then python3 -m venv venv; fi",

            # Upgrade pip
            "cd ~/ultravox-pipeline && "
            "source venv/bin/activate && "
            "pip install --upgrade pip -q",

            # Install dependencies
            "cd ~/ultravox-pipeline && "
            "source venv/bin/activate && "
            "pip install -r requirements.txt -q",
        ])

        # Service-specific commands
        if base_service_id == "llm":
            # Get GPU utilization from config
            gpu_util = service_info.gpu_memory_utilization or 0.85
            port = service_info.remote_port or 8100

            commands.extend([
                # Install vLLM if needed
                "cd ~/ultravox-pipeline && "
                "source venv/bin/activate && "
                "pip list | grep -q vllm || pip install vllm -q",

                # Kill existing LLM process
                "pkill -f 'python.*src.services.llm.service' || true",

                # Wait for process to die
                "sleep 2",

                # Start LLM service in background
                f"cd ~/ultravox-pipeline && "
                f"source venv/bin/activate && "
                f"nohup python -m src.services.llm.service "
                f"--service llm "
                f"--port {port} "
                f"> logs/llm.log 2>&1 &",

                # Wait for service to start
                "sleep 5",

                # Check if process is running
                "pgrep -f 'src.services.llm.service' || echo 'WARNING: LLM process not found'"
            ])

        elif service_id == "tts":
            commands.extend([
                # Start TTS service in background
                "cd ~/ultravox-pipeline && "
                "source venv/bin/activate && "
                "pkill -f 'python.*src/services/tts' || true && "
                "nohup python -m src.services.tts.service > /tmp/tts.log 2>&1 &"
            ])

        # Wait for service to start
        commands.append("sleep 5")

        return commands

    async def _run_post_start_benchmark(
        self, service_id: str, service_info: Any
    ) -> None:
        """
        Run auto-benchmark after service starts (if configured)

        Args:
            service_id: Service identifier
            service_info: Service configuration
        """
        try:
            import yaml

            # Load runpod_services.yaml config
            config_path = os.path.join(
                os.path.dirname(__file__), "../../../config/runpod_services.yaml"
            )

            if not os.path.exists(config_path):
                logger.debug(f"No runpod_services.yaml found, skipping benchmark")
                return

            with open(config_path, "r") as f:
                runpod_config = yaml.safe_load(f)

            # Get base service ID (remove _remote suffix if present)
            base_service_id = service_id.replace("_remote", "")

            # Check if benchmark is enabled for this service
            service_config = runpod_config.get("services", {}).get(base_service_id, {})
            benchmark_config = service_config.get("post_start_benchmark", {})

            if not benchmark_config.get("enabled", False):
                logger.debug(f"[{service_id}] Auto-benchmark not enabled, skipping")
                return

            logger.info(f"📊 [{service_id}] Starting auto-benchmark...")

            # Get configuration
            iterations = benchmark_config.get("iterations", 10)
            wait_seconds = benchmark_config.get("wait_for_service_seconds", 15)
            service_port = service_info.remote_port or 8100

            # Wait for service to be fully ready
            logger.info(f"⏳ [{service_id}] Waiting {wait_seconds}s for service to be fully ready...")
            await asyncio.sleep(wait_seconds)

            # Get callback URL (Service Manager)
            callback_url = "http://localhost:8888/benchmark/pod-results"

            # Get Pod ID
            pod_id = getattr(service_info, "runpod_pod_id", "unknown")

            # Upload benchmark script to Pod
            logger.info(f"📤 [{service_id}] Uploading benchmark script to Pod...")

            script_source = str(Path.home() / ".cache" / "ultravox-pipeline" / "benchmark_pod.py"
            script_dest = str(Path.home() / ".cache" / "ultravox-pipeline" / "benchmark_pod.py"

            # Copy script to Pod using runpodctl send
            import subprocess

            send_result = subprocess.run(
                [str(Path.home() / ".cache" / "ultravox-pipeline" / "runpodctl", "send", script_source, f"{pod_id}:{script_dest}"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if send_result.returncode != 0:
                logger.error(f"❌ [{service_id}] Failed to upload benchmark script: {send_result.stderr}")
                return

            logger.info(f"✅ [{service_id}] Benchmark script uploaded")

            # Execute benchmark on Pod in background
            logger.info(f"🚀 [{service_id}] Starting benchmark (this may take a few minutes)...")

            benchmark_command = (
                f"cd ~/ultravox-pipeline && "
                f"source venv/bin/activate && "
                f"BENCHMARK_CALLBACK_URL='{callback_url}' "
                f"BENCHMARK_ITERATIONS={iterations} "
                f"SERVICE_PORT={service_port} "
                f"RUNPOD_POD_ID='{pod_id}' "
                f"nohup python {script_dest} > /tmp/benchmark.log 2>&1 &"
            )

            result = await self.remote_manager.execute_remote_command(
                service_id=service_id,
                command=benchmark_command,
                timeout=10
            )

            if result and result.get("success"):
                logger.info(
                    f"✅ [{service_id}] Benchmark started in background\n"
                    f"   Results will be sent to {callback_url}\n"
                    f"   Check logs: /tmp/benchmark.log on Pod"
                )
            else:
                logger.error(
                    f"❌ [{service_id}] Failed to start benchmark: "
                    f"{result.get('stderr') if result else 'Unknown error'}"
                )

        except Exception as e:
            logger.error(f"❌ [{service_id}] Error running post-start benchmark: {e}")

    def get_remote_manager(self) -> Optional[RemoteServiceManager]:
        """Get the remote service manager instance"""
        return self.remote_manager


# Global singleton instance
_remote_launcher: Optional[RemoteServiceLauncher] = None


async def get_remote_launcher() -> RemoteServiceLauncher:
    """Get global remote launcher instance"""
    global _remote_launcher

    if _remote_launcher is None:
        _remote_launcher = RemoteServiceLauncher()
        await _remote_launcher.initialize()

    return _remote_launcher
