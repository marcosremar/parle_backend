#!/usr/bin/env python3
"""
Isolated Service Manager - Manages long-running subprocesses with isolated Python venvs
Each service (STT, TTS, LLM) runs in its own subprocess with its own venv

Features:
- Automatic process lifecycle management
- Health monitoring and auto-restart
- Isolated Python environments (no dependency conflicts)
- Process supervision
"""

import asyncio
import logging
import signal
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service status enum"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    RESTARTING = "restarting"


@dataclass
class ServiceConfig:
    """Configuration for an isolated service"""
    name: str
    script_path: str
    venv_path: str
    port: int
    health_endpoint: str = "/health"
    startup_timeout: int = 10
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    env_vars: Optional[Dict[str, str]] = None  # Environment variables for subprocess


class IsolatedService:
    """Represents a single isolated service subprocess"""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.status = ServiceStatus.STOPPED
        self.restart_count = 0
        self.health_check_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """Start the service subprocess"""
        if self.status == ServiceStatus.RUNNING:
            logger.warning(f"Service {self.config.name} already running")
            return True

        self.status = ServiceStatus.STARTING
        logger.info(f"🚀 Starting {self.config.name} service...")
        logger.info(f"   Venv: {self.config.venv_path}")
        logger.info(f"   Port: {self.config.port}")

        try:
            # Get Python executable from venv
            python_exe = Path(self.config.venv_path) / "bin" / "python"

            if not python_exe.exists():
                # Try alternative path for virtual environments
                python_exe = Path(self.config.venv_path) / "bin" / "python3"

            if not python_exe.exists():
                logger.error(f"❌ Python executable not found in venv: {self.config.venv_path}")
                logger.error(f"   Expected: {python_exe}")
                # Fallback to system python
                python_exe = "python3"
                logger.warning(f"   Using system python instead")

            # Check if script exists
            script_path = Path(self.config.script_path)
            if not script_path.exists():
                logger.error(f"❌ Script not found: {script_path}")
                self.status = ServiceStatus.FAILED
                return False

            # Start subprocess
            self.process = await asyncio.create_subprocess_exec(
                str(python_exe),
                str(script_path),
                "--port", str(self.config.port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(script_path.parent.parent)  # Project root
            )

            logger.info(f"   Process started (PID: {self.process.pid})")

            # Wait for service to be healthy
            if await self._wait_for_health():
                self.status = ServiceStatus.RUNNING
                logger.info(f"✅ {self.config.name} service ready")

                # Start health monitoring
                self.health_check_task = asyncio.create_task(self._monitor_health())

                return True
            else:
                logger.error(f"❌ {self.config.name} failed health check")
                await self.stop()
                self.status = ServiceStatus.FAILED
                return False

        except Exception as e:
            logger.error(f"❌ Failed to start {self.config.name}: {e}")
            self.status = ServiceStatus.FAILED
            return False

    async def stop(self) -> bool:
        """Stop the service subprocess"""
        if self.process is None:
            return True

        logger.info(f"🛑 Stopping {self.config.name} service...")

        # Cancel health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        try:
            # Send SIGTERM
            self.process.terminate()

            # Wait for graceful shutdown (5 seconds)
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.info(f"✅ {self.config.name} stopped gracefully")
            except asyncio.TimeoutError:
                # Force kill
                logger.warning(f"⚠️  {self.config.name} didn't stop, force killing...")
                self.process.kill()
                await self.process.wait()
                logger.info(f"✅ {self.config.name} force killed")

            self.process = None
            self.status = ServiceStatus.STOPPED
            return True

        except Exception as e:
            logger.error(f"❌ Error stopping {self.config.name}: {e}")
            return False

    async def restart(self) -> bool:
        """Restart the service"""
        self.status = ServiceStatus.RESTARTING
        logger.info(f"🔄 Restarting {self.config.name} service...")

        await self.stop()
        await asyncio.sleep(2)  # Wait before restart

        return await self.start()

    async def _wait_for_health(self) -> bool:
        """Wait for service to respond to health check"""
        health_url = f"http://localhost:{self.config.port}{self.config.health_endpoint}"

        for attempt in range(self.config.startup_timeout):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        return True
            except (httpx.RequestError, httpx.TimeoutException, asyncio.TimeoutError):
                # Service not yet responding to health checks
                pass

            await asyncio.sleep(1)

        return False

    async def _monitor_health(self):
        """Continuously monitor service health and restart if needed"""
        consecutive_failures = 0

        while True:
            await asyncio.sleep(10)  # Check every 10 seconds

            try:
                # Check if process is still running
                if self.process.returncode is not None:
                    logger.error(f"❌ {self.config.name} process died (exit code: {self.process.returncode})")
                    consecutive_failures += 1
                else:
                    # Check HTTP health
                    health_url = f"http://localhost:{self.config.port}{self.config.health_endpoint}"
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(health_url)
                        if response.status_code == 200:
                            consecutive_failures = 0
                            continue
                        else:
                            logger.warning(f"⚠️  {self.config.name} health check returned {response.status_code}")
                            consecutive_failures += 1

            except Exception as e:
                logger.warning(f"⚠️  {self.config.name} health check failed: {e}")
                consecutive_failures += 1

            # Auto-restart if configured
            if consecutive_failures >= 3:
                if self.config.restart_on_failure and self.restart_count < self.config.max_restart_attempts:
                    self.restart_count += 1
                    logger.warning(f"⚠️  Auto-restarting {self.config.name} (attempt {self.restart_count}/{self.config.max_restart_attempts})")

                    success = await self.restart()
                    if success:
                        consecutive_failures = 0
                        self.restart_count = 0
                    else:
                        logger.error(f"❌ {self.config.name} restart failed")
                else:
                    logger.error(f"❌ {self.config.name} max restart attempts reached, giving up")
                    self.status = ServiceStatus.FAILED
                    break

    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "name": self.config.name,
            "status": self.status.value,
            "port": self.config.port,
            "pid": self.process.pid if self.process else None,
            "restart_count": self.restart_count
        }


class IsolatedServiceManager:
    """
    Manages multiple isolated services with separate Python venvs
    Each service runs as a subprocess with its own virtual environment
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.services: Dict[str, IsolatedService] = {}
        self._shutdown = False

    def register_service(self, config: ServiceConfig):
        """Register a service to be managed"""
        service = IsolatedService(config)
        self.services[config.name] = service
        logger.info(f"📝 Registered service: {config.name}")

    async def start_all(self) -> bool:
        """Start all registered services"""
        logger.info("=" * 60)
        logger.info("🚀 Starting all isolated services...")
        logger.info("=" * 60)

        tasks = []
        for name, service in self.services.items():
            tasks.append(service.start())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        success_count = sum(1 for r in results if r is True)
        total = len(results)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"📊 Service Startup Summary: {success_count}/{total} succeeded")
        logger.info("=" * 60)

        for name, service in self.services.items():
            status = service.get_status()
            if status["status"] == "running":
                logger.info(f"   ✅ {name} - Port {status['port']} - PID {status['pid']}")
            else:
                logger.error(f"   ❌ {name} - {status['status']}")

        logger.info("=" * 60)

        return success_count == total

    async def stop_all(self):
        """Stop all services"""
        self._shutdown = True

        logger.info("")
        logger.info("=" * 60)
        logger.info("🛑 Stopping all isolated services...")
        logger.info("=" * 60)

        tasks = []
        for name, service in self.services.items():
            tasks.append(service.stop())

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("✅ All isolated services stopped")
        logger.info("=" * 60)

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service"""
        if service_name not in self.services:
            logger.error(f"❌ Service not found: {service_name}")
            return False

        return await self.services[service_name].restart()

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services"""
        return {
            name: service.get_status()
            for name, service in self.services.items()
        }

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services"""
        results = {}

        for name, service in self.services.items():
            if service.status == ServiceStatus.RUNNING:
                try:
                    health_url = f"http://localhost:{service.config.port}{service.config.health_endpoint}"
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(health_url)
                        results[name] = response.status_code == 200
                except (httpx.RequestError, httpx.TimeoutException, asyncio.TimeoutError):
                    # Service health check failed
                    results[name] = False
            else:
                results[name] = False

        return results
