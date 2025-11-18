"""
Service Lifecycle Manager - Handles service start/stop/restart operations.

Part of Service Manager refactoring (Phase 3).
Manages service lifecycle: start, stop, restart, reload.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
from loguru import logger

from .service_configuration_manager import ServiceConfig, ProcessStatus, ServiceConfigurationManager
from .process_manager import ProcessManager


class ServiceLifecycleManager:
    """
    Manages service lifecycle - start, stop, restart operations.

    Responsibilities:
    - Start services (internal/external/module modes)
    - Stop services gracefully
    - Restart services
    - Bulk operations (start all, stop all)
    - Track service state transitions

    SOLID Principles:
    - Single Responsibility: Only handles lifecycle
    - Open/Closed: Easy to add new lifecycle strategies
    - Dependency Inversion: Depends on abstractions (ProcessManager, ConfigManager)
    """

    def __init__(
        self,
        config_manager: ServiceConfigurationManager,
        process_manager: ProcessManager
    ):
        """
        Initialize lifecycle manager.

        Args:
            config_manager: Service configuration manager
            process_manager: Process manager for monitoring
        """
        self.config_manager = config_manager
        self.process_manager = process_manager

        # Track lifecycle operations
        self.startup_tasks: Dict[str, asyncio.Task] = {}
        self.shutdown_tasks: Dict[str, asyncio.Task] = {}

        logger.info("🔄 Service Lifecycle Manager initialized")

    async def start_service(
        self,
        service_id: str,
        force: bool = False,
        app: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Start a service.

        Args:
            service_id: Service identifier
            force: Force start even if already running
            app: FastAPI app instance (for internal services)

        Returns:
            Dict with success status and details

        Example:
            lifecycle_mgr = ServiceLifecycleManager(config_mgr, process_mgr)
            result = await lifecycle_mgr.start_service("orchestrator")
            # Returns: {"success": True, "status": "starting", "port": 8500}
        """
        service = self.config_manager.get_service(service_id)
        if not service:
            return {"success": False, "error": f"Service {service_id} not found"}

        # Check if already running
        if not force and service.status == ProcessStatus.RUNNING:
            return {
                "success": True,
                "status": "running",
                "message": f"{service.name} is already running"
            }

        # Check if port is available
        if not self.process_manager.check_port(service.port):
            if not force:
                return {
                    "success": False,
                    "error": f"Port {service.port} is already in use"
                }
            # Force mode - kill process on port
            self.process_manager.kill_process_on_port(service.port)

        try:
            # Update status
            service.status = ProcessStatus.STARTING

            # Start service based on mode
            if self.config_manager.is_internal_service(service_id):
                success = await self._start_internal_service(service_id, app)
            elif self.config_manager.is_module_service(service_id):
                success = await self._start_module_service(service_id, app)
            else:
                success = await self._start_external_service(service_id)

            if success:
                service.status = ProcessStatus.RUNNING
                # Monitor startup
                self.process_manager.monitor_service_startup(
                    service_id,
                    service.port,
                    callback=self._on_startup_complete
                )
                return {
                    "success": True,
                    "status": "starting",
                    "message": f"{service.name} is starting on port {service.port}",
                    "port": service.port
                }
            else:
                service.status = ProcessStatus.FAILED
                return {"success": False, "error": f"Failed to start {service.name}"}

        except Exception as e:
            service.status = ProcessStatus.FAILED
            logger.error(f"❌ Failed to start {service_id}: {e}")
            return {"success": False, "error": str(e)}

    async def stop_service(self, service_id: str) -> Dict[str, Any]:
        """
        Stop a service.

        Args:
            service_id: Service identifier

        Returns:
            Dict with success status and details
        """
        service = self.config_manager.get_service(service_id)
        if not service:
            return {"success": False, "error": f"Service {service_id} not found"}

        # Check if already stopped
        if service.status == ProcessStatus.STOPPED:
            return {
                "success": True,
                "status": "stopped",
                "message": f"{service.name} is already stopped"
            }

        try:
            # Update status
            service.status = ProcessStatus.STOPPING

            # Get PID
            pid = self.process_manager.get_service_pid(Path(service.script).name)

            if pid:
                # Stop service based on mode
                if self.config_manager.is_internal_service(service_id):
                    success = await self._stop_internal_service(service_id)
                else:
                    success = await self._stop_external_service(service_id, pid)

                if success:
                    service.status = ProcessStatus.STOPPED
                    service.pid = None
                    return {
                        "success": True,
                        "status": "stopped",
                        "message": f"{service.name} stopped successfully"
                    }
                else:
                    return {"success": False, "error": f"Failed to stop {service.name}"}
            else:
                # Process not found - assume stopped
                service.status = ProcessStatus.STOPPED
                service.pid = None
                return {
                    "success": True,
                    "status": "stopped",
                    "message": f"{service.name} was not running"
                }

        except Exception as e:
            logger.error(f"❌ Failed to stop {service_id}: {e}")
            return {"success": False, "error": str(e)}

    async def restart_service(self, service_id: str) -> Dict[str, Any]:
        """
        Restart a service.

        Args:
            service_id: Service identifier

        Returns:
            Dict with success status and details
        """
        logger.info(f"🔄 Restarting {service_id}...")

        # Stop first
        stop_result = await self.stop_service(service_id)
        if not stop_result.get("success"):
            return stop_result

        # Wait a bit for cleanup
        await asyncio.sleep(2)

        # Start again
        start_result = await self.start_service(service_id, force=True)
        return start_result

    async def start_all_services(self, force: bool = False) -> Dict[str, Any]:
        """
        Start all services in dependency order.

        Args:
            force: Force start even if already running

        Returns:
            Dict with summary of results
        """
        start_order = self.config_manager.get_start_order()
        results = {}

        for service_id in start_order:
            logger.info(f"🚀 Starting {service_id}...")
            result = await self.start_service(service_id, force=force)
            results[service_id] = result

            # Small delay between starts
            await asyncio.sleep(1)

        success_count = sum(1 for r in results.values() if r.get("success"))
        total_count = len(results)

        return {
            "success": success_count == total_count,
            "started": success_count,
            "total": total_count,
            "results": results
        }

    async def stop_all_services(self) -> Dict[str, Any]:
        """
        Stop all running services.

        Returns:
            Dict with summary of results
        """
        services = self.config_manager.get_all_services()
        results = {}

        # Stop in reverse order (dependencies last)
        start_order = self.config_manager.get_start_order()
        stop_order = list(reversed(start_order))

        for service_id in stop_order:
            if service_id in services:
                logger.info(f"⏹️  Stopping {service_id}...")
                result = await self.stop_service(service_id)
                results[service_id] = result

        success_count = sum(1 for r in results.values() if r.get("success"))
        total_count = len(results)

        return {
            "success": success_count == total_count,
            "stopped": success_count,
            "total": total_count,
            "results": results
        }

    # ============================================================================
    # Private Methods - Service Mode Specific
    # ============================================================================

    async def _start_internal_service(self, service_id: str, app: Optional[Any]) -> bool:
        """
        Start internal service (in-process, no separate process).

        This is a placeholder - actual implementation would:
        1. Load service module
        2. Initialize service instance
        3. Mount routes to FastAPI app
        4. Register in internal_services dict

        Args:
            service_id: Service identifier
            app: FastAPI app instance

        Returns:
            True if started successfully
        """
        logger.info(f"📦 Starting internal service: {service_id}")
        # TODO: Implement internal service loading
        # - Load module dynamically
        # - Create service instance with DI
        # - Mount routes to app
        return True

    async def _start_module_service(self, service_id: str, app: Optional[Any]) -> bool:
        """
        Start module service (in-process module).

        Args:
            service_id: Service identifier
            app: FastAPI app instance

        Returns:
            True if started successfully
        """
        logger.info(f"📦 Starting module service: {service_id}")
        # TODO: Implement module service loading
        return True

    async def _start_external_service(self, service_id: str) -> bool:
        """
        Start external service (separate process).

        This is a placeholder - actual implementation would:
        1. Get service venv path
        2. Build command with uvicorn
        3. Start subprocess
        4. Track PID

        Args:
            service_id: Service identifier

        Returns:
            True if started successfully
        """
        logger.info(f"🚀 Starting external service: {service_id}")
        # TODO: Implement external service launch via ServiceLauncher
        # - Build uvicorn command
        # - Start subprocess
        # - Track PID
        return True

    async def _stop_internal_service(self, service_id: str) -> bool:
        """
        Stop internal service.

        Args:
            service_id: Service identifier

        Returns:
            True if stopped successfully
        """
        logger.info(f"⏹️  Stopping internal service: {service_id}")
        # TODO: Implement internal service cleanup
        # - Remove from internal_services dict
        # - Call cleanup hooks
        return True

    async def _stop_external_service(self, service_id: str, pid: int) -> bool:
        """
        Stop external service.

        Args:
            service_id: Service identifier
            pid: Process ID

        Returns:
            True if stopped successfully
        """
        import signal
        import psutil

        try:
            proc = psutil.Process(pid)
            logger.info(f"⏹️  Stopping external service {service_id} (PID {pid})")

            # Send SIGTERM
            proc.send_signal(signal.SIGTERM)

            # Monitor shutdown
            self.process_manager.monitor_service_stop(
                service_id,
                pid,
                callback=self._on_shutdown_complete
            )

            return True

        except psutil.NoSuchProcess:
            logger.warning(f"⚠️  Process {pid} not found (already stopped?)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to stop external service {service_id}: {e}")
            return False

    def _on_startup_complete(self, service_id: str, success: bool):
        """Callback when service startup completes."""
        if success:
            logger.info(f"✅ {service_id} startup complete")
        else:
            logger.error(f"❌ {service_id} startup failed")

    def _on_shutdown_complete(self, service_id: str, success: bool):
        """Callback when service shutdown completes."""
        if success:
            logger.info(f"✅ {service_id} shutdown complete")
        else:
            logger.warning(f"⚠️  {service_id} shutdown timeout (force killed)")
