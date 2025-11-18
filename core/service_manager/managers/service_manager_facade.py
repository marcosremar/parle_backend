"""
Service Manager Facade - Simple orchestrating interface for all managers.

Part of Service Manager refactoring (Phase 3).
Provides a unified, simple interface that orchestrates all manager components.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from loguru import logger

from .process_manager import ProcessManager
from .service_configuration_manager import ServiceConfigurationManager, ServiceConfig, ProcessStatus
from .service_lifecycle_manager import ServiceLifecycleManager
from .service_health_monitor import ServiceHealthMonitor


class ServiceManagerFacade:
    """
    Facade that orchestrates all service management components.

    This is the MAIN entry point for service management operations.
    It provides a simple, unified interface while delegating work to
    specialized managers.

    SOLID Principles:
    - Single Responsibility: Orchestrates managers, doesn't implement logic
    - Open/Closed: Easy to add new managers without changing interface
    - Liskov Substitution: Can replace old ServiceManager with this facade
    - Interface Segregation: Clients only see simple interface
    - Dependency Inversion: Depends on manager abstractions

    Architecture:
    ```
    ServiceManagerFacade
        ├── ProcessManager (process monitoring)
        ├── ServiceConfigurationManager (config loading)
        └── ServiceLifecycleManager (start/stop/restart)
    ```

    Example:
        # Simple usage
        facade = ServiceManagerFacade()
        await facade.start_service("orchestrator")
        await facade.stop_service("orchestrator")
        status = facade.get_all_status()
    """

    def __init__(self, config_path: Optional[Path] = None, check_interval: int = 30):
        """
        Initialize Service Manager Facade.

        Args:
            config_path: Path to services_config.yaml (optional)
            check_interval: Health check interval in seconds (default: 30)
        """
        # Initialize managers in dependency order
        self.process_manager = ProcessManager()
        self.config_manager = ServiceConfigurationManager(config_path)
        self.lifecycle_manager = ServiceLifecycleManager(
            self.config_manager,
            self.process_manager
        )
        self.health_monitor = ServiceHealthMonitor(
            self.config_manager,
            check_interval=check_interval
        )

        logger.info("🎯 Service Manager Facade initialized")
        logger.info(f"📋 Managing {len(self.config_manager.get_all_services())} services")
        logger.info(f"🏥 Health monitoring configured (interval: {check_interval}s)")

    # ============================================================================
    # Service Lifecycle Operations (delegate to LifecycleManager)
    # ============================================================================

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
        """
        return await self.lifecycle_manager.start_service(service_id, force, app)

    async def stop_service(self, service_id: str) -> Dict[str, Any]:
        """
        Stop a service.

        Args:
            service_id: Service identifier

        Returns:
            Dict with success status and details
        """
        return await self.lifecycle_manager.stop_service(service_id)

    async def restart_service(self, service_id: str) -> Dict[str, Any]:
        """
        Restart a service.

        Args:
            service_id: Service identifier

        Returns:
            Dict with success status and details
        """
        return await self.lifecycle_manager.restart_service(service_id)

    async def start_all_services(self, force: bool = False) -> Dict[str, Any]:
        """
        Start all services in dependency order.

        Args:
            force: Force start even if already running

        Returns:
            Dict with summary of results
        """
        return await self.lifecycle_manager.start_all_services(force)

    async def stop_all_services(self) -> Dict[str, Any]:
        """
        Stop all running services.

        Returns:
            Dict with summary of results
        """
        return await self.lifecycle_manager.stop_all_services()

    # ============================================================================
    # Service Status and Info (delegate to ConfigManager + ProcessManager)
    # ============================================================================

    def get_all_status(self) -> Dict[str, Any]:
        """
        Get status of all services.

        Returns:
            Dict with service statuses
        """
        services = self.config_manager.get_all_services()
        statuses = {}

        for service_id, service in services.items():
            # Get PID from process manager
            pid = self.process_manager.get_service_pid(Path(service.script).name)

            # Check port availability
            port_available = self.process_manager.check_port(service.port)

            statuses[service_id] = {
                "name": service.name,
                "port": service.port,
                "status": service.status.value,
                "pid": pid,
                "port_available": port_available,
                "is_running": pid is not None
            }

        return {
            "services": statuses,
            "total": len(statuses),
            "running": sum(1 for s in statuses.values() if s["is_running"])
        }

    async def get_all_status_with_health(self) -> Dict[str, Any]:
        """
        Get status of all services with health checks.

        Integrates with HealthMonitor for detailed health info.

        Returns:
            Dict with service statuses and health
        """
        base_status = self.get_all_status()

        # Add health information for each service
        for service_id, status in base_status["services"].items():
            health_result = self.health_monitor.get_health_status(service_id)
            if health_result:
                status["health"] = {
                    "status": health_result.status.value,
                    "response_time_ms": health_result.response_time_ms,
                    "last_check": health_result.timestamp.isoformat(),
                    "error": health_result.error
                }
            else:
                status["health"] = {"status": "unknown"}

        return base_status

    def get_service_info(self, service_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a service.

        Args:
            service_id: Service identifier

        Returns:
            Dict with service info, or None if not found
        """
        service = self.config_manager.get_service(service_id)
        if not service:
            return None

        pid = self.process_manager.get_service_pid(Path(service.script).name)
        port_available = self.process_manager.check_port(service.port)

        return {
            "name": service.name,
            "service_id": service_id,
            "port": service.port,
            "script": service.script,
            "status": service.status.value,
            "pid": pid,
            "port_available": port_available,
            "is_running": pid is not None,
            "is_internal": self.config_manager.is_internal_service(service_id),
            "is_module": self.config_manager.is_module_service(service_id),
            "env": service.env
        }

    # ============================================================================
    # Configuration Operations (delegate to ConfigManager)
    # ============================================================================

    def get_start_order(self) -> List[str]:
        """
        Get recommended service start order based on dependencies.

        Returns:
            List of service IDs in start order
        """
        return self.config_manager.get_start_order()

    def get_service_port(self, service_id: str) -> int:
        """
        Get port for a service.

        Args:
            service_id: Service identifier

        Returns:
            Port number
        """
        return self.config_manager.get_service_port(service_id)

    # ============================================================================
    # Process Operations (delegate to ProcessManager)
    # ============================================================================

    def check_port(self, port: int) -> bool:
        """
        Check if a port is available.

        Args:
            port: Port number to check

        Returns:
            True if port is available, False if in use
        """
        return self.process_manager.check_port(port)

    def kill_process_on_port(self, port: int) -> bool:
        """
        Kill process using a specific port.

        Args:
            port: Port number

        Returns:
            True if process was killed, False if no process found
        """
        return self.process_manager.kill_process_on_port(port)

    def get_service_pid(self, script_name: str) -> Optional[int]:
        """
        Get PID of a service by script name.

        Args:
            script_name: Name of the service script

        Returns:
            PID if found, None otherwise
        """
        return self.process_manager.get_service_pid(script_name)

    # ============================================================================
    # Health Monitoring Operations (delegate to ServiceHealthMonitor)
    # ============================================================================

    async def start_health_monitoring(self) -> None:
        """
        Start background health monitoring for all services.

        Example:
            await facade.start_health_monitoring()
        """
        await self.health_monitor.start(manager=self)

    async def stop_health_monitoring(self) -> None:
        """
        Stop background health monitoring.

        Example:
            await facade.stop_health_monitoring()
        """
        await self.health_monitor.stop()

    async def update_service_health(self, service_id: str):
        """
        Update health status for a service (async).

        Triggers an immediate health check.

        Args:
            service_id: Service identifier

        Example:
            result = await facade.update_service_health("orchestrator")
        """
        return await self.health_monitor.update_service_health(service_id)

    def get_health_history(self, service_id: str, limit: Optional[int] = None) -> List:
        """
        Get health check history for a service.

        Args:
            service_id: Service identifier
            limit: Max number of results (None = all)

        Returns:
            List of HealthCheckResult (most recent first)

        Example:
            history = facade.get_health_history("orchestrator", limit=10)
            for result in history:
                print(f"{result.timestamp}: {result.status.value}")
        """
        return self.health_monitor.get_health_history(service_id, limit)

    def get_health_statistics(self, service_id: str) -> Dict[str, Any]:
        """
        Get health statistics for a service.

        Args:
            service_id: Service identifier

        Returns:
            Dict with statistics (success_rate, avg_response_time, etc.)

        Example:
            stats = facade.get_health_statistics("orchestrator")
            print(f"Success rate: {stats['success_rate']}%")
            print(f"Avg response time: {stats['avg_response_time_ms']}ms")
        """
        return self.health_monitor.get_health_statistics(service_id)

    # ============================================================================
    # Configuration Validation Methods (delegate to ConfigManager)
    # ============================================================================

    def validate_configuration_consistency(self) -> Dict[str, Any]:
        """
        Validate configuration consistency.

        Returns:
            Dict with validation results (issues, warnings, status)
        """
        return self.config_manager.validate_configuration_consistency()

    def validate_port_configuration(self) -> Dict[str, Any]:
        """
        Validate port configuration.

        Returns:
            Dict with validation results
        """
        return self.config_manager.validate_port_configuration()

    def is_internal_service(self, service_id: str) -> bool:
        """Check if service runs locally (MODULE or SERVICE mode)."""
        return self.config_manager.is_internal_service(service_id)

    def is_module_service(self, service_id: str) -> bool:
        """Check if service runs in-process (MODULE mode)."""
        return self.config_manager.is_module_service(service_id)

    def is_service_service(self, service_id: str) -> bool:
        """Check if service runs in separate local process (SERVICE mode)."""
        return self.config_manager.is_service_service(service_id)

    # ============================================================================
    # Process & Health Monitoring Methods (delegate to ProcessManager)
    # ============================================================================

    def check_gpu_memory(self) -> Optional[int]:
        """
        Check available GPU memory.

        Returns:
            Free GPU memory in MB, or None if no GPU available
        """
        return self.process_manager.check_gpu_memory()

    def check_service_health(
        self,
        service_id: str,
        port: int,
        is_module: bool = False,
        timeout: int = 5
    ):
        """
        Check health status of a service.

        Args:
            service_id: Service identifier
            port: Service port
            is_module: True if service runs in-process
            timeout: HTTP request timeout

        Returns:
            Tuple of (status, details)
        """
        return self.process_manager.check_service_health(
            service_id, port, is_module, timeout
        )

    def monitor_service_startup(
        self,
        service_id: str,
        port: int,
        timeout: int = 30,
        callback = None
    ):
        """
        Monitor service startup in background.

        Args:
            service_id: Service identifier
            port: Port to monitor
            timeout: Max time to wait (seconds)
            callback: Optional callback when service is ready
        """
        self.process_manager.monitor_service_startup(
            service_id, port, timeout, callback
        )

    def monitor_service_stop(
        self,
        service_id: str,
        pid: int,
        timeout: int = 10,
        callback = None
    ):
        """
        Monitor service shutdown in background.

        Args:
            service_id: Service identifier
            pid: Process ID to monitor
            timeout: Max time to wait (seconds)
            callback: Optional callback when service stops
        """
        self.process_manager.monitor_service_stop(
            service_id, pid, timeout, callback
        )


    # ============================================================================
    # Utility Methods
    # ============================================================================

    def cleanup(self):
        """Clean up resources (monitor threads, etc.)."""
        self.process_manager.cleanup()
        logger.info("🧹 Service Manager Facade cleaned up")

    def __repr__(self) -> str:
        """String representation."""
        service_count = len(self.config_manager.get_all_services())
        return f"ServiceManagerFacade(services={service_count})"
