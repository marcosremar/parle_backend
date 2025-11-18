"""
Service Health Monitor - Wrapper/Adapter for background health monitoring.

Part of Service Manager refactoring (Phase 3).
Integrates HealthMonitor into the manager architecture.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from ..core.health_monitor import HealthMonitor, HealthStatus, HealthCheckResult
from .service_configuration_manager import ServiceConfigurationManager


class ServiceHealthMonitor:
    """
    Manages service health monitoring - wrapper for HealthMonitor.

    Responsibilities:
    - Coordinate background health checks
    - Provide simple interface for health status
    - Integrate with ServiceConfigurationManager
    - Track health history

    SOLID Principles:
    - Single Responsibility: Only handles health monitoring coordination
    - Open/Closed: Easy to swap health monitoring implementation
    - Dependency Inversion: Depends on HealthMonitor abstraction

    Architecture:
    This is an Adapter/Wrapper that:
    - Wraps the existing HealthMonitor (in core/health_monitor.py)
    - Provides a simpler interface for ServiceManager components
    - Bridges configuration and health monitoring

    Example:
        health_monitor = ServiceHealthMonitor(
            config_manager=config_mgr,
            check_interval=30
        )
        await health_monitor.start()

        # Check health
        status, details = health_monitor.check_service_health("orchestrator")
        # Returns: (HealthStatus.HEALTHY, {"response_time_ms": 15, ...})
    """

    def __init__(
        self,
        config_manager: ServiceConfigurationManager,
        check_interval: int = 30,
        timeout: int = 5,
        parallel_checks: bool = True
    ):
        """
        Initialize service health monitor.

        Args:
            config_manager: Service configuration manager
            check_interval: Interval between health checks (seconds)
            timeout: Timeout for each health check (seconds)
            parallel_checks: Whether to run checks in parallel
        """
        self.config_manager = config_manager

        # Create underlying HealthMonitor
        self.health_monitor = HealthMonitor(
            check_interval=check_interval,
            timeout=timeout,
            parallel_checks=parallel_checks
        )

        self._is_monitoring = False
        logger.info("🏥 Service Health Monitor initialized")

    # ============================================================================
    # Monitoring Lifecycle
    # ============================================================================

    async def start(self, manager: Optional[Any] = None) -> None:
        """
        Start background health monitoring.

        Args:
            manager: ServiceManager instance (for compatibility)
        """
        if self._is_monitoring:
            logger.warning("⚠️  Health monitoring already started")
            return

        logger.info("🏥 Starting background health monitoring...")
        await self.health_monitor.start(manager)
        self._is_monitoring = True
        logger.info("✅ Background health monitoring started")

    async def stop(self) -> None:
        """Stop background health monitoring."""
        if not self._is_monitoring:
            logger.warning("⚠️  Health monitoring not running")
            return

        logger.info("🏥 Stopping background health monitoring...")
        await self.health_monitor.stop()
        self._is_monitoring = False
        logger.info("✅ Background health monitoring stopped")

    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._is_monitoring

    # ============================================================================
    # Health Checking
    # ============================================================================

    def check_service_health(self, service_id: str) -> tuple[HealthStatus, dict]:
        """
        Check health of a service (synchronous).

        This is the main method used by other components to check service health.

        Args:
            service_id: Service identifier

        Returns:
            Tuple of (HealthStatus, details_dict)

        Example:
            status, details = health_monitor.check_service_health("orchestrator")
            if status == HealthStatus.HEALTHY:
                print(f"Service is healthy: {details['response_time_ms']}ms")
            else:
                print(f"Service unhealthy: {details.get('error')}")
        """
        service = self.config_manager.get_service(service_id)
        if not service:
            return HealthStatus.UNKNOWN, {"error": f"Service {service_id} not found"}

        # Get health status from HealthMonitor
        health_result = self.health_monitor.get_health_status(service_id)

        if health_result:
            return health_result.status, self._result_to_dict(health_result)
        else:
            # No health data yet - perform on-demand check
            return self._perform_immediate_check(service_id, service.port)

    async def update_service_health(self, service_id: str) -> HealthCheckResult:
        """
        Update health status for a service (async).

        Triggers an immediate health check and updates cached status.

        Args:
            service_id: Service identifier

        Returns:
            HealthCheckResult with status and details
        """
        service = self.config_manager.get_service(service_id)
        if not service:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                error=f"Service {service_id} not found"
            )

        # Trigger async check via HealthMonitor
        return await self.health_monitor.check_service_health_async(
            service_id,
            service.port
        )

    # ============================================================================
    # Health Status Retrieval
    # ============================================================================

    def get_health_status(self, service_id: str) -> Optional[HealthCheckResult]:
        """
        Get cached health status for a service.

        Args:
            service_id: Service identifier

        Returns:
            HealthCheckResult if available, None if no data
        """
        return self.health_monitor.get_health_status(service_id)

    def get_health_history(
        self,
        service_id: str,
        limit: Optional[int] = None
    ) -> List[HealthCheckResult]:
        """
        Get health check history for a service.

        Args:
            service_id: Service identifier
            limit: Max number of results (None = all)

        Returns:
            List of HealthCheckResult (most recent first)
        """
        return self.health_monitor.get_health_history(service_id, limit)

    def get_all_health_status(self) -> Dict[str, HealthCheckResult]:
        """
        Get health status for all services.

        Returns:
            Dict mapping service_id to HealthCheckResult
        """
        services = self.config_manager.get_all_services()
        statuses = {}

        for service_id in services.keys():
            health_result = self.health_monitor.get_health_status(service_id)
            if health_result:
                statuses[service_id] = health_result

        return statuses

    # ============================================================================
    # Statistics and Reporting
    # ============================================================================

    def get_health_statistics(self, service_id: str) -> Dict[str, Any]:
        """
        Get health statistics for a service.

        Args:
            service_id: Service identifier

        Returns:
            Dict with statistics (success_rate, avg_response_time, etc.)
        """
        return self.health_monitor.get_health_statistics(service_id)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get overall monitoring status.

        Returns:
            Dict with monitoring info
        """
        return {
            "is_monitoring": self._is_monitoring,
            "check_interval": self.health_monitor.check_interval,
            "timeout": self.health_monitor.timeout,
            "services_monitored": len(self.health_monitor.health_status),
            "parallel_checks": self.health_monitor.parallel_checks
        }

    # ============================================================================
    # Private Helpers
    # ============================================================================

    def _perform_immediate_check(self, service_id: str, port: int) -> tuple[HealthStatus, dict]:
        """
        Perform immediate health check (synchronous fallback).

        Args:
            service_id: Service identifier
            port: Service port

        Returns:
            Tuple of (HealthStatus, details_dict)
        """
        import requests
        from datetime import datetime

        try:
            url = f"http://localhost:{port}/health"
            start_time = datetime.now()
            response = requests.get(url, timeout=self.health_monitor.timeout)
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if response.status_code == 200:
                return HealthStatus.HEALTHY, {
                    "response_time_ms": response_time_ms,
                    "http_status": response.status_code,
                    "timestamp": start_time.isoformat()
                }
            else:
                return HealthStatus.DEGRADED, {
                    "response_time_ms": response_time_ms,
                    "http_status": response.status_code,
                    "error": f"HTTP {response.status_code}"
                }

        except requests.exceptions.Timeout:
            return HealthStatus.UNHEALTHY, {"error": "Timeout"}
        except requests.exceptions.ConnectionError:
            return HealthStatus.UNHEALTHY, {"error": "Connection refused"}
        except Exception as e:
            return HealthStatus.UNHEALTHY, {"error": str(e)}

    def _result_to_dict(self, result: HealthCheckResult) -> dict:
        """Convert HealthCheckResult to dict."""
        return {
            "status": result.status.value,
            "timestamp": result.timestamp.isoformat(),
            "response_time_ms": result.response_time_ms,
            "http_status": result.http_status,
            "error": result.error,
            "details": result.details or {}
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ServiceHealthMonitor(monitoring={self._is_monitoring}, "
            f"services={len(self.config_manager.get_all_services())})"
        )
