#!/usr/bin/env python3
"""
Health Monitor Module - Background health checking for services

Provides continuous health monitoring with configurable intervals,
automatic health status tracking, detailed health history, and
advanced features like circuit breaker pattern and parallel checks.

Features:
- Asynchronous health checking with configurable intervals
- Circuit breaker to prevent overwhelming failing services
- Parallel health checks for improved performance
- Comprehensive health history with statistics
- Automatic cleanup of old history data
- Detailed metrics and reporting
"""

import asyncio
import aiohttp
import time
from enum import Enum
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from loguru import logger

# Import Prometheus metrics
from . import health_metrics

# Standard health check timeout (seconds) - used across all health check operations
HEALTH_CHECK_TIMEOUT = 5


class HealthStatus(Enum):
    """Service functional health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """
    Result of a health check operation.

    Attributes:
        status: Health status (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)
        timestamp: When the check was performed
        response_time_ms: HTTP response time in milliseconds
        http_status: HTTP status code (if applicable)
        error: Error message (if check failed)
        details: Additional details about the health check
    """
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None
    http_status: Optional[int] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "http_status": self.http_status,
            "error": self.error,
            "details": self.details or {}
        }


class HealthMonitor:
    """
    Background health monitoring for services.

    Features:
    - Periodic health checks (configurable interval)
    - Automatic health status tracking
    - Health check history
    - Timeout handling
    - Error recovery

    Usage:
        monitor = HealthMonitor(check_interval=30)
        await monitor.start(manager)

        # Later...
        await monitor.stop()
    """

    def __init__(
        self,
        check_interval: int = 30,
        timeout: int = 5,
        max_history: int = 100,
        parallel_checks: bool = True,
        circuit_breaker_threshold: int = 5,
        auto_restart_enabled: bool = True,
        auto_restart_threshold: int = 3,
        max_restart_attempts: int = 5,
        restart_cooldown_minutes: int = 60
    ):
        """
        Initialize health monitor with advanced features.

        Args:
            check_interval: Seconds between health checks (default: 30)
            timeout: HTTP request timeout in seconds (default: 5)
            max_history: Maximum number of history entries per service (default: 100)
            parallel_checks: Enable parallel health checking (default: True)
            circuit_breaker_threshold: Failed checks before circuit breaks (default: 5)
            auto_restart_enabled: Enable automatic service restart (default: True)
            auto_restart_threshold: Failed checks before auto-restart (default: 3)
            max_restart_attempts: Maximum restarts in cooldown period (default: 5)
            restart_cooldown_minutes: Cooldown period for restart attempts (default: 60)

        Raises:
            ValueError: If parameters are invalid
        """
        if check_interval < 1:
            raise ValueError("check_interval must be at least 1 second")
        if timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        if circuit_breaker_threshold < 1:
            raise ValueError("circuit_breaker_threshold must be at least 1")
        if auto_restart_threshold < 1:
            raise ValueError("auto_restart_threshold must be at least 1")
        if max_restart_attempts < 1:
            raise ValueError("max_restart_attempts must be at least 1")
        if restart_cooldown_minutes < 1:
            raise ValueError("restart_cooldown_minutes must be at least 1")

        self.check_interval = check_interval
        self.timeout = timeout
        self.max_history = max_history
        self.parallel_checks = parallel_checks
        self.circuit_breaker_threshold = circuit_breaker_threshold

        # Auto-restart configuration
        self.auto_restart_enabled = auto_restart_enabled
        self.auto_restart_threshold = auto_restart_threshold
        self.max_restart_attempts = max_restart_attempts
        self.restart_cooldown_minutes = restart_cooldown_minutes

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._manager = None
        self._session: Optional[aiohttp.ClientSession] = None

        # Health history: service_id -> deque of HealthCheckResult
        self.health_history: Dict[str, deque] = {}

        # Circuit breaker: service_id -> consecutive_failures
        self._circuit_breaker: Dict[str, int] = {}

        # Restart history: service_id -> deque of restart timestamps
        self._restart_history: Dict[str, deque] = {}

        # Restart statistics
        self._total_restarts = 0
        self._failed_restarts = 0

        # Statistics
        self._total_checks = 0
        self._failed_checks = 0

    async def start(self, manager):
        """
        Start background health checking.

        Args:
            manager: ServiceManager instance to monitor
        """
        if self._running:
            logger.warning("⚠️  Health monitor already running")
            return

        self._manager = manager
        self._running = True

        # Create reusable HTTP session for connection pooling
        self._session = aiohttp.ClientSession()

        # Initialize Prometheus metrics
        health_metrics.initialize_metrics(version="1.0.0")

        self._task = asyncio.create_task(self._background_health_checker())

        logger.bind(
            event_type="health_monitor_start",
            check_interval=self.check_interval,
            parallel_checks=self.parallel_checks,
            auto_restart_enabled=self.auto_restart_enabled
        ).info(
            f"✅ Health monitor started (interval: {self.check_interval}s, "
            f"parallel: {self.parallel_checks}, auto_restart: {self.auto_restart_enabled})"
        )

    async def stop(self):
        """Stop background health checking and cleanup resources."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None

        logger.bind(event_type="health_monitor_stop").info("🛑 Health monitor stopped")

    async def _background_health_checker(self):
        """
        Background task to periodically check service health.

        Runs continuously while _running is True, checking all services
        every check_interval seconds. Implements circuit breaker pattern
        and optional parallel checking.
        """
        while True:
            try:
                await asyncio.sleep(self.check_interval)

                # Skip if manager not set
                if not self._manager:
                    continue

                # Update services monitored count
                total_services = len([
                    s for sid, s in self._manager.services.items()
                    if not self._manager._is_internal_service(sid)
                ])
                health_metrics.set_services_monitored(total_services)

                # Collect services to check
                services_to_check = []
                for service_id, service in self._manager.services.items():
                    # Skip internal services
                    if self._manager._is_internal_service(service_id):
                        continue

                    # Circuit breaker: skip if too many consecutive failures
                    if service_id in self._circuit_breaker:
                        if self._circuit_breaker[service_id] >= self.circuit_breaker_threshold:
                            logger.debug(f"⚡ Circuit breaker open for {service_id}, skipping check")
                            continue

                    # Only check running services
                    if not self._manager.check_port(service.port):
                        service.health_status = HealthStatus.UNKNOWN
                        service.health_details = {"reason": "Port not open"}
                        continue

                    services_to_check.append((service_id, service))

                # Update active health checks gauge
                health_metrics.set_active_health_checks(len(services_to_check))

                # Perform health checks (parallel or sequential)
                if self.parallel_checks and len(services_to_check) > 1:
                    # Parallel checking
                    tasks = [
                        self._check_and_update_service(service_id, service)
                        for service_id, service in services_to_check
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Sequential checking
                    for service_id, service in services_to_check:
                        await self._check_and_update_service(service_id, service)

                # Update service uptime percentages
                for service_id in self.health_history.keys():
                    uptime = self.get_service_uptime_percentage(service_id)
                    health_metrics.update_service_uptime(service_id, uptime)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.bind(
                    event_type="background_health_checker_error",
                    error=str(e)
                ).warning(f"⚠️  Health check background task error: {e}")

    async def _check_and_update_service(self, service_id: str, service):
        """
        Check and update a single service with circuit breaker tracking and auto-restart.

        Args:
            service_id: Service identifier
            service: Service object to update
        """
        check_start_time = time.time()

        try:
            # Perform health check
            result = await self.check_service(service_id, service.port)
            self._total_checks += 1

            # Calculate duration
            check_duration = time.time() - check_start_time

            # Record health check metrics
            status_str = 'healthy' if result.status == HealthStatus.HEALTHY else 'unhealthy'
            health_metrics.record_health_check(service_id, status_str, check_duration)

            # Update circuit breaker
            if result.status == HealthStatus.HEALTHY:
                # Reset circuit breaker on success
                self._circuit_breaker[service_id] = 0

                # Structured logging with context
                logger.bind(
                    event_type="health_check",
                    service_id=service_id,
                    status="healthy",
                    response_time_ms=result.response_time_ms
                ).info(f"Health check passed for {service_id}")
            else:
                # Increment failure count
                self._circuit_breaker[service_id] = self._circuit_breaker.get(service_id, 0) + 1
                self._failed_checks += 1
                consecutive_failures = self._circuit_breaker[service_id]

                # Update circuit breaker metrics
                health_metrics.update_circuit_breaker_status(
                    service_id,
                    consecutive_failures,
                    self.circuit_breaker_threshold
                )

                # Structured logging with context
                logger.bind(
                    event_type="health_check_failure",
                    service_id=service_id,
                    status="unhealthy",
                    consecutive_failures=consecutive_failures,
                    error=result.error
                ).warning(
                    f"Health check failed for {service_id} "
                    f"({consecutive_failures} consecutive failures)"
                )

                # Auto-restart logic
                if self.auto_restart_enabled:
                    # Check if we've reached the auto-restart threshold
                    if consecutive_failures >= self.auto_restart_threshold:
                        # Check if we should restart (prevent restart loops)
                        if self._should_restart(service_id):
                            # Record auto-restart trigger
                            health_metrics.record_auto_restart_trigger(
                                service_id,
                                "threshold_reached"
                            )

                            logger.bind(
                                event_type="auto_restart_triggered",
                                service_id=service_id,
                                consecutive_failures=consecutive_failures,
                                threshold=self.auto_restart_threshold
                            ).warning(
                                f"🔄 Service {service_id} has {consecutive_failures} consecutive failures. "
                                f"Attempting auto-restart..."
                            )

                            try:
                                # Attempt to restart the service
                                restart_start_time = time.time()
                                restart_success = await self._restart_service(service_id)
                                restart_duration = time.time() - restart_start_time

                                if restart_success:
                                    # Record successful restart
                                    health_metrics.record_restart(
                                        service_id,
                                        "auto_restart",
                                        "success",
                                        restart_duration
                                    )

                                    logger.bind(
                                        event_type="restart_success",
                                        service_id=service_id,
                                        restart_duration=restart_duration,
                                        trigger="auto_restart"
                                    ).info(f"✅ Successfully restarted {service_id}")

                                    self._total_restarts += 1
                                    self._record_restart(service_id)
                                    # Reset circuit breaker after successful restart
                                    self._circuit_breaker[service_id] = 0
                                else:
                                    # Record failed restart
                                    health_metrics.record_restart(
                                        service_id,
                                        "auto_restart",
                                        "failure",
                                        restart_duration
                                    )

                                    logger.bind(
                                        event_type="restart_failure",
                                        service_id=service_id,
                                        restart_duration=restart_duration,
                                        trigger="auto_restart"
                                    ).error(f"❌ Failed to restart {service_id}")

                                    self._failed_restarts += 1

                            except Exception as restart_error:
                                restart_duration = time.time() - restart_start_time

                                # Record failed restart
                                health_metrics.record_restart(
                                    service_id,
                                    "auto_restart",
                                    "failure",
                                    restart_duration
                                )

                                logger.bind(
                                    event_type="restart_error",
                                    service_id=service_id,
                                    error=str(restart_error),
                                    trigger="auto_restart"
                                ).error(f"❌ Error during restart of {service_id}: {restart_error}")

                                self._failed_restarts += 1
                        else:
                            # Record restart loop prevention
                            health_metrics.record_restart_loop_prevented(service_id)

                            recent_restarts = len(self._restart_history.get(service_id, []))
                            logger.bind(
                                event_type="restart_loop_prevented",
                                service_id=service_id,
                                recent_restarts=recent_restarts,
                                max_attempts=self.max_restart_attempts,
                                cooldown_minutes=self.restart_cooldown_minutes
                            ).warning(
                                f"⚠️  Service {service_id} in restart loop. "
                                f"Skipping auto-restart (too many recent attempts)."
                            )

            # Update service status
            service.health_status = result.status
            service.health_details = result.details
            service.last_check = result.timestamp

            # Store in history using deque
            if service_id not in self.health_history:
                self.health_history[service_id] = deque(maxlen=self.max_history)
            self.health_history[service_id].append(result)

        except Exception as e:
            logger.bind(
                event_type="health_check_error",
                service_id=service_id,
                error=str(e)
            ).error(f"⚠️  Error checking service {service_id}: {e}")
            self._failed_checks += 1

    def _should_restart(self, service_id: str) -> bool:
        """
        Check if service should be restarted (prevent restart loops).

        Args:
            service_id: Service identifier

        Returns:
            True if service should be restarted, False if in restart loop
        """
        # Initialize restart history if not exists
        if service_id not in self._restart_history:
            self._restart_history[service_id] = deque(maxlen=self.max_restart_attempts)

        # Get recent restarts within cooldown window
        now = datetime.now()
        cooldown_start = now - timedelta(minutes=self.restart_cooldown_minutes)

        # Filter out restarts outside cooldown window
        recent_restarts = [
            timestamp for timestamp in self._restart_history[service_id]
            if timestamp > cooldown_start
        ]

        # Update history with only recent restarts
        self._restart_history[service_id].clear()
        self._restart_history[service_id].extend(recent_restarts)

        # Check if we've exceeded max restart attempts in cooldown period
        if len(recent_restarts) >= self.max_restart_attempts:
            logger.warning(
                f"⚠️  Service {service_id} has {len(recent_restarts)} restarts "
                f"in the last {self.restart_cooldown_minutes} minutes. "
                f"Max allowed: {self.max_restart_attempts}"
            )
            return False

        return True

    def _record_restart(self, service_id: str) -> None:
        """
        Record a restart attempt for a service.

        Args:
            service_id: Service identifier
        """
        if service_id not in self._restart_history:
            self._restart_history[service_id] = deque(maxlen=self.max_restart_attempts)

        self._restart_history[service_id].append(datetime.now())

    async def _restart_service(self, service_id: str) -> bool:
        """
        Restart a failed service.

        Args:
            service_id: Service identifier

        Returns:
            True if restart was successful, False otherwise
        """
        if not self._manager:
            logger.error(f"❌ Cannot restart {service_id}: No manager reference")
            return False

        try:
            # Check if manager has restart_service method
            if not hasattr(self._manager, "restart_service"):
                logger.error(f"❌ Cannot restart {service_id}: Manager doesn't support restart")
                return False

            # Call manager's restart_service method
            logger.info(f"🔄 Calling Service Manager to restart {service_id}...")
            result = await self._manager.restart_service(service_id)

            if result:
                logger.info(f"✅ Service {service_id} restarted successfully")
                # Give service time to start up
                await asyncio.sleep(5)
                return True
            else:
                logger.error(f"❌ Service Manager failed to restart {service_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Exception during restart of {service_id}: {e}")
            return False

    async def check_service(self, service_id: str, port: int) -> HealthCheckResult:
        """
        Check health of a single service.

        Args:
            service_id: Service identifier
            port: Service port

        Returns:
            HealthCheckResult with status and details
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Use reusable session if available, otherwise create temporary one
            session = self._session
            if session is None:
                session = aiohttp.ClientSession()
                close_session = True
            else:
                close_session = False

            try:
                async with session.get(
                    f"http://localhost:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_time = (asyncio.get_event_loop().time() - start_time) * 1000

                    if response.status == 200:
                        return HealthCheckResult(
                            status=HealthStatus.HEALTHY,
                            response_time_ms=response_time,
                            http_status=200,
                            details={"status": "responding"}
                        )
                    else:
                        return HealthCheckResult(
                            status=HealthStatus.DEGRADED,
                            response_time_ms=response_time,
                            http_status=response.status,
                            details={"status": f"HTTP {response.status}"}
                        )
            finally:
                if close_session:
                    await session.close()

        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                error="timeout",
                details={"reason": "Health check timeout"}
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                error=str(e),
                details={"reason": str(e)}
            )

    async def check_all_services(self) -> Dict[str, HealthCheckResult]:
        """
        Check health of all services.

        Returns:
            Dictionary mapping service_id to HealthCheckResult
        """
        if not self._manager:
            return {}

        results = {}
        for service_id, service in self._manager.services.items():
            # Skip internal services
            if self._manager._is_internal_service(service_id):
                continue

            # Only check services with open ports
            if self._manager.check_port(service.port):
                results[service_id] = await self.check_service(service_id, service.port)
            else:
                results[service_id] = HealthCheckResult(
                    status=HealthStatus.UNKNOWN,
                    details={"reason": "Port not open"}
                )

        return results

    def get_health_history(self, service_id: str, limit: int = 10) -> list:
        """
        Get health check history for a service.

        Args:
            service_id: Service identifier
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of HealthCheckResult (most recent first)
        """
        if service_id not in self.health_history:
            return []
        # Convert deque to list before slicing
        history_list = list(self.health_history[service_id])
        return list(reversed(history_list[-limit:]))

    def get_service_uptime_percentage(self, service_id: str, lookback_count: int = 100) -> float:
        """
        Calculate service uptime percentage based on recent health checks.

        Args:
            service_id: Service identifier
            lookback_count: Number of recent checks to analyze (default: 100)

        Returns:
            Uptime percentage (0.0 to 100.0)
        """
        if service_id not in self.health_history:
            return 0.0

        recent = list(self.health_history[service_id])[-lookback_count:]
        if not recent:
            return 0.0

        healthy_count = sum(1 for r in recent if r.status == HealthStatus.HEALTHY)
        return (healthy_count / len(recent)) * 100.0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get health monitoring statistics including auto-restart metrics.

        Returns:
            Dictionary with monitoring statistics
        """
        success_rate = 0.0
        if self._total_checks > 0:
            success_rate = ((self._total_checks - self._failed_checks) / self._total_checks) * 100.0

        restart_success_rate = 0.0
        total_restart_attempts = self._total_restarts + self._failed_restarts
        if total_restart_attempts > 0:
            restart_success_rate = (self._total_restarts / total_restart_attempts) * 100.0

        return {
            "total_checks": self._total_checks,
            "failed_checks": self._failed_checks,
            "success_rate_percent": round(success_rate, 2),
            "services_monitored": len(self.health_history),
            "circuit_breakers_open": sum(
                1 for count in self._circuit_breaker.values()
                if count >= self.circuit_breaker_threshold
            ),
            "auto_restart_enabled": self.auto_restart_enabled,
            "total_restarts": self._total_restarts,
            "failed_restarts": self._failed_restarts,
            "restart_success_rate_percent": round(restart_success_rate, 2),
            "services_with_restart_history": len(self._restart_history)
        }

    def reset_circuit_breaker(self, service_id: str) -> None:
        """
        Manually reset circuit breaker for a service.

        Args:
            service_id: Service identifier
        """
        if service_id in self._circuit_breaker:
            self._circuit_breaker[service_id] = 0

            # Record circuit breaker reset
            health_metrics.record_circuit_breaker_reset(service_id)
            health_metrics.update_circuit_breaker_status(service_id, 0, self.circuit_breaker_threshold)

            logger.bind(
                event_type="circuit_breaker_reset",
                service_id=service_id
            ).info(f"✅ Circuit breaker reset for {service_id}")

    def get_circuit_breaker_status(self, service_id: str) -> Dict[str, Any]:
        """
        Get circuit breaker status for a service.

        Args:
            service_id: Service identifier

        Returns:
            Dictionary with circuit breaker information
        """
        consecutive_failures = self._circuit_breaker.get(service_id, 0)
        is_open = consecutive_failures >= self.circuit_breaker_threshold

        return {
            "service_id": service_id,
            "consecutive_failures": consecutive_failures,
            "threshold": self.circuit_breaker_threshold,
            "is_open": is_open,
            "status": "OPEN" if is_open else "CLOSED"
        }


# Legacy background_health_checker() function removed - use HealthMonitor class instead
