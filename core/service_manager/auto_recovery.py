#!/usr/bin/env python3
"""
Auto-Recovery System for Service Manager
Monitors service health and automatically restarts crashed/hung services

Features:
- Continuous health monitoring
- Automatic restart on failure
- Circuit breaker to prevent restart loops
- Crash notifications
- Recovery metrics
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Service health states"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    CRASHED = "crashed"
    RECOVERING = "recovering"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Too many failures, stop recovery
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RecoveryStats:
    """Statistics for service recovery"""
    service_id: str
    total_crashes: int = 0
    total_recoveries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    last_crash_time: Optional[datetime] = None
    last_recovery_time: Optional[datetime] = None
    crash_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def record_crash(self, exception_info: Dict[str, str] = None):
        """Record a service crash"""
        self.total_crashes += 1
        self.last_crash_time = datetime.now()

        crash_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "crash"
        }

        # Add exception info if available
        if exception_info:
            crash_entry["exception"] = {
                "type": exception_info.get("type"),
                "message": exception_info.get("message"),
                "traceback": exception_info.get("traceback")
            }

        self.crash_history.append(crash_entry)

    def record_recovery_attempt(self, success: bool):
        """Record a recovery attempt"""
        self.total_recoveries += 1
        if success:
            self.successful_recoveries += 1
        else:
            self.failed_recoveries += 1
        self.last_recovery_time = datetime.now()
        self.crash_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "recovery",
            "success": success
        })

    def get_recovery_rate(self) -> float:
        """Get recovery success rate"""
        if self.total_recoveries == 0:
            return 0.0
        return (self.successful_recoveries / self.total_recoveries) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "service_id": self.service_id,
            "total_crashes": self.total_crashes,
            "total_recoveries": self.total_recoveries,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "recovery_rate": self.get_recovery_rate(),
            "last_crash_time": self.last_crash_time.isoformat() if self.last_crash_time else None,
            "last_recovery_time": self.last_recovery_time.isoformat() if self.last_recovery_time else None,
            "crash_history": list(self.crash_history)
        }


@dataclass
class CircuitBreaker:
    """Circuit breaker to prevent restart loops"""
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: int = 300  # Seconds to wait before trying again
    half_open_timeout: int = 60  # Seconds to test in half-open state

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None

    def record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.open_circuit()

    def record_success(self):
        """Record a success"""
        if self.state == CircuitState.HALF_OPEN:
            self.close_circuit()
        self.failure_count = 0

    def open_circuit(self):
        """Open the circuit (stop recovery)"""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.opened_at = datetime.now()
            logger.warning(f"Circuit breaker OPEN: Too many failures ({self.failure_count})")

    def close_circuit(self):
        """Close the circuit (resume normal operation)"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at = None
        logger.info("Circuit breaker CLOSED: Service recovered")

    def can_attempt_recovery(self) -> bool:
        """Check if recovery attempt is allowed"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if enough time has passed to try again
            if self.opened_at:
                elapsed = (datetime.now() - self.opened_at).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker HALF_OPEN: Testing recovery")
                    return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "recovery_timeout": self.recovery_timeout,
            "can_attempt_recovery": self.can_attempt_recovery()
        }


class AutoRecoveryManager:
    """
    Manages automatic recovery of crashed services

    Features:
    - Periodic health checks
    - Automatic restart on failure
    - Circuit breaker to prevent loops
    - Recovery statistics
    - Crash notifications
    """

    def __init__(
        self,
        service_manager,
        check_interval: int = 30,
        restart_delay: int = 5,
        enable_auto_recovery: bool = True,
        stateful_services: List[str] = None
    ):
        """
        Initialize Auto-Recovery Manager

        Args:
            service_manager: ServiceManager instance
            check_interval: Seconds between health checks (default: 30s)
            restart_delay: Seconds to wait before restarting (default: 5s)
            enable_auto_recovery: Enable automatic recovery (default: True)
            stateful_services: List of stateful services (won't be auto-restarted)
        """
        self.service_manager = service_manager
        self.check_interval = check_interval
        self.restart_delay = restart_delay
        self.enable_auto_recovery = enable_auto_recovery
        self.stateful_services = stateful_services or [
            "communication", "database", "session", "conversation_store"
        ]

        # Monitoring state
        self.service_health: Dict[str, ServiceHealth] = {}
        self.recovery_stats: Dict[str, RecoveryStats] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Background task
        self.monitor_task: Optional[asyncio.Task] = None
        self.running = False

        # Callbacks
        self.crash_callbacks: List[Callable] = []

        logger.info("🛡️ AutoRecoveryManager initialized")
        logger.info(f"   Check interval: {check_interval}s")
        logger.info(f"   Auto-recovery: {'enabled' if enable_auto_recovery else 'disabled'}")
        logger.info(f"   Stateful services (protected): {', '.join(self.stateful_services)}")

    def start(self):
        """Start health monitoring"""
        if self.running:
            logger.warning("Auto-recovery already running")
            return

        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("✅ Auto-recovery started")

    def stop(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
        logger.info("🛑 Auto-recovery stopped")

    def register_crash_callback(self, callback: Callable):
        """Register callback for crash notifications"""
        self.crash_callbacks.append(callback)

    async def _monitor_loop(self):
        """Main monitoring loop"""
        logger.info("🔍 Health monitoring loop started")

        while self.running:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_all_services()
            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                import traceback
                traceback.print_exc()

    async def _check_all_services(self):
        """Check health of all services"""
        logger.debug("🔍 Checking health of all services...")

        # Get list of services to monitor
        services_to_check = []

        # Internal services (in-process modules)
        if hasattr(self.service_manager, 'internal_services'):
            services_to_check.extend(self.service_manager.internal_services.keys())

        # External services (separate processes with ports)
        if hasattr(self.service_manager, 'service_configs'):
            for service_id, config in self.service_manager.service_configs.items():
                if config.get('type') == 'service' and 'port' in config:
                    if service_id not in services_to_check:
                        services_to_check.append(service_id)

        for service_id in services_to_check:
            await self._check_service_health(service_id)

    async def _check_service_health(self, service_id: str):
        """Check health of a specific service"""
        try:
            health_result = None

            # Try internal service first (in-process)
            service = None
            if hasattr(self.service_manager, 'internal_services'):
                service = self.service_manager.internal_services.get(service_id)

            if service and hasattr(service, 'health_check'):
                # Direct call for in-process services
                health_result = await service.health_check()
            else:
                # Try HTTP for external services
                if hasattr(self.service_manager, 'service_configs'):
                    config = self.service_manager.service_configs.get(service_id)
                    if config and config.get('type') == 'service' and 'port' in config:
                        import httpx
                        port = config['port']
                        url = f"http://localhost:{port}/health"

                        async with httpx.AsyncClient(timeout=5.0) as client:
                            response = await client.get(url)
                            if response.status_code == 200:
                                health_result = response.json()

            if not health_result:
                self.service_health[service_id] = ServiceHealth.UNKNOWN
                return

            # Determine health status
            if isinstance(health_result, dict):
                status = health_result.get('status', 'unknown')

                if status in ['healthy', 'running']:
                    new_health = ServiceHealth.HEALTHY
                elif status in ['unhealthy', 'degraded']:
                    new_health = ServiceHealth.UNHEALTHY
                else:
                    new_health = ServiceHealth.UNKNOWN
            else:
                new_health = ServiceHealth.UNKNOWN

            # Check for state transitions
            old_health = self.service_health.get(service_id)
            self.service_health[service_id] = new_health

            # Detect crash (healthy -> unhealthy)
            if old_health == ServiceHealth.HEALTHY and new_health == ServiceHealth.UNHEALTHY:
                await self._handle_service_crash(service_id)

            # Detect recovery (unhealthy -> healthy)
            elif old_health == ServiceHealth.UNHEALTHY and new_health == ServiceHealth.HEALTHY:
                await self._handle_service_recovery(service_id)

        except Exception as e:
            # Capture full traceback for crash analysis
            import traceback
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))

            logger.error(f"💥 Health check exception for {service_id}:")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            logger.error(f"   Traceback:\n{tb_str}")

            # Mark as crashed if previously healthy
            old_health = self.service_health.get(service_id)
            self.service_health[service_id] = ServiceHealth.CRASHED

            if old_health in [ServiceHealth.HEALTHY, ServiceHealth.UNKNOWN]:
                await self._handle_service_crash(service_id, exception_info={
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": tb_str
                })

    async def _handle_service_crash(self, service_id: str, exception_info: Dict[str, str] = None):
        """Handle service crash"""
        logger.warning(f"💥 Service crashed: {service_id}")

        # Log exception details if available
        if exception_info:
            logger.warning(f"   Crash details:")
            logger.warning(f"   • Type: {exception_info.get('type', 'Unknown')}")
            logger.warning(f"   • Message: {exception_info.get('message', 'No message')}")

        # Initialize stats if needed
        if service_id not in self.recovery_stats:
            self.recovery_stats[service_id] = RecoveryStats(service_id)

        # Record crash with exception info
        self.recovery_stats[service_id].record_crash(exception_info)

        # Notify crash callbacks
        for callback in self.crash_callbacks:
            try:
                await callback(service_id, "crash")
            except Exception as e:
                logger.error(f"Error in crash callback: {e}")

        # Attempt recovery if enabled
        if self.enable_auto_recovery:
            await self._attempt_recovery(service_id)

    async def _handle_service_recovery(self, service_id: str):
        """Handle service recovery"""
        logger.info(f"✅ Service recovered: {service_id}")

        # Update circuit breaker
        if service_id in self.circuit_breakers:
            self.circuit_breakers[service_id].record_success()

    async def _attempt_recovery(self, service_id: str):
        """Attempt to recover a crashed service"""
        # Don't auto-restart stateful services
        if service_id in self.stateful_services:
            logger.warning(f"⚠️  {service_id} is stateful - manual intervention required")
            return

        # Initialize circuit breaker if needed
        if service_id not in self.circuit_breakers:
            self.circuit_breakers[service_id] = CircuitBreaker()

        # Check circuit breaker
        circuit = self.circuit_breakers[service_id]
        if not circuit.can_attempt_recovery():
            logger.warning(f"🚫 Circuit breaker OPEN for {service_id} - skipping recovery")
            return

        logger.info(f"🔄 Attempting to recover {service_id}...")
        self.service_health[service_id] = ServiceHealth.RECOVERING

        # Wait before restart
        await asyncio.sleep(self.restart_delay)

        try:
            # Restart service
            success = await self._restart_service(service_id)

            if success:
                logger.info(f"✅ Successfully recovered {service_id}")
                self.recovery_stats[service_id].record_recovery_attempt(True)
                circuit.record_success()
            else:
                logger.error(f"❌ Failed to recover {service_id}")
                self.recovery_stats[service_id].record_recovery_attempt(False)
                circuit.record_failure()

        except Exception as e:
            logger.error(f"❌ Recovery attempt failed for {service_id}: {e}")
            self.recovery_stats[service_id].record_recovery_attempt(False)
            circuit.record_failure()

    async def _restart_service(self, service_id: str) -> bool:
        """Restart a service"""
        try:
            logger.info(f"🔄 Restarting {service_id}...")

            # Get service instance
            service = None
            if hasattr(self.service_manager, 'internal_services'):
                service = self.service_manager.internal_services.get(service_id)

            if not service:
                logger.error(f"Service {service_id} not found")
                return False

            # Stop service
            if hasattr(service, 'shutdown'):
                await service.shutdown()
                logger.info(f"   Stopped {service_id}")

            # Wait a bit
            await asyncio.sleep(2)

            # Start service
            if hasattr(service, 'initialize'):
                success = await service.initialize()
                if success:
                    logger.info(f"   Started {service_id}")
                    return True
                else:
                    logger.error(f"   Failed to start {service_id}")
                    return False

            return False

        except Exception as e:
            logger.error(f"Error restarting {service_id}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get auto-recovery status"""
        return {
            "enabled": self.enable_auto_recovery,
            "running": self.running,
            "check_interval": self.check_interval,
            "restart_delay": self.restart_delay,
            "stateful_services": self.stateful_services,
            "monitored_services": len(self.service_health),
            "service_health": {
                service_id: health.value
                for service_id, health in self.service_health.items()
            },
            "recovery_stats": {
                service_id: stats.to_dict()
                for service_id, stats in self.recovery_stats.items()
            },
            "circuit_breakers": {
                service_id: breaker.to_dict()
                for service_id, breaker in self.circuit_breakers.items()
            }
        }

    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific service"""
        if service_id not in self.service_health:
            return None

        return {
            "service_id": service_id,
            "health": self.service_health[service_id].value,
            "is_stateful": service_id in self.stateful_services,
            "stats": self.recovery_stats.get(service_id).to_dict() if service_id in self.recovery_stats else None,
            "circuit_breaker": self.circuit_breakers.get(service_id).to_dict() if service_id in self.circuit_breakers else None
        }


# Singleton instance
_auto_recovery_manager: Optional[AutoRecoveryManager] = None


def get_auto_recovery_manager() -> Optional[AutoRecoveryManager]:
    """Get singleton AutoRecoveryManager instance"""
    global _auto_recovery_manager
    return _auto_recovery_manager


def initialize_auto_recovery(
    service_manager,
    check_interval: int = 30,
    enable_auto_recovery: bool = True
) -> AutoRecoveryManager:
    """Initialize and return AutoRecoveryManager"""
    global _auto_recovery_manager

    if _auto_recovery_manager is None:
        _auto_recovery_manager = AutoRecoveryManager(
            service_manager=service_manager,
            check_interval=check_interval,
            enable_auto_recovery=enable_auto_recovery
        )

    return _auto_recovery_manager
