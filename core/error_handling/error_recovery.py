#!/usr/bin/env python3
"""
Error Recovery Mechanisms
Implements auto-recovery strategies and health monitoring
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import (
    UltravoxError,
    ErrorCategory,
    ResourceExhaustedError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions"""

    RESTART = "restart"  # Restart component
    CLEANUP = "cleanup"  # Clean up resources
    RESET = "reset"  # Reset state
    SCALE_UP = "scale_up"  # Scale up resources
    SCALE_DOWN = "scale_down"  # Scale down resources
    FALLBACK = "fallback"  # Switch to fallback
    NOTIFY = "notify"  # Notify administrators
    WAIT = "wait"  # Wait and retry


@dataclass
class RecoveryStrategy:
    """Recovery strategy for specific error types"""

    error_category: ErrorCategory
    actions: List[RecoveryAction]
    max_attempts: int = 3
    cooldown_seconds: int = 60
    auto_recover: bool = True


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""

    timestamp: datetime
    error: UltravoxError
    action: RecoveryAction
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class RecoveryManager:
    """
    Manages error recovery strategies and execution
    """

    def __init__(self, name: str = "recovery_manager"):
        self.name = name
        self.strategies: Dict[ErrorCategory, RecoveryStrategy] = {}
        self.recovery_history: List[RecoveryAttempt] = []
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self._cooldown_until: Dict[str, datetime] = {}

        # Default strategies
        self._setup_default_strategies()

    def _setup_default_strategies(self):
        """Setup default recovery strategies"""

        # Resource exhausted -> cleanup, wait, scale up
        self.add_strategy(
            RecoveryStrategy(
                error_category=ErrorCategory.RESOURCE_EXHAUSTED,
                actions=[RecoveryAction.CLEANUP, RecoveryAction.WAIT, RecoveryAction.SCALE_UP],
                max_attempts=2,
                cooldown_seconds=120,
            )
        )

        # Service unavailable -> wait, fallback, notify
        self.add_strategy(
            RecoveryStrategy(
                error_category=ErrorCategory.SERVICE_UNAVAILABLE,
                actions=[RecoveryAction.WAIT, RecoveryAction.FALLBACK, RecoveryAction.NOTIFY],
                max_attempts=3,
                cooldown_seconds=60,
            )
        )

        # Timeout -> reset, restart
        self.add_strategy(
            RecoveryStrategy(
                error_category=ErrorCategory.TIMEOUT,
                actions=[RecoveryAction.RESET, RecoveryAction.RESTART],
                max_attempts=2,
                cooldown_seconds=30,
            )
        )

        # Network errors -> wait, fallback
        self.add_strategy(
            RecoveryStrategy(
                error_category=ErrorCategory.NETWORK,
                actions=[RecoveryAction.WAIT, RecoveryAction.FALLBACK],
                max_attempts=3,
                cooldown_seconds=45,
            )
        )

    def add_strategy(self, strategy: RecoveryStrategy):
        """Add or update recovery strategy"""
        self.strategies[strategy.error_category] = strategy
        logger.info(
            f"Added recovery strategy for {strategy.error_category.value}: "
            f"{[a.value for a in strategy.actions]}"
        )

    def register_handler(self, action: RecoveryAction, handler: Callable):
        """
        Register handler for recovery action

        Args:
            action: Recovery action type
            handler: Async function to execute action
        """
        self.recovery_handlers[action] = handler
        logger.info(f"Registered handler for recovery action: {action.value}")

    async def attempt_recovery(self, error: UltravoxError, context: Optional[Dict] = None) -> bool:
        """
        Attempt to recover from error

        Args:
            error: Error to recover from
            context: Additional context for recovery

        Returns:
            True if recovery successful, False otherwise
        """
        context = context or {}

        # Get recovery strategy for error category
        strategy = self.strategies.get(error.category)
        if not strategy:
            logger.debug(f"No recovery strategy for category: {error.category.value}")
            return False

        if not strategy.auto_recover:
            logger.info(f"Auto-recovery disabled for category: {error.category.value}")
            return False

        # Check cooldown
        cooldown_key = f"{error.category.value}_{error.context.component}"
        if self._is_in_cooldown(cooldown_key):
            logger.debug(f"Recovery in cooldown for: {cooldown_key}")
            return False

        # Execute recovery actions
        for action in strategy.actions:
            success = await self._execute_recovery_action(action, error, context)

            # Record attempt
            self.recovery_history.append(
                RecoveryAttempt(
                    timestamp=datetime.utcnow(),
                    error=error,
                    action=action,
                    success=success,
                    details=context,
                )
            )

            if success:
                logger.info(
                    f"Recovery successful using action: {action.value} "
                    f"for error: {error.error_code}"
                )
                return True

            logger.warning(f"Recovery action {action.value} failed, trying next action")

        # All actions failed - set cooldown
        self._set_cooldown(cooldown_key, strategy.cooldown_seconds)

        logger.error(f"All recovery actions failed for error: {error.error_code}")
        return False

    async def _execute_recovery_action(
        self, action: RecoveryAction, error: UltravoxError, context: Dict
    ) -> bool:
        """Execute specific recovery action"""

        # Check if handler is registered
        handler = self.recovery_handlers.get(action)
        if not handler:
            logger.warning(f"No handler registered for recovery action: {action.value}")
            return await self._default_action_handler(action, error, context)

        try:
            logger.info(f"Executing recovery action: {action.value}")

            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(error, context)
            else:
                result = handler(error, context)

            return bool(result)

        except Exception as e:
            logger.error(f"Error executing recovery action {action.value}: {e}")
            return False

    async def _default_action_handler(
        self, action: RecoveryAction, error: UltravoxError, context: Dict
    ) -> bool:
        """Default handlers for common actions"""

        if action == RecoveryAction.WAIT:
            # Wait for retry_after or default
            wait_time = error.retry_after or 30
            logger.info(f"Waiting {wait_time}s before retry")
            await asyncio.sleep(wait_time)
            return True

        elif action == RecoveryAction.NOTIFY:
            # Log critical notification
            logger.critical(
                f"RECOVERY NOTIFICATION: {error.error_code} - {error.message}",
                extra={"error_id": error.context.error_id, "component": error.context.component},
            )
            return True

        elif action == RecoveryAction.CLEANUP:
            # Generic cleanup logging
            logger.info("Executing generic cleanup")
            return True

        else:
            logger.warning(f"No default handler for action: {action.value}")
            return False

    def _is_in_cooldown(self, key: str) -> bool:
        """Check if key is in cooldown period"""
        if key not in self._cooldown_until:
            return False

        return datetime.utcnow() < self._cooldown_until[key]

    def _set_cooldown(self, key: str, seconds: int):
        """Set cooldown period for key"""
        self._cooldown_until[key] = datetime.utcnow() + timedelta(seconds=seconds)
        logger.debug(f"Set cooldown for {key}: {seconds}s")

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for attempt in self.recovery_history if attempt.success)

        # Group by action
        actions_stats = {}
        for attempt in self.recovery_history:
            action_name = attempt.action.value
            if action_name not in actions_stats:
                actions_stats[action_name] = {"total": 0, "successful": 0}

            actions_stats[action_name]["total"] += 1
            if attempt.success:
                actions_stats[action_name]["successful"] += 1

        return {
            "name": self.name,
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": (
                successful_attempts / total_attempts if total_attempts > 0 else 0.0
            ),
            "actions": actions_stats,
            "active_cooldowns": len(
                [k for k, v in self._cooldown_until.items() if datetime.utcnow() < v]
            ),
            "registered_strategies": len(self.strategies),
            "registered_handlers": len(self.recovery_handlers),
        }

    def clear_history(self):
        """Clear recovery history"""
        self.recovery_history.clear()
        logger.info(f"Recovery history cleared for {self.name}")


class HealthMonitor:
    """
    Health monitoring with periodic checks and auto-recovery
    """

    def __init__(
        self,
        name: str,
        check_interval: int = 60,
        unhealthy_threshold: int = 3,
        recovery_manager: Optional[RecoveryManager] = None,
    ):
        self.name = name
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.recovery_manager = recovery_manager or RecoveryManager(f"{name}_recovery")

        self._health_checks: Dict[str, Callable] = {}
        self._health_status: Dict[str, bool] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    def register_health_check(self, component: str, check_func: Callable):
        """
        Register health check for component

        Args:
            component: Component name
            check_func: Async function returning bool (True = healthy)
        """
        self._health_checks[component] = check_func
        self._health_status[component] = True
        self._consecutive_failures[component] = 0
        logger.info(f"Registered health check for: {component}")

    async def start_monitoring(self):
        """Start periodic health monitoring"""
        if self._running:
            logger.warning("Health monitoring already running")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started health monitoring for {self.name}")

    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Stopped health monitoring for {self.name}")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _run_health_checks(self):
        """Run all registered health checks"""
        for component, check_func in self._health_checks.items():
            try:
                # Execute health check
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await check_func()
                else:
                    is_healthy = check_func()

                if is_healthy:
                    # Component healthy
                    if not self._health_status[component]:
                        logger.info(f"Component '{component}' recovered")
                    self._health_status[component] = True
                    self._consecutive_failures[component] = 0

                else:
                    # Component unhealthy
                    self._consecutive_failures[component] += 1
                    logger.warning(
                        f"Component '{component}' unhealthy "
                        f"(failures: {self._consecutive_failures[component]}/"
                        f"{self.unhealthy_threshold})"
                    )

                    # Trigger recovery if threshold reached
                    if self._consecutive_failures[component] >= self.unhealthy_threshold:
                        await self._trigger_recovery(component)

            except Exception as e:
                logger.error(f"Error running health check for '{component}': {e}")
                self._consecutive_failures[component] += 1

    async def _trigger_recovery(self, component: str):
        """Trigger recovery for unhealthy component"""
        logger.error(
            f"Component '{component}' exceeded unhealthy threshold, "
            f"triggering recovery"
        )

        # Create error for recovery
        error = ServiceUnavailableError(
            service=component, message=f"Health check failed for {component}"
        )
        error.context.component = component

        # Attempt recovery
        recovered = await self.recovery_manager.attempt_recovery(
            error, context={"health_check": True}
        )

        if recovered:
            self._consecutive_failures[component] = 0
            self._health_status[component] = True
        else:
            self._health_status[component] = False

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        return {
            "name": self.name,
            "monitoring": self._running,
            "check_interval": self.check_interval,
            "components": {
                component: {
                    "healthy": status,
                    "consecutive_failures": self._consecutive_failures[component],
                }
                for component, status in self._health_status.items()
            },
            "overall_health": all(self._health_status.values()),
        }


# Global recovery manager instance
global_recovery_manager = RecoveryManager("global")
