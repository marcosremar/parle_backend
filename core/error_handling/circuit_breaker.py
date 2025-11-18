"""
Circuit Breaker Pattern Implementation.

Prevents cascading failures by stopping requests to failing services.
Implements the classic 3-state circuit breaker: CLOSED → OPEN → HALF_OPEN.

States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are blocked immediately
    - HALF_OPEN: Testing if service has recovered

Usage:
    circuit_breaker = CircuitBreaker(
        name="llm_service",
        failure_threshold=5,
        recovery_timeout=60,
        success_threshold=2
    )

    async with circuit_breaker:
        result = await call_external_service()

Reference:
    - Martin Fowler's Circuit Breaker pattern
    - Extracted and improved from src/core/error_handler.py
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, Type
from loguru import logger


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Service is failing
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class CircuitBreakerState:
    """
    State tracking for circuit breaker.

    Attributes:
        state: Current circuit state
        failure_count: Number of consecutive failures
        success_count: Number of consecutive successes in HALF_OPEN
        last_failure_time: Timestamp of last failure
        last_state_change: Timestamp of last state transition
        total_requests: Total requests processed
        total_failures: Total failures across all time
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.utcnow)
    total_requests: int = 0
    total_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for monitoring."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change": self.last_state_change.isoformat(),
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
        }


class CircuitBreakerOpenError(Exception):
    """
    Raised when circuit breaker is OPEN.

    This error indicates the circuit breaker has blocked the request
    to prevent cascading failures. The client should implement fallback
    logic or retry after the recovery timeout.
    """

    def __init__(self, name: str, retry_after: int):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker '{name}' is OPEN. Retry after {retry_after}s")


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    The circuit breaker monitors failures and automatically stops requests
    when a failure threshold is reached. After a recovery timeout, it allows
    a limited number of test requests to check if the service has recovered.

    Attributes:
        name: Identifier for this circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        success_threshold: Successes needed in HALF_OPEN to close circuit
        expected_exception: Exception type to count as failure
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        expected_exception: Type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for monitoring and logging
            failure_threshold: Failures before opening (default: 5)
            recovery_timeout: Seconds before attempting recovery (default: 60)
            success_threshold: Successes in HALF_OPEN to close (default: 2)
            expected_exception: Exception type to monitor (default: Exception)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.expected_exception = expected_exception

        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()

        logger.info(
            f"🔌 Circuit breaker '{name}' initialized "
            f"(threshold: {failure_threshold}, timeout: {recovery_timeout}s)"
        )

    async def __aenter__(self):
        """
        Context manager entry - check circuit state before allowing request.

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN and recovery timeout not reached
        """
        async with self._lock:
            self.state.total_requests += 1

            if self.state.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    # Transition to HALF_OPEN
                    self._transition_to(CircuitState.HALF_OPEN)
                    logger.info(f"🔄 Circuit breaker '{self.name}' → HALF_OPEN (testing recovery)")
                else:
                    # Still OPEN, block request
                    logger.warning(f"🚫 Circuit breaker '{self.name}' is OPEN - request blocked")
                    raise CircuitBreakerOpenError(self.name, self.recovery_timeout)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - update state based on request result.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            False - never suppress exceptions
        """
        async with self._lock:
            if exc_type is None:
                # SUCCESS
                self._handle_success()
            elif issubclass(exc_type, self.expected_exception):
                # FAILURE (expected exception)
                self._handle_failure()

        return False  # Never suppress exceptions

    def _handle_success(self) -> None:
        """Handle successful request based on current state."""
        if self.state.state == CircuitState.HALF_OPEN:
            self.state.success_count += 1
            logger.debug(
                f"✅ Circuit breaker '{self.name}' success in HALF_OPEN "
                f"({self.state.success_count}/{self.success_threshold})"
            )

            if self.state.success_count >= self.success_threshold:
                # Enough successes, close circuit
                self._transition_to(CircuitState.CLOSED)
                self.state.failure_count = 0
                self.state.success_count = 0
                logger.info(f"✅ Circuit breaker '{self.name}' → CLOSED (recovered)")

        elif self.state.state == CircuitState.CLOSED:
            # Reset failure count on success
            if self.state.failure_count > 0:
                logger.debug(f"✅ Circuit breaker '{self.name}' - resetting failure count")
                self.state.failure_count = 0

    def _handle_failure(self) -> None:
        """Handle failed request based on current state."""
        self.state.failure_count += 1
        self.state.total_failures += 1
        self.state.last_failure_time = datetime.utcnow()

        logger.warning(
            f"❌ Circuit breaker '{self.name}' failure "
            f"({self.state.failure_count}/{self.failure_threshold})"
        )

        if self.state.state == CircuitState.HALF_OPEN:
            # Failure in HALF_OPEN → back to OPEN
            self._transition_to(CircuitState.OPEN)
            self.state.success_count = 0
            logger.warning(f"🔴 Circuit breaker '{self.name}' → OPEN (recovery failed)")

        elif self.state.state == CircuitState.CLOSED:
            # Check if threshold reached
            if self.state.failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.error(
                    f"🔴 Circuit breaker '{self.name}' → OPEN "
                    f"(threshold {self.failure_threshold} reached)"
                )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state and update timestamp."""
        old_state = self.state.state
        self.state.state = new_state
        self.state.last_state_change = datetime.utcnow()

        logger.info(
            f"🔄 Circuit breaker '{self.name}': {old_state.value} → {new_state.value}"
        )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed since last failure to attempt recovery."""
        if self.state.last_failure_time is None:
            return True

        time_since_failure = datetime.utcnow() - self.state.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout

    def get_state(self) -> Dict[str, Any]:
        """
        Get current circuit breaker state for monitoring.

        Returns:
            Dictionary with state information
        """
        return {
            "name": self.name,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "success_threshold": self.success_threshold,
            **self.state.to_dict(),
        }

    def reset(self) -> None:
        """
        Manually reset circuit breaker to CLOSED state.

        Use with caution - only for testing or manual intervention.
        """
        logger.warning(f"⚠️  Circuit breaker '{self.name}' manually reset to CLOSED")
        self.state = CircuitBreakerState()

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently OPEN."""
        return self.state.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is currently CLOSED."""
        return self.state.state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is currently HALF_OPEN."""
        return self.state.state == CircuitState.HALF_OPEN


# Global registry for circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.

    This maintains a global registry to ensure the same circuit breaker
    instance is shared across all calls to the same service.

    Args:
        name: Circuit breaker identifier
        **kwargs: Arguments for CircuitBreaker constructor (only used if creating new)

    Returns:
        CircuitBreaker instance

    Example:
        cb = get_circuit_breaker("llm_service", failure_threshold=5)
        async with cb:
            result = await call_llm()
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """
    Get all registered circuit breakers.

    Returns:
        Dictionary mapping names to CircuitBreaker instances

    Example:
        for name, cb in get_all_circuit_breakers().items():
            print(f"{name}: {cb.get_state()}")
    """
    return _circuit_breakers.copy()


def reset_all_circuit_breakers() -> None:
    """
    Reset all circuit breakers to CLOSED state.

    Use with caution - only for testing or emergency situations.
    """
    logger.warning("⚠️  Resetting ALL circuit breakers")
    for cb in _circuit_breakers.values():
        cb.reset()
