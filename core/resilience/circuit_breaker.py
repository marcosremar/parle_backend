#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation
Prevents cascading failures in distributed systems

States:
- CLOSED: Normal operation, requests flow through
- OPEN: Failure threshold exceeded, requests immediately fail
- HALF_OPEN: Testing if service recovered, limited requests allowed

Thread-safe implementation for async operations.
"""

import time
import logging
from typing import Optional, Callable, Any, Dict
from enum import Enum
from dataclasses import dataclass, field
import asyncio

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failure threshold exceeded
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before entering HALF_OPEN state
        success_threshold: Number of successes in HALF_OPEN to close circuit
        timeout: Request timeout in seconds
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    timeout: float = 30.0


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_rejections: int = 0  # Rejected due to OPEN state
    state_history: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_rejections": self.total_rejections,
            "success_rate": self.total_successes / self.total_calls if self.total_calls > 0 else 0,
            "recent_state_changes": self.state_history[-5:]  # Last 5 state changes
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementation for fault tolerance

    Usage:
        circuit = CircuitBreaker(name="external_api")

        # Async function
        result = await circuit.call(async_function, arg1, arg2, kwarg1=value)

        # Check state
        if circuit.is_open():
            print("Circuit is open, service unavailable")
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker

        Args:
            name: Circuit identifier (e.g., "groq_api", "tts_service")
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

        logger.info(
            f"🔌 Circuit Breaker '{name}' initialized\n"
            f"   Failure threshold: {self.config.failure_threshold}\n"
            f"   Recovery timeout: {self.config.recovery_timeout}s\n"
            f"   Success threshold: {self.config.success_threshold}"
        )

    def is_open(self) -> bool:
        """Check if circuit is OPEN"""
        return self.stats.state == CircuitState.OPEN

    def is_closed(self) -> bool:
        """Check if circuit is CLOSED"""
        return self.stats.state == CircuitState.CLOSED

    def is_half_open(self) -> bool:
        """Check if circuit is HALF_OPEN"""
        return self.stats.state == CircuitState.HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker

        Args:
            func: Async function to execute
            *args: Function positional arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is OPEN
            Exception: Original function exceptions (in CLOSED/HALF_OPEN states)
        """
        async with self._lock:
            # Check if we should attempt recovery
            if self.stats.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.total_rejections += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Waiting {self.config.recovery_timeout}s for recovery."
                    )

            # HALF_OPEN: Limited testing
            if self.stats.state == CircuitState.HALF_OPEN:
                # In HALF_OPEN, we allow limited requests to test recovery
                logger.debug(f"🔄 Circuit '{self.name}' HALF_OPEN: Testing recovery...")

        # Execute function
        self.stats.total_calls += 1
        start_time = time.time()

        try:
            # Call function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )

            # Success!
            async with self._lock:
                await self._on_success()

            latency = (time.time() - start_time) * 1000
            logger.debug(f"✅ Circuit '{self.name}' call succeeded ({latency:.0f}ms)")

            return result

        except asyncio.TimeoutError as e:
            async with self._lock:
                await self._on_failure(e, "timeout")
            raise

        except Exception as e:
            async with self._lock:
                await self._on_failure(e, "exception")
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt recovery"""
        if self.stats.last_failure_time is None:
            return True

        elapsed = time.time() - self.stats.last_failure_time
        return elapsed >= self.config.recovery_timeout

    async def _on_success(self):
        """Handle successful call"""
        self.stats.total_successes += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0

        if self.stats.state == CircuitState.HALF_OPEN:
            # Check if we have enough successes to close circuit
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()

        elif self.stats.state == CircuitState.CLOSED:
            # Reset failure counter on success
            self.stats.failure_count = 0

    async def _on_failure(self, exception: Exception, reason: str):
        """Handle failed call"""
        self.stats.total_failures += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.failure_count += 1
        self.stats.last_failure_time = time.time()

        logger.warning(
            f"⚠️  Circuit '{self.name}' failure ({reason}): {exception}\n"
            f"   Consecutive failures: {self.stats.consecutive_failures}/{self.config.failure_threshold}"
        )

        if self.stats.state == CircuitState.HALF_OPEN:
            # Failure during testing - reopen circuit
            self._transition_to_open()

        elif self.stats.state == CircuitState.CLOSED:
            # Check if we exceeded failure threshold
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state"""
        if self.stats.state != CircuitState.OPEN:
            logger.error(
                f"🔴 Circuit '{self.name}' OPENED\n"
                f"   Consecutive failures: {self.stats.consecutive_failures}\n"
                f"   Will retry after {self.config.recovery_timeout}s"
            )
            self._record_state_change(CircuitState.OPEN)
            self.stats.state = CircuitState.OPEN

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        if self.stats.state != CircuitState.HALF_OPEN:
            logger.info(f"🟡 Circuit '{self.name}' HALF_OPEN: Testing recovery...")
            self._record_state_change(CircuitState.HALF_OPEN)
            self.stats.state = CircuitState.HALF_OPEN
            self.stats.consecutive_successes = 0
            self.stats.consecutive_failures = 0

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        if self.stats.state != CircuitState.CLOSED:
            logger.info(
                f"🟢 Circuit '{self.name}' CLOSED: Service recovered!\n"
                f"   Consecutive successes: {self.stats.consecutive_successes}"
            )
            self._record_state_change(CircuitState.CLOSED)
            self.stats.state = CircuitState.CLOSED
            self.stats.failure_count = 0
            self.stats.consecutive_failures = 0

    def _record_state_change(self, new_state: CircuitState):
        """Record state change in history"""
        self.stats.state_history.append({
            "timestamp": time.time(),
            "from_state": self.stats.state.value,
            "to_state": new_state.value,
            "consecutive_failures": self.stats.consecutive_failures,
            "consecutive_successes": self.stats.consecutive_successes
        })

        # Keep only last 50 state changes
        if len(self.stats.state_history) > 50:
            self.stats.state_history = self.stats.state_history[-50:]

    async def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        async with self._lock:
            logger.info(f"🔄 Circuit '{self.name}' manually reset to CLOSED")
            self.stats.state = CircuitState.CLOSED
            self.stats.failure_count = 0
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return self.stats.to_dict()


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers

    Usage:
        registry = CircuitBreakerRegistry()

        # Get or create circuit breaker
        circuit = registry.get("groq_api")

        # Execute through circuit
        result = await circuit.call(api_function, arg1, arg2)

        # Get all stats
        all_stats = registry.get_all_stats()
    """

    def __init__(self):
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()
        logger.info("🔌 Circuit Breaker Registry initialized")

    def get(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get or create circuit breaker

        Args:
            name: Circuit identifier
            config: Optional custom configuration

        Returns:
            CircuitBreaker instance
        """
        if name not in self._circuits:
            self._circuits[name] = CircuitBreaker(
                name=name,
                config=config or self._default_config
            )

        return self._circuits[name]

    def remove(self, name: str):
        """Remove circuit breaker from registry"""
        if name in self._circuits:
            del self._circuits[name]
            logger.info(f"🗑️  Circuit '{name}' removed from registry")

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: circuit.get_stats()
            for name, circuit in self._circuits.items()
        }

    async def reset_all(self):
        """Reset all circuit breakers to CLOSED state"""
        for circuit in self._circuits.values():
            await circuit.reset()
        logger.info("🔄 All circuits reset to CLOSED")

    def set_default_config(self, config: CircuitBreakerConfig):
        """Set default configuration for new circuit breakers"""
        self._default_config = config
        logger.info(f"⚙️  Default circuit breaker config updated")


# Global registry instance
_circuit_breaker_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry (singleton)"""
    global _circuit_breaker_registry

    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry()

    return _circuit_breaker_registry
