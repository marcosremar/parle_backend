"""
Resilience Manager - Handles circuit breaker and retry logic.

Part of Communication Manager refactoring (Phase 2).
Implements resilience patterns for reliable service communication.
"""

from typing import Callable, Any, Optional, Dict
import asyncio
from loguru import logger
from src.core.exceptions import ServiceUnavailableError, ServiceTimeoutError, is_retryable


class ResilienceManager:
    """
    Manages resilience patterns for service communication.

    Implements:
    - Retry with exponential backoff
    - Circuit breaker pattern
    - Timeout handling
    - Fallback logic

    SOLID Principles:
    - Single Responsibility: Only handles resilience
    - Open/Closed: Easy to add new resilience patterns
    """

    def __init__(
        self,
        default_retries: int = 3,
        default_timeout_ms: int = 30000,
        exponential_backoff: bool = True,
    ):
        """
        Initialize resilience manager.

        Args:
            default_retries: Default number of retries
            default_timeout_ms: Default timeout in milliseconds
            exponential_backoff: Whether to use exponential backoff
        """
        self.default_retries = default_retries
        self.default_timeout_ms = default_timeout_ms
        self.exponential_backoff = exponential_backoff

        # Circuit breaker state per service
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}

        logger.info("🛡️  Resilience Manager initialized")

    async def execute_with_resilience(
        self,
        fn: Callable,
        service_name: str,
        retries: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        fallback: Optional[Callable] = None,
    ) -> Any:
        """
        Execute a function with resilience patterns.

        Args:
            fn: Async function to execute
            service_name: Name of service (for circuit breaker)
            retries: Number of retries (overrides default)
            timeout_ms: Timeout in milliseconds (overrides default)
            fallback: Optional fallback function if all retries fail

        Returns:
            Result from fn or fallback

        Raises:
            ServiceUnavailableError: If all retries fail and no fallback
            ServiceTimeoutError: If operation times out

        Example:
            manager = ResilienceManager()

            async def call_llm():
                return await http_client.post("/generate", json=data)

            result = await manager.execute_with_resilience(
                call_llm,
                service_name="llm",
                retries=3,
                timeout_ms=5000
            )
        """
        retries = retries if retries is not None else self.default_retries
        timeout_ms = timeout_ms if timeout_ms is not None else self.default_timeout_ms

        # Check circuit breaker
        if self._is_circuit_open(service_name):
            logger.warning(f"⚠️  Circuit breaker OPEN for {service_name} - failing fast")
            if fallback:
                logger.info(f"Using fallback for {service_name}")
                return await fallback()
            raise ServiceUnavailableError(service_name)

        last_error = None

        for attempt in range(retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(fn(), timeout=timeout_ms / 1000.0)

                # Success - record and return
                self._record_success(service_name)
                if attempt > 0:
                    logger.info(f"✅ {service_name} succeeded on attempt {attempt + 1}/{retries + 1}")
                return result

            except asyncio.TimeoutError as e:
                last_error = ServiceTimeoutError(service_name, timeout_ms, e)
                self._record_failure(service_name)
                logger.warning(f"⏱️  {service_name} timeout (attempt {attempt + 1}/{retries + 1})")

            except Exception as e:
                last_error = e
                self._record_failure(service_name)

                # Check if error is retryable
                if not is_retryable(e):
                    logger.error(f"❌ {service_name} failed with non-retryable error: {e}")
                    break

                logger.warning(f"⚠️  {service_name} failed (attempt {attempt + 1}/{retries + 1}): {e}")

            # Don't sleep after last attempt
            if attempt < retries:
                delay = self._calculate_backoff(attempt)
                logger.debug(f"Waiting {delay:.2f}s before retry...")
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"❌ {service_name} failed after {retries + 1} attempts")

        # Try fallback
        if fallback:
            logger.info(f"Using fallback for {service_name}")
            try:
                return await fallback()
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")

        # No fallback or fallback failed - raise last error
        raise last_error

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff delay.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        if self.exponential_backoff:
            # Exponential: 1s, 2s, 4s, 8s, ...
            return min(2**attempt, 30)  # Cap at 30s
        else:
            # Linear: 1s, 2s, 3s, 4s, ...
            return min(attempt + 1, 10)  # Cap at 10s

    def _is_circuit_open(self, service_name: str) -> bool:
        """
        Check if circuit breaker is open for a service.

        Args:
            service_name: Name of the service

        Returns:
            True if circuit is open (service unavailable), False otherwise
        """
        if service_name not in self._circuit_breakers:
            return False

        breaker = self._circuit_breakers[service_name]

        # Simple implementation: circuit opens after 5 consecutive failures
        return breaker.get("consecutive_failures", 0) >= 5

    def _record_success(self, service_name: str) -> None:
        """Record successful call (closes circuit if open)."""
        if service_name in self._circuit_breakers:
            # Reset on success
            self._circuit_breakers[service_name]["consecutive_failures"] = 0
            logger.debug(f"Circuit breaker CLOSED for {service_name}")

    def _record_failure(self, service_name: str) -> None:
        """Record failed call (may open circuit)."""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = {"consecutive_failures": 0}

        self._circuit_breakers[service_name]["consecutive_failures"] += 1
        failures = self._circuit_breakers[service_name]["consecutive_failures"]

        if failures >= 5:
            logger.warning(f"🔴 Circuit breaker OPENED for {service_name} ({failures} failures)")

    def reset_circuit_breaker(self, service_name: str) -> None:
        """
        Manually reset circuit breaker for a service.

        Args:
            service_name: Name of the service
        """
        if service_name in self._circuit_breakers:
            self._circuit_breakers[service_name]["consecutive_failures"] = 0
            logger.info(f"Circuit breaker reset for {service_name}")

    def get_circuit_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all circuit breakers.

        Returns:
            Dict mapping service names to breaker status
        """
        return {
            service: {
                "consecutive_failures": breaker["consecutive_failures"],
                "is_open": breaker["consecutive_failures"] >= 5,
            }
            for service, breaker in self._circuit_breakers.items()
        }
