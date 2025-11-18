#!/usr/bin/env python3
"""
Circuit Breaker for LLM Failover
Automatically switches from primary (local) to fallback (external) LLM on failures
"""

import time
import logging
import asyncio
from enum import Enum
from typing import Callable, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation - using primary
    OPEN = "open"            # Circuit open - using fallback
    HALF_OPEN = "half_open"  # Testing recovery - trying primary


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 3       # Failures before opening circuit
    recovery_timeout: int = 30       # Seconds before retry
    half_open_max_calls: int = 1     # Test calls in half-open state
    primary_timeout: int = 10        # Timeout for primary calls
    fallback_timeout: int = 15       # Timeout for fallback calls


class CircuitBreaker:
    """
    Circuit breaker pattern for automatic LLM failover

    States:
    - CLOSED: Normal operation, using primary LLM
    - OPEN: Too many failures, using fallback LLM
    - HALF_OPEN: Testing if primary has recovered

    Flow:
    CLOSED --(3 failures)--> OPEN --(30s timeout)--> HALF_OPEN --(success)--> CLOSED
                                                            |
                                                      (failure)
                                                            |
                                                            v
                                                          OPEN
    """

    def __init__(self, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker

        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.success_count_in_half_open = 0

        logger.info(f"🔌 Circuit breaker initialized: threshold={config.failure_threshold}, "
                   f"recovery_timeout={config.recovery_timeout}s")

    async def call_with_fallback(
        self,
        primary_fn: Callable,
        fallback_fn: Callable,
        context: dict
    ) -> Tuple[Any, str]:
        """
        Call primary function with automatic fallback on failure

        Args:
            primary_fn: Primary function to call (local LLM)
            fallback_fn: Fallback function (external LLM)
            context: Context to pass to functions

        Returns:
            Tuple of (result, llm_used: "primary"|"fallback")
        """
        # Check if we should attempt to reset the circuit
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count_in_half_open = 0
                logger.info("🔄 Circuit breaker HALF_OPEN - testing primary LLM recovery")
            else:
                # Circuit still open, use fallback
                logger.debug(f"⚠️  Circuit OPEN - using fallback LLM "
                           f"(retry in {self._time_until_retry():.1f}s)")
                result = await self._call_fallback(fallback_fn, context)
                return result, "fallback"

        # Try primary LLM
        try:
            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                logger.debug(f"🎯 Attempting primary LLM (state={self.state.value})")

                result = await asyncio.wait_for(
                    primary_fn(context),
                    timeout=self.config.primary_timeout
                )

                # Success!
                self._on_success()
                return result, "primary"

        except asyncio.TimeoutError:
            logger.error(f"⏱️  Primary LLM timeout after {self.config.primary_timeout}s")
            self._on_failure()

        except Exception as e:
            logger.error(f"❌ Primary LLM failed: {type(e).__name__}: {e}")
            self._on_failure()

        # Primary failed, use fallback
        logger.info("🔄 Switching to fallback LLM")
        result = await self._call_fallback(fallback_fn, context)
        return result, "fallback"

    async def _call_fallback(self, fallback_fn: Callable, context: dict) -> Any:
        """
        Call fallback function with timeout

        Args:
            fallback_fn: Fallback function
            context: Context to pass

        Returns:
            Fallback result

        Raises:
            Exception if fallback also fails
        """
        try:
            result = await asyncio.wait_for(
                fallback_fn(context),
                timeout=self.config.fallback_timeout
            )
            return result

        except asyncio.TimeoutError:
            logger.error(f"⏱️  Fallback LLM timeout after {self.config.fallback_timeout}s")
            raise Exception(f"Fallback LLM timeout after {self.config.fallback_timeout}s")

        except Exception as e:
            logger.error(f"❌ Fallback LLM also failed: {type(e).__name__}: {e}")
            raise

    def _on_success(self):
        """Handle successful call to primary"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count_in_half_open += 1

            if self.success_count_in_half_open >= self.config.half_open_max_calls:
                # Recovery successful
                logger.info("✅ Primary LLM recovered - closing circuit")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count_in_half_open = 0
        else:
            # Reset failure count on success
            if self.failure_count > 0:
                logger.debug(f"✅ Primary success - resetting failure count (was {self.failure_count})")
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed call to primary"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test - back to OPEN
            logger.warning("❌ Primary still failing - circuit back to OPEN")
            self.state = CircuitState.OPEN
            self.success_count_in_half_open = 0

        elif self.failure_count >= self.config.failure_threshold:
            # Too many failures - open circuit
            logger.warning(f"🚨 Circuit breaker OPEN after {self.failure_count} failures")
            self.state = CircuitState.OPEN

        else:
            logger.warning(f"⚠️  Primary failure {self.failure_count}/{self.config.failure_threshold}")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.state != CircuitState.OPEN:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout

    def _time_until_retry(self) -> float:
        """Get seconds until retry attempt"""
        if self.state != CircuitState.OPEN:
            return 0

        time_since_failure = time.time() - self.last_failure_time
        return max(0, self.config.recovery_timeout - time_since_failure)

    def get_state(self) -> dict:
        """
        Get current circuit breaker state

        Returns:
            State information
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "time_since_failure": time.time() - self.last_failure_time if self.last_failure_time > 0 else None,
            "time_until_retry": self._time_until_retry() if self.state == CircuitState.OPEN else None
        }

    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        logger.info("🔄 Manually resetting circuit breaker to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count_in_half_open = 0
