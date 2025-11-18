#!/usr/bin/env python3
"""
Retry Policy with Exponential Backoff
Automatically retries failed requests with increasing delays

Strategies:
- Exponential backoff: delay *= multiplier
- Jitter: Random delay to prevent thundering herd
- Max attempts: Stop after N failures
"""

import asyncio
import random
import time
import logging
from typing import Optional, Callable, Any, Tuple, Type
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry backoff strategies"""
    FIXED = "fixed"              # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Delay increases exponentially
    LINEAR = "linear"            # Delay increases linearly


@dataclass
class RetryPolicyConfig:
    """
    Retry policy configuration

    Args:
        max_attempts: Maximum number of retry attempts (including first try)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        multiplier: Backoff multiplier for exponential strategy
        strategy: Retry backoff strategy
        jitter: Add random jitter to delays (prevents thundering herd)
        retryable_exceptions: Tuple of exception types to retry
    """
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


@dataclass
class RetryStats:
    """Retry statistics"""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retries: int = 0
    total_backoff_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "total_retries": self.total_retries,
            "total_backoff_time_ms": self.total_backoff_time * 1000,
            "avg_retries_per_call": (
                self.total_retries / self.total_attempts if self.total_attempts > 0 else 0
            ),
            "success_rate": (
                self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0
            )
        }


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted"""
    pass


class RetryPolicy:
    """
    Retry policy with exponential backoff and jitter

    Usage:
        retry_policy = RetryPolicy(
            name="groq_api",
            config=RetryPolicyConfig(max_attempts=5, initial_delay=1.0)
        )

        # Async function
        result = await retry_policy.execute(api_call, arg1, arg2, kwarg1=value)

        # Check stats
        print(retry_policy.get_stats())
    """

    def __init__(self, name: str, config: Optional[RetryPolicyConfig] = None):
        """
        Initialize retry policy

        Args:
            name: Policy identifier (e.g., "groq_api_retry")
            config: Retry policy configuration
        """
        self.name = name
        self.config = config or RetryPolicyConfig()
        self.stats = RetryStats()

        logger.info(
            f"🔄 Retry Policy '{name}' initialized\n"
            f"   Max attempts: {self.config.max_attempts}\n"
            f"   Initial delay: {self.config.initial_delay}s\n"
            f"   Strategy: {self.config.strategy.value}\n"
            f"   Jitter: {self.config.jitter}"
        )

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic

        Args:
            func: Async function to execute
            *args: Function positional arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            RetryExhaustedError: If all retry attempts fail
            Exception: Last exception if all retries failed
        """
        self.stats.total_attempts += 1
        attempt = 1
        last_exception = None

        while attempt <= self.config.max_attempts:
            try:
                start_time = time.time()

                # Execute function
                result = await func(*args, **kwargs)

                # Success!
                if attempt > 1:
                    logger.info(
                        f"✅ Retry '{self.name}' succeeded on attempt {attempt}/{self.config.max_attempts}"
                    )

                self.stats.successful_attempts += 1
                if attempt > 1:
                    self.stats.total_retries += (attempt - 1)

                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.warning(f"⚠️  Retry '{self.name}' got non-retryable exception: {type(e).__name__}")
                    self.stats.failed_attempts += 1
                    raise

                # Check if we have more attempts
                if attempt >= self.config.max_attempts:
                    logger.error(
                        f"❌ Retry '{self.name}' exhausted all {self.config.max_attempts} attempts\n"
                        f"   Last error: {type(e).__name__}: {str(e)}"
                    )
                    self.stats.failed_attempts += 1
                    raise RetryExhaustedError(
                        f"Retry policy '{self.name}' exhausted after {self.config.max_attempts} attempts. "
                        f"Last error: {type(e).__name__}: {str(e)}"
                    ) from e

                # Calculate backoff delay
                delay = self._calculate_delay(attempt)
                self.stats.total_backoff_time += delay

                logger.warning(
                    f"⚠️  Retry '{self.name}' attempt {attempt}/{self.config.max_attempts} failed: {type(e).__name__}\n"
                    f"   Retrying in {delay:.2f}s..."
                )

                # Wait before retry
                await asyncio.sleep(delay)

                attempt += 1

        # Should never reach here, but just in case
        self.stats.failed_attempts += 1
        if last_exception:
            raise last_exception
        raise RetryExhaustedError(f"Retry policy '{self.name}' failed after {self.config.max_attempts} attempts")

    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return isinstance(exception, self.config.retryable_exceptions)

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate backoff delay based on strategy

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.initial_delay

        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.initial_delay * attempt

        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.multiplier ** (attempt - 1))

        else:
            delay = self.config.initial_delay

        # Cap delay at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            # Add random jitter between 0% and 25% of delay
            jitter_amount = random.uniform(0, delay * 0.25)
            delay += jitter_amount

        return delay

    def get_stats(self) -> dict:
        """Get retry statistics"""
        return self.stats.to_dict()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = RetryStats()
        logger.info(f"📊 Retry policy '{self.name}' stats reset")


class RetryPolicyRegistry:
    """
    Registry for managing multiple retry policies

    Usage:
        registry = RetryPolicyRegistry()

        # Get or create retry policy
        retry = registry.get("groq_api")

        # Execute with retry
        result = await retry.execute(api_function, arg1, arg2)

        # Get all stats
        all_stats = registry.get_all_stats()
    """

    def __init__(self):
        self._policies: dict[str, RetryPolicy] = {}
        self._default_config = RetryPolicyConfig()
        logger.info("🔄 Retry Policy Registry initialized")

    def get(self, name: str, config: Optional[RetryPolicyConfig] = None) -> RetryPolicy:
        """
        Get or create retry policy

        Args:
            name: Policy identifier
            config: Optional custom configuration

        Returns:
            RetryPolicy instance
        """
        if name not in self._policies:
            self._policies[name] = RetryPolicy(
                name=name,
                config=config or self._default_config
            )

        return self._policies[name]

    def remove(self, name: str):
        """Remove retry policy from registry"""
        if name in self._policies:
            del self._policies[name]
            logger.info(f"🗑️  Retry policy '{name}' removed from registry")

    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all retry policies"""
        return {
            name: policy.get_stats()
            for name, policy in self._policies.items()
        }

    def reset_all_stats(self):
        """Reset statistics for all retry policies"""
        for policy in self._policies.values():
            policy.reset_stats()
        logger.info("📊 All retry policy stats reset")

    def set_default_config(self, config: RetryPolicyConfig):
        """Set default configuration for new retry policies"""
        self._default_config = config
        logger.info("⚙️  Default retry policy config updated")


# Global registry instance
_retry_policy_registry: Optional[RetryPolicyRegistry] = None


def get_retry_policy_registry() -> RetryPolicyRegistry:
    """Get global retry policy registry (singleton)"""
    global _retry_policy_registry

    if _retry_policy_registry is None:
        _retry_policy_registry = RetryPolicyRegistry()

    return _retry_policy_registry
