#!/usr/bin/env python3
"""
Advanced Retry Logic
Implements various retry strategies with backoff algorithms and jitter
"""

import asyncio
import logging
import random
from typing import Callable, Optional, Type, Tuple, Union
from enum import Enum
from dataclasses import dataclass

from .exceptions import (
    UltravoxError,
    TimeoutError,
    NetworkError,
    ServiceUnavailableError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry logic"""

    CONSTANT = "constant"  # Fixed delay
    LINEAR = "linear"  # Linear increase
    EXPONENTIAL = "exponential"  # Exponential increase (2^n)
    FIBONACCI = "fibonacci"  # Fibonacci sequence
    EXPONENTIAL_JITTER = "exponential_jitter"  # Exponential with full jitter
    DECORRELATED_JITTER = "decorrelated_jitter"  # AWS decorrelated jitter


class JitterType(Enum):
    """Jitter types for randomization"""

    NONE = "none"  # No jitter
    FULL = "full"  # Random [0, backoff]
    EQUAL = "equal"  # backoff/2 + random[0, backoff/2]
    DECORRELATED = "decorrelated"  # min(cap, random[base, previous*3])


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter_type: JitterType = JitterType.FULL
    exponential_base: float = 2.0
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        TimeoutError,
        NetworkError,
        ServiceUnavailableError,
    )
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    on_retry: Optional[Callable] = None


class RetryStrategy:
    """Base class for retry strategies"""

    def __init__(self, config: RetryConfig):
        self.config = config
        self._fibonacci_cache = [0, 1]

    def calculate_delay(self, attempt: int, previous_delay: Optional[float] = None) -> float:
        """Calculate delay for given attempt number"""
        if self.config.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.config.base_delay

        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.base_delay * attempt

        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))

        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            delay = self.config.base_delay * self._get_fibonacci(attempt)

        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = self.config.base_delay * (
                self.config.exponential_base ** (attempt - 1)
            )
            delay = self._apply_jitter(base_delay, JitterType.FULL)

        elif self.config.backoff_strategy == BackoffStrategy.DECORRELATED_JITTER:
            if previous_delay is None:
                previous_delay = self.config.base_delay
            delay = min(
                self.config.max_delay,
                random.uniform(self.config.base_delay, previous_delay * 3),
            )

        else:
            delay = self.config.base_delay

        # Apply jitter if configured (except for decorrelated which handles its own)
        if (
            self.config.jitter_type != JitterType.NONE
            and self.config.backoff_strategy != BackoffStrategy.DECORRELATED_JITTER
        ):
            delay = self._apply_jitter(delay, self.config.jitter_type)

        # Cap at max_delay
        return min(delay, self.config.max_delay)

    def _apply_jitter(self, delay: float, jitter_type: JitterType) -> float:
        """Apply jitter to delay"""
        if jitter_type == JitterType.NONE:
            return delay

        elif jitter_type == JitterType.FULL:
            # Random value between 0 and delay
            return random.uniform(0, delay)

        elif jitter_type == JitterType.EQUAL:
            # Half deterministic, half random
            return delay / 2 + random.uniform(0, delay / 2)

        return delay

    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number (cached)"""
        while len(self._fibonacci_cache) <= n:
            self._fibonacci_cache.append(
                self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            )
        return self._fibonacci_cache[n]

    def is_retryable(self, exception: Exception) -> bool:
        """Check if exception should trigger retry"""
        # Non-retryable takes precedence
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        # Check if it's in retryable list
        if isinstance(exception, self.config.retryable_exceptions):
            return True

        # Check if it's an UltravoxError and is recoverable
        if isinstance(exception, UltravoxError):
            return exception.recoverable

        return False


async def retry_async(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
):
    """
    Retry async function with configurable strategy

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries exhausted
    """
    config = config or RetryConfig()
    strategy = RetryStrategy(config)

    last_exception = None
    previous_delay = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            result = await func(*args, **kwargs)
            if attempt > 1:
                logger.info(f"Retry succeeded on attempt {attempt}")
            return result

        except Exception as e:
            last_exception = e

            # Check if should retry
            if not strategy.is_retryable(e):
                logger.debug(f"Exception {type(e).__name__} is not retryable")
                raise

            # Last attempt - don't wait
            if attempt >= config.max_attempts:
                logger.warning(
                    f"All {config.max_attempts} retry attempts exhausted for {func.__name__}"
                )
                raise

            # Calculate delay
            delay = strategy.calculate_delay(attempt, previous_delay)
            previous_delay = delay

            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: "
                f"{type(e).__name__}: {e}. Retrying in {delay:.2f}s"
            )

            # Call retry callback if provided
            if config.on_retry:
                try:
                    if asyncio.iscoroutinefunction(config.on_retry):
                        await config.on_retry(attempt, e, delay)
                    else:
                        config.on_retry(attempt, e, delay)
                except Exception as callback_error:
                    logger.error(f"Error in retry callback: {callback_error}")

            # Wait before retry
            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def retry_sync(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
):
    """
    Retry synchronous function with configurable strategy

    Args:
        func: Sync function to retry
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries exhausted
    """
    import time

    config = config or RetryConfig()
    strategy = RetryStrategy(config)

    last_exception = None
    previous_delay = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            result = func(*args, **kwargs)
            if attempt > 1:
                logger.info(f"Retry succeeded on attempt {attempt}")
            return result

        except Exception as e:
            last_exception = e

            # Check if should retry
            if not strategy.is_retryable(e):
                logger.debug(f"Exception {type(e).__name__} is not retryable")
                raise

            # Last attempt - don't wait
            if attempt >= config.max_attempts:
                logger.warning(
                    f"All {config.max_attempts} retry attempts exhausted for {func.__name__}"
                )
                raise

            # Calculate delay
            delay = strategy.calculate_delay(attempt, previous_delay)
            previous_delay = delay

            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: "
                f"{type(e).__name__}: {e}. Retrying in {delay:.2f}s"
            )

            # Call retry callback if provided
            if config.on_retry:
                try:
                    config.on_retry(attempt, e, delay)
                except Exception as callback_error:
                    logger.error(f"Error in retry callback: {callback_error}")

            # Wait before retry
            time.sleep(delay)

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


# Convenience functions for common retry patterns


async def retry_with_exponential_backoff(
    func: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs,
):
    """Retry with exponential backoff and full jitter"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
    )
    return await retry_async(func, *args, config=config, **kwargs)


async def retry_network_errors(
    func: Callable,
    *args,
    max_attempts: int = 5,
    **kwargs,
):
    """Retry network-related errors with exponential backoff"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=1.0,
        max_delay=30.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
        retryable_exceptions=(NetworkError, TimeoutError, ServiceUnavailableError),
    )
    return await retry_async(func, *args, config=config, **kwargs)


async def retry_rate_limited(
    func: Callable,
    *args,
    max_attempts: int = 3,
    **kwargs,
):
    """Retry rate-limited requests with longer delays"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=5.0,
        max_delay=120.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        retryable_exceptions=(RateLimitError,),
    )
    return await retry_async(func, *args, config=config, **kwargs)
