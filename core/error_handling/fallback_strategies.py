#!/usr/bin/env python3
"""
Fallback Strategies
Implements various fallback patterns for graceful degradation
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import OrderedDict

from .exceptions import UltravoxError, ServiceUnavailableError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class FallbackResult:
    """Result of fallback execution"""

    success: bool
    value: Any
    source: str  # Which fallback succeeded
    attempts: int
    errors: List[Exception]
    execution_time: float


class CascadingFallback:
    """
    Cascading fallback - tries multiple sources in priority order
    Example: Primary LLM -> Fallback LLM -> Cached response
    """

    def __init__(self, name: str = "cascading_fallback"):
        self.name = name
        self.sources: List[tuple[str, Callable]] = []

    def add_source(self, name: str, func: Callable, priority: Optional[int] = None):
        """
        Add fallback source

        Args:
            name: Source name
            func: Callable to execute
            priority: Lower number = higher priority. If None, appends to end
        """
        if priority is not None:
            self.sources.insert(priority, (name, func))
        else:
            self.sources.append((name, func))

    async def execute(self, *args, **kwargs) -> FallbackResult:
        """
        Execute cascading fallback

        Args:
            *args: Arguments to pass to each source
            **kwargs: Keyword arguments to pass to each source

        Returns:
            FallbackResult with success status and value
        """
        start_time = datetime.utcnow()
        errors = []

        for attempt, (source_name, source_func) in enumerate(self.sources, 1):
            try:
                logger.info(
                    f"Trying fallback source '{source_name}' "
                    f"(attempt {attempt}/{len(self.sources)})"
                )

                # Execute (async or sync)
                if asyncio.iscoroutinefunction(source_func):
                    result = await source_func(*args, **kwargs)
                else:
                    result = source_func(*args, **kwargs)

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                logger.info(f"Fallback source '{source_name}' succeeded")

                return FallbackResult(
                    success=True,
                    value=result,
                    source=source_name,
                    attempts=attempt,
                    errors=errors,
                    execution_time=execution_time,
                )

            except Exception as e:
                logger.warning(f"Fallback source '{source_name}' failed: {e}")
                errors.append(e)
                continue

        # All sources failed
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"All {len(self.sources)} fallback sources failed")

        return FallbackResult(
            success=False,
            value=None,
            source="none",
            attempts=len(self.sources),
            errors=errors,
            execution_time=execution_time,
        )


class CachedFallback:
    """
    Cached fallback - returns cached value when primary fails
    Uses LRU cache with TTL
    """

    def __init__(
        self, max_cache_size: int = 100, ttl_seconds: int = 300, name: str = "cached_fallback"
    ):
        self.name = name
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Simple key generation - can be overridden for complex cases
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        return f"{args_str}_{kwargs_str}"

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached value is still valid"""
        age = (datetime.utcnow() - timestamp).total_seconds()
        return age < self.ttl_seconds

    async def execute(self, func: Callable, *args, use_cache: bool = True, **kwargs) -> Any:
        """
        Execute with cached fallback

        Args:
            func: Primary function to execute
            *args: Arguments for func
            use_cache: Whether to use cache on failure
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or cached value
        """
        cache_key = self._get_cache_key(*args, **kwargs)

        try:
            # Try primary function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Update cache on success
            self._cache[cache_key] = (result, datetime.utcnow())

            # Maintain cache size limit
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)  # Remove oldest

            logger.debug(f"Cached fallback: primary succeeded, cached result")
            return result

        except Exception as e:
            logger.warning(f"Cached fallback: primary failed: {e}")

            if not use_cache:
                raise

            # Try cache
            if cache_key in self._cache:
                cached_value, cached_time = self._cache[cache_key]

                if self._is_cache_valid(cached_time):
                    age = (datetime.utcnow() - cached_time).total_seconds()
                    logger.info(
                        f"Cached fallback: returning cached value "
                        f"(age: {age:.1f}s, ttl: {self.ttl_seconds}s)"
                    )
                    return cached_value
                else:
                    logger.debug("Cached fallback: cached value expired")
                    del self._cache[cache_key]

            # No valid cache - re-raise original error
            logger.error("Cached fallback: no valid cached value available")
            raise

    def clear_cache(self):
        """Clear all cached values"""
        self._cache.clear()
        logger.info(f"Cached fallback '{self.name}': cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = datetime.utcnow()
        valid_entries = sum(
            1 for _, (_, timestamp) in self._cache.items() if self._is_cache_valid(timestamp)
        )

        return {
            "name": self.name,
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "max_size": self.max_cache_size,
            "ttl_seconds": self.ttl_seconds,
        }


class DefaultValueFallback:
    """
    Default value fallback - returns a default value when operation fails
    Useful for non-critical operations
    """

    def __init__(self, default_value: Any, name: str = "default_value_fallback"):
        self.name = name
        self.default_value = default_value

    async def execute(
        self,
        func: Callable,
        *args,
        default: Optional[Any] = None,
        log_errors: bool = True,
        **kwargs,
    ) -> Any:
        """
        Execute with default value fallback

        Args:
            func: Function to execute
            *args: Arguments for func
            default: Override default value for this call
            log_errors: Whether to log errors
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or default value
        """
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            if log_errors:
                logger.warning(
                    f"Default value fallback '{self.name}': operation failed, "
                    f"returning default value. Error: {e}"
                )

            return default if default is not None else self.default_value


class DegradedServiceFallback:
    """
    Degraded service fallback - provides simplified/degraded functionality
    Example: Full voice synthesis -> Simple TTS -> Text-only response
    """

    def __init__(self, name: str = "degraded_service"):
        self.name = name
        self.degradation_levels: List[tuple[str, Callable]] = []

    def add_degradation_level(self, level_name: str, func: Callable):
        """
        Add degradation level (in order from best to worst)

        Args:
            level_name: Name of degradation level
            func: Function providing degraded service
        """
        self.degradation_levels.append((level_name, func))

    async def execute(self, *args, **kwargs) -> FallbackResult:
        """
        Execute with progressive degradation

        Returns:
            FallbackResult with degradation level used
        """
        start_time = datetime.utcnow()
        errors = []

        for attempt, (level_name, level_func) in enumerate(self.degradation_levels, 1):
            try:
                logger.info(f"Trying degradation level: {level_name}")

                if asyncio.iscoroutinefunction(level_func):
                    result = await level_func(*args, **kwargs)
                else:
                    result = level_func(*args, **kwargs)

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                if attempt > 1:
                    logger.warning(
                        f"Service degraded to level '{level_name}' "
                        f"(attempt {attempt}/{len(self.degradation_levels)})"
                    )

                return FallbackResult(
                    success=True,
                    value=result,
                    source=level_name,
                    attempts=attempt,
                    errors=errors,
                    execution_time=execution_time,
                )

            except Exception as e:
                logger.warning(f"Degradation level '{level_name}' failed: {e}")
                errors.append(e)
                continue

        # All degradation levels failed
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"All {len(self.degradation_levels)} degradation levels failed")

        return FallbackResult(
            success=False,
            value=None,
            source="none",
            attempts=len(self.degradation_levels),
            errors=errors,
            execution_time=execution_time,
        )


# Convenience function for simple fallback


async def with_fallback(
    primary: Callable,
    fallback: Callable,
    *args,
    primary_name: str = "primary",
    fallback_name: str = "fallback",
    **kwargs,
) -> FallbackResult:
    """
    Execute with simple two-level fallback

    Args:
        primary: Primary function to try first
        fallback: Fallback function if primary fails
        *args: Arguments for both functions
        primary_name: Name for logging
        fallback_name: Name for logging
        **kwargs: Keyword arguments for both functions

    Returns:
        FallbackResult
    """
    cascade = CascadingFallback("simple_fallback")
    cascade.add_source(primary_name, primary)
    cascade.add_source(fallback_name, fallback)

    return await cascade.execute(*args, **kwargs)
