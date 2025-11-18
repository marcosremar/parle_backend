#!/usr/bin/env python3
"""
Rate Limiting Middleware for FastAPI
Prevents API abuse with configurable rate limits per endpoint/user/IP

Strategies:
- Fixed Window: Simple counter per time window
- Sliding Window: More accurate, prevents burst at window boundaries
- Token Bucket: Allows burst traffic within limits

Storage backends:
- Memory: In-process dict (default, single instance)
- Redis: Distributed rate limiting (multi-instance)
"""

import time
import logging
from typing import Optional, Dict, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import asyncio

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"       # Simple counter per time window
    SLIDING_WINDOW = "sliding_window"   # Sliding window log
    TOKEN_BUCKET = "token_bucket"       # Token bucket algorithm


@dataclass
class RateLimitConfig:
    """
    Rate limit configuration

    Args:
        requests_per_minute: Max requests per minute
        requests_per_hour: Max requests per hour
        burst_size: Max burst size for token bucket
        strategy: Rate limiting strategy
        key_func: Function to extract rate limit key from request
        whitelist: List of IPs/keys to whitelist
        enabled: Enable/disable rate limiting
    """
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    key_func: Optional[Callable] = None
    whitelist: list = field(default_factory=list)
    enabled: bool = True


@dataclass
class RateLimitEntry:
    """Rate limit entry for tracking requests"""
    # Fixed window
    count: int = 0
    window_start: float = 0.0

    # Sliding window
    request_times: list = field(default_factory=list)

    # Token bucket
    tokens: float = 0.0
    last_refill: float = 0.0


class RateLimiter:
    """
    Rate limiter implementation

    Usage:
        limiter = RateLimiter(
            config=RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000
            )
        )

        # Check if request allowed
        allowed, retry_after = await limiter.check_limit(key="user_123")

        if not allowed:
            raise HTTPException(429, detail=f"Rate limit exceeded. Retry after {retry_after}s")
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self.entries: Dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._lock = asyncio.Lock()
        self._cleanup_task = None

        if self.config.enabled:
            logger.info(
                f"🚦 Rate Limiter initialized\n"
                f"   Strategy: {self.config.strategy.value}\n"
                f"   Per minute: {self.config.requests_per_minute}\n"
                f"   Per hour: {self.config.requests_per_hour}\n"
                f"   Burst size: {self.config.burst_size}"
            )
            # Start cleanup task
            self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_old_entries()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def _cleanup_old_entries(self):
        """Remove old entries to prevent memory leak"""
        async with self._lock:
            current_time = time.time()
            keys_to_remove = []

            for key, entry in self.entries.items():
                # Remove entries older than 1 hour
                if current_time - entry.window_start > 3600:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.entries[key]

            if keys_to_remove:
                logger.debug(f"🧹 Cleaned up {len(keys_to_remove)} old rate limit entries")

    async def check_limit(self, key: str) -> Tuple[bool, float]:
        """
        Check if request is within rate limit

        Args:
            key: Rate limit key (e.g., user ID, IP address)

        Returns:
            Tuple of (allowed: bool, retry_after: float in seconds)
        """
        if not self.config.enabled:
            return True, 0.0

        # Check whitelist
        if key in self.config.whitelist:
            return True, 0.0

        async with self._lock:
            entry = self.entries[key]
            current_time = time.time()

            if self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(entry, current_time)

            elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(entry, current_time)

            elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(entry, current_time)

            else:
                return True, 0.0

    async def _check_fixed_window(self, entry: RateLimitEntry, current_time: float) -> Tuple[bool, float]:
        """Fixed window rate limiting"""
        window_size = 60.0  # 1 minute

        # Check if we're in a new window
        if current_time - entry.window_start >= window_size:
            entry.window_start = current_time
            entry.count = 0

        # Check limit
        if entry.count >= self.config.requests_per_minute:
            retry_after = window_size - (current_time - entry.window_start)
            return False, retry_after

        entry.count += 1
        return True, 0.0

    async def _check_sliding_window(self, entry: RateLimitEntry, current_time: float) -> Tuple[bool, float]:
        """Sliding window rate limiting"""
        minute_ago = current_time - 60.0
        hour_ago = current_time - 3600.0

        # Remove requests older than 1 hour
        entry.request_times = [t for t in entry.request_times if t > hour_ago]

        # Count requests in last minute and hour
        requests_last_minute = sum(1 for t in entry.request_times if t > minute_ago)
        requests_last_hour = len(entry.request_times)

        # Check minute limit
        if requests_last_minute >= self.config.requests_per_minute:
            oldest_in_window = min([t for t in entry.request_times if t > minute_ago])
            retry_after = 60.0 - (current_time - oldest_in_window)
            return False, retry_after

        # Check hour limit
        if requests_last_hour >= self.config.requests_per_hour:
            oldest_in_window = min(entry.request_times)
            retry_after = 3600.0 - (current_time - oldest_in_window)
            return False, retry_after

        # Allow request
        entry.request_times.append(current_time)
        return True, 0.0

    async def _check_token_bucket(self, entry: RateLimitEntry, current_time: float) -> Tuple[bool, float]:
        """Token bucket rate limiting"""
        # Initialize tokens
        if entry.tokens == 0.0 and entry.last_refill == 0.0:
            entry.tokens = self.config.burst_size
            entry.last_refill = current_time

        # Refill tokens based on time elapsed
        time_elapsed = current_time - entry.last_refill
        refill_rate = self.config.requests_per_minute / 60.0  # tokens per second
        tokens_to_add = time_elapsed * refill_rate

        entry.tokens = min(self.config.burst_size, entry.tokens + tokens_to_add)
        entry.last_refill = current_time

        # Check if we have tokens
        if entry.tokens < 1.0:
            retry_after = (1.0 - entry.tokens) / refill_rate
            return False, retry_after

        # Consume token
        entry.tokens -= 1.0
        return True, 0.0

    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        return {
            "enabled": self.config.enabled,
            "strategy": self.config.strategy.value,
            "requests_per_minute": self.config.requests_per_minute,
            "requests_per_hour": self.config.requests_per_hour,
            "active_keys": len(self.entries),
            "whitelist_size": len(self.config.whitelist)
        }

    async def reset_key(self, key: str):
        """Reset rate limit for a specific key"""
        async with self._lock:
            if key in self.entries:
                del self.entries[key]
                logger.info(f"🔄 Reset rate limit for key: {key}")

    async def shutdown(self):
        """Shutdown rate limiter"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting

    Usage:
        app = FastAPI()

        # Add rate limiting middleware
        app.add_middleware(
            RateLimitMiddleware,
            config=RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000
            )
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[RateLimitConfig] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Initialize middleware

        Args:
            app: FastAPI application
            config: Rate limit configuration
            key_func: Function to extract rate limit key from request
        """
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.limiter = RateLimiter(config=self.config)

        # Default key function: use client IP
        self.key_func = key_func or self._default_key_func

    def _default_key_func(self, request: Request) -> str:
        """Default key function: client IP"""
        # Try to get real IP from X-Forwarded-For header
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fallback to client host
        if request.client:
            return request.client.host

        return "unknown"

    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter"""
        if not self.config.enabled:
            return await call_next(request)

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Extract rate limit key
        key = self.key_func(request)

        # Check rate limit
        allowed, retry_after = await self.limiter.check_limit(key)

        if not allowed:
            logger.warning(f"🚫 Rate limit exceeded for {key} on {request.url.path}")

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please retry after {retry_after:.1f} seconds.",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Limit": str(self.config.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after))
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)

        return response


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get global rate limiter instance (singleton)"""
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config=config)

    return _rate_limiter
