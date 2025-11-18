"""
Advanced Rate Limiter with Redis support and multiple strategies.

Features:
- Multiple backends: Memory, Redis (distributed)
- Multiple strategies: Fixed Window, Sliding Window, Token Bucket
- Per-IP, Per-User, Per-Endpoint rate limiting
- Dynamic rate limit adjustment
- Gradual backoff and throttling
- Whitelist/blacklist support
- Comprehensive metrics

Author: Ultravox Team
Version: 1.0.0
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"  # Simple counter per time window
    SLIDING_WINDOW = "sliding_window"  # More accurate, prevents boundary issues
    TOKEN_BUCKET = "token_bucket"  # Allows bursts within limits


class RateLimitBackend(str, Enum):
    """Rate limit storage backends."""

    MEMORY = "memory"  # In-memory (single instance)
    REDIS = "redis"  # Redis (distributed, multi-instance)


class RateLimitScope(str, Enum):
    """Scope for rate limiting."""

    IP = "ip"  # Per client IP
    USER = "user"  # Per authenticated user
    API_KEY = "api_key"  # Per API key
    ENDPOINT = "endpoint"  # Per endpoint
    GLOBAL = "global"  # Global limit


class RateLimitConfig(BaseModel):
    """Rate limiter configuration."""

    # Strategy
    strategy: RateLimitStrategy = Field(
        RateLimitStrategy.SLIDING_WINDOW, description="Rate limiting strategy"
    )
    backend: RateLimitBackend = Field(
        RateLimitBackend.MEMORY, description="Storage backend"
    )

    # Limits
    requests_per_minute: int = Field(60, description="Max requests per minute")
    requests_per_hour: int = Field(1000, description="Max requests per hour")
    requests_per_day: int = Field(10000, description="Max requests per day")

    # Token bucket specific
    burst_size: int = Field(10, description="Max burst size for token bucket")
    refill_rate: float = Field(1.0, description="Tokens refilled per second")

    # Scopes (multiple can be active)
    enable_ip_limit: bool = Field(True, description="Enable per-IP limiting")
    enable_user_limit: bool = Field(True, description="Enable per-user limiting")
    enable_api_key_limit: bool = Field(True, description="Enable per-API-key limiting")
    enable_endpoint_limit: bool = Field(False, description="Enable per-endpoint limiting")
    enable_global_limit: bool = Field(False, description="Enable global limiting")

    # Whitelist/Blacklist
    whitelist_ips: Set[str] = Field(default_factory=set, description="Whitelisted IPs")
    blacklist_ips: Set[str] = Field(default_factory=set, description="Blacklisted IPs")
    whitelist_users: Set[str] = Field(
        default_factory=set, description="Whitelisted users"
    )

    # Redis configuration
    redis_url: Optional[str] = Field(None, description="Redis URL for distributed backend")
    redis_key_prefix: str = Field(
        "ultravox:ratelimit:", description="Redis key prefix"
    )

    # Backoff configuration
    enable_gradual_backoff: bool = Field(
        True, description="Enable gradual throttling instead of hard block"
    )
    backoff_threshold: float = Field(
        0.8, description="Start throttling at this % of limit"
    )
    backoff_delay_ms: int = Field(100, description="Max delay in milliseconds when throttling")

    # Cleanup
    cleanup_interval_seconds: int = Field(
        300, description="Interval for cleaning old entries"
    )


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool  # Whether request is allowed
    remaining: int  # Remaining requests in current window
    reset_at: float  # Timestamp when limit resets
    retry_after: Optional[int] = None  # Seconds to wait before retry (if blocked)
    throttle_delay_ms: Optional[int] = None  # Milliseconds to delay (gradual backoff)


class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple backends and strategies.

    Example:
        >>> config = RateLimitConfig(
        ...     requests_per_minute=60,
        ...     backend=RateLimitBackend.REDIS,
        ...     redis_url="redis://localhost:6379"
        ... )
        >>> limiter = AdvancedRateLimiter(config)
        >>> result = limiter.check_limit("192.168.1.1", scope=RateLimitScope.IP)
        >>> if not result.allowed:
        ...     print(f"Rate limited! Retry after {result.retry_after}s")
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config

        # In-memory storage (used even with Redis for local caching)
        self._fixed_window_counters: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._sliding_window_requests: Dict[str, deque] = defaultdict(deque)
        self._token_buckets: Dict[str, Tuple[float, float]] = {}  # (tokens, last_update)

        # Redis client
        self._redis_client = None
        if config.backend == RateLimitBackend.REDIS:
            self._init_redis()

        # Metrics
        self._total_requests = 0
        self._total_blocked = 0
        self._total_throttled = 0

        # Last cleanup
        self._last_cleanup = time.time()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        if not self.config.redis_url:
            raise ValueError("Redis backend requires redis_url configuration")

        try:
            import redis

            self._redis_client = redis.from_url(
                self.config.redis_url, decode_responses=True
            )
            # Test connection
            self._redis_client.ping()
        except ImportError:
            raise ImportError(
                "Redis backend requires 'redis' package. Install with: pip install redis"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def check_limit(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.IP,
        endpoint: Optional[str] = None,
    ) -> RateLimitResult:
        """
        Check if request is allowed under rate limits.

        Args:
            identifier: Identifier (IP, user_id, api_key, etc.)
            scope: Rate limit scope
            endpoint: Optional endpoint path for per-endpoint limiting

        Returns:
            RateLimitResult with allow/deny decision and metadata

        Example:
            >>> result = limiter.check_limit("192.168.1.1", RateLimitScope.IP)
            >>> if result.throttle_delay_ms:
            ...     await asyncio.sleep(result.throttle_delay_ms / 1000)
        """
        self._total_requests += 1

        # Periodic cleanup
        self._cleanup_if_needed()

        # Check whitelist
        if self._is_whitelisted(identifier, scope):
            return RateLimitResult(
                allowed=True, remaining=999999, reset_at=time.time() + 60
            )

        # Check blacklist
        if self._is_blacklisted(identifier, scope):
            self._total_blocked += 1
            return RateLimitResult(
                allowed=False, remaining=0, reset_at=time.time() + 3600, retry_after=3600
            )

        # Generate key
        key = self._generate_key(identifier, scope, endpoint)

        # Check limit based on strategy
        if self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            result = self._check_fixed_window(key)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            result = self._check_sliding_window(key)
        else:  # TOKEN_BUCKET
            result = self._check_token_bucket(key)

        # Apply gradual backoff if enabled
        if self.config.enable_gradual_backoff and result.allowed:
            limit = self.config.requests_per_minute
            usage_ratio = (limit - result.remaining) / limit

            if usage_ratio >= self.config.backoff_threshold:
                # Calculate throttle delay
                excess_ratio = (usage_ratio - self.config.backoff_threshold) / (
                    1.0 - self.config.backoff_threshold
                )
                delay_ms = int(excess_ratio * self.config.backoff_delay_ms)
                result.throttle_delay_ms = delay_ms
                self._total_throttled += 1

        if not result.allowed:
            self._total_blocked += 1

        return result

    def _check_fixed_window(self, key: str) -> RateLimitResult:
        """Fixed window rate limiting."""
        now = time.time()
        window_start = int(now / 60) * 60  # 1-minute windows
        limit = self.config.requests_per_minute

        if self._redis_client:
            # Redis implementation
            redis_key = f"{self.config.redis_key_prefix}{key}:{window_start}"
            pipe = self._redis_client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, 60)  # Expire after 60 seconds
            count, _ = pipe.execute()
        else:
            # Memory implementation
            count = self._fixed_window_counters[key][window_start]
            self._fixed_window_counters[key][window_start] += 1
            count += 1

        allowed = count <= limit
        remaining = max(0, limit - count)
        reset_at = window_start + 60

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=int(reset_at - now) if not allowed else None,
        )

    def _check_sliding_window(self, key: str) -> RateLimitResult:
        """Sliding window rate limiting (more accurate)."""
        now = time.time()
        window_start = now - 60  # 1-minute window
        limit = self.config.requests_per_minute

        if self._redis_client:
            # Redis sorted set implementation
            redis_key = f"{self.config.redis_key_prefix}{key}"
            pipe = self._redis_client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(redis_key, 0, window_start)

            # Count entries in window
            pipe.zcard(redis_key)

            # Add current request
            pipe.zadd(redis_key, {str(now): now})

            # Set expiration
            pipe.expire(redis_key, 60)

            _, count, _, _ = pipe.execute()
        else:
            # Memory implementation
            requests = self._sliding_window_requests[key]

            # Remove old requests
            while requests and requests[0] < window_start:
                requests.popleft()

            count = len(requests)
            requests.append(now)

        allowed = count < limit
        remaining = max(0, limit - count - 1)
        reset_at = now + 60

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=int(60 - (now - window_start)) if not allowed else None,
        )

    def _check_token_bucket(self, key: str) -> RateLimitResult:
        """Token bucket rate limiting (allows bursts)."""
        now = time.time()

        if self._redis_client:
            # Redis implementation using Lua script for atomicity
            lua_script = """
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local refill_rate = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])

            local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
            local tokens = tonumber(bucket[1]) or capacity
            local last_update = tonumber(bucket[2]) or now

            -- Refill tokens
            local elapsed = now - last_update
            tokens = math.min(capacity, tokens + elapsed * refill_rate)

            -- Consume one token
            if tokens >= 1 then
                tokens = tokens - 1
                redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
                redis.call('EXPIRE', key, 60)
                return {1, math.floor(tokens)}
            else
                return {0, 0}
            end
            """
            redis_key = f"{self.config.redis_key_prefix}{key}"
            result = self._redis_client.eval(
                lua_script,
                1,
                redis_key,
                self.config.burst_size,
                self.config.refill_rate,
                now,
            )
            allowed = result[0] == 1
            remaining = int(result[1])
        else:
            # Memory implementation
            if key not in self._token_buckets:
                self._token_buckets[key] = (
                    self.config.burst_size - 1,
                    now,
                )  # Start with full bucket minus 1
                allowed = True
                remaining = self.config.burst_size - 1
            else:
                tokens, last_update = self._token_buckets[key]

                # Refill tokens
                elapsed = now - last_update
                tokens = min(
                    self.config.burst_size, tokens + elapsed * self.config.refill_rate
                )

                # Try to consume 1 token
                if tokens >= 1:
                    tokens -= 1
                    self._token_buckets[key] = (tokens, now)
                    allowed = True
                    remaining = int(tokens)
                else:
                    allowed = False
                    remaining = 0

        reset_at = now + (1.0 / self.config.refill_rate)  # Time to get next token

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=int(1.0 / self.config.refill_rate) if not allowed else None,
        )

    def _is_whitelisted(self, identifier: str, scope: RateLimitScope) -> bool:
        """Check if identifier is whitelisted."""
        if scope == RateLimitScope.IP:
            return identifier in self.config.whitelist_ips
        elif scope == RateLimitScope.USER:
            return identifier in self.config.whitelist_users
        return False

    def _is_blacklisted(self, identifier: str, scope: RateLimitScope) -> bool:
        """Check if identifier is blacklisted."""
        if scope == RateLimitScope.IP:
            return identifier in self.config.blacklist_ips
        return False

    def _generate_key(
        self, identifier: str, scope: RateLimitScope, endpoint: Optional[str]
    ) -> str:
        """Generate cache key for rate limit."""
        parts = [scope.value, identifier]
        if endpoint and self.config.enable_endpoint_limit:
            parts.append(endpoint)
        return ":".join(parts)

    def _cleanup_if_needed(self) -> None:
        """Clean up old entries periodically."""
        now = time.time()
        if now - self._last_cleanup < self.config.cleanup_interval_seconds:
            return

        # Clean fixed window counters
        cutoff = int(now / 60) * 60 - 300  # Keep last 5 minutes
        for key in list(self._fixed_window_counters.keys()):
            self._fixed_window_counters[key] = {
                k: v
                for k, v in self._fixed_window_counters[key].items()
                if k >= cutoff
            }
            if not self._fixed_window_counters[key]:
                del self._fixed_window_counters[key]

        # Clean sliding window requests
        window_start = now - 60
        for key in list(self._sliding_window_requests.keys()):
            requests = self._sliding_window_requests[key]
            while requests and requests[0] < window_start:
                requests.popleft()
            if not requests:
                del self._sliding_window_requests[key]

        # Clean token buckets (remove inactive ones)
        inactive_cutoff = now - 3600  # Remove buckets inactive for 1 hour
        for key in list(self._token_buckets.keys()):
            _, last_update = self._token_buckets[key]
            if last_update < inactive_cutoff:
                del self._token_buckets[key]

        self._last_cleanup = now

    def get_metrics(self) -> Dict[str, int]:
        """Get rate limiter metrics."""
        return {
            "total_requests": self._total_requests,
            "total_blocked": self._total_blocked,
            "total_throttled": self._total_throttled,
            "block_rate": (
                self._total_blocked / self._total_requests
                if self._total_requests > 0
                else 0
            ),
            "active_keys": (
                len(self._fixed_window_counters)
                + len(self._sliding_window_requests)
                + len(self._token_buckets)
            ),
        }

    def reset_limits(self, identifier: str, scope: RateLimitScope) -> None:
        """Reset rate limits for an identifier."""
        key_prefix = f"{scope.value}:{identifier}"

        # Clear memory storage
        for key in list(self._fixed_window_counters.keys()):
            if key.startswith(key_prefix):
                del self._fixed_window_counters[key]

        for key in list(self._sliding_window_requests.keys()):
            if key.startswith(key_prefix):
                del self._sliding_window_requests[key]

        for key in list(self._token_buckets.keys()):
            if key.startswith(key_prefix):
                del self._token_buckets[key]

        # Clear Redis if enabled
        if self._redis_client:
            pattern = f"{self.config.redis_key_prefix}{key_prefix}*"
            for key in self._redis_client.scan_iter(match=pattern):
                self._redis_client.delete(key)
