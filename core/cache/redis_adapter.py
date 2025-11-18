"""
Redis Adapter (L2) - Redis-backed cache for persistence and multi-process sharing

Features:
- Persistent cache across process restarts
- Shared cache between multiple service instances
- TTL support
- Automatic serialization/deserialization
"""

import hashlib
import json
import pickle
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import logging

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


logger = logging.getLogger(__name__)


class RedisAdapter:
    """
    L2 Redis-backed cache

    Features:
    - Persistent storage
    - Multi-process sharing
    - Automatic TTL management
    - Graceful degradation (no Redis = no L2)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "ultravox:",
        default_ttl: int = 3600,
        max_size_mb: int = 1024,
    ):
        """
        Initialize Redis adapter

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
            max_size_mb: Max cache size in MB (for monitoring)
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.max_size_mb = max_size_mb

        self._client: Optional[redis.Redis] = None
        self._available = False

        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

        # Initialize connection
        if REDIS_AVAILABLE:
            self._connect()
        else:
            logger.warning(
                "⚠️  Redis not available (redis package not installed). "
                "L2 cache disabled. Install with: pip install redis"
            )

    def _connect(self) -> bool:
        """
        Connect to Redis

        Returns:
            True if connected successfully
        """
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            self._client.ping()
            self._available = True
            logger.info(f"✅ Redis connected: {self.host}:{self.port} (db={self.db})")
            return True

        except (RedisError, RedisConnectionError) as e:
            logger.warning(f"⚠️  Redis connection failed: {e}. L2 cache disabled.")
            self._available = False
            self._client = None
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self._available or not self._client:
            return None

        try:
            full_key = self._make_redis_key(key)
            data = self._client.get(full_key)

            if data is None:
                self._stats["misses"] += 1
                return None

            # Deserialize
            value = pickle.loads(data)
            self._stats["hits"] += 1
            return value

        except (RedisError, pickle.PickleError) as e:
            logger.warning(f"⚠️  Redis get error: {e}")
            self._stats["errors"] += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in Redis cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: use default_ttl)

        Returns:
            True if successfully cached
        """
        if not self._available or not self._client:
            return False

        try:
            full_key = self._make_redis_key(key)
            ttl = ttl or self.default_ttl

            # Serialize
            data = pickle.dumps(value)

            # Store with TTL
            self._client.setex(full_key, ttl, data)
            self._stats["sets"] += 1
            return True

        except (RedisError, pickle.PickleError) as e:
            logger.warning(f"⚠️  Redis set error: {e}")
            self._stats["errors"] += 1
            return False

    def delete(self, key: str) -> bool:
        """
        Delete value from Redis cache

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        if not self._available or not self._client:
            return False

        try:
            full_key = self._make_redis_key(key)
            result = self._client.delete(full_key)
            self._stats["deletes"] += 1
            return result > 0

        except RedisError as e:
            logger.warning(f"⚠️  Redis delete error: {e}")
            self._stats["errors"] += 1
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis cache

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        if not self._available or not self._client:
            return False

        try:
            full_key = self._make_redis_key(key)
            return self._client.exists(full_key) > 0

        except RedisError as e:
            logger.warning(f"⚠️  Redis exists error: {e}")
            self._stats["errors"] += 1
            return False

    def clear(self) -> None:
        """Clear all cache entries with our prefix"""
        if not self._available or not self._client:
            return

        try:
            # Delete all keys with our prefix
            pattern = f"{self.prefix}*"
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
                logger.info(f"🧹 Cleared {len(keys)} Redis cache entries")

        except RedisError as e:
            logger.warning(f"⚠️  Redis clear error: {e}")
            self._stats["errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache stats (hits, misses, size, etc.)
        """
        total_ops = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_ops * 100) if total_ops > 0 else 0

        stats = {
            **self._stats,
            "available": self._available,
            "hit_rate": round(hit_rate, 2),
        }

        # Add Redis info if available
        if self._available and self._client:
            try:
                info = self._client.info("memory")
                stats["used_memory_mb"] = info.get("used_memory", 0) / (1024 * 1024)
                stats["used_memory_peak_mb"] = info.get("used_memory_peak", 0) / (1024 * 1024)

                # Count keys with our prefix
                pattern = f"{self.prefix}*"
                stats["total_entries"] = len(self._client.keys(pattern))

            except RedisError as e:
                logger.debug(f"Redis info error: {e}")

        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters"""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

    def _make_redis_key(self, key: str) -> str:
        """
        Generate Redis key with prefix

        Args:
            key: Cache key

        Returns:
            Prefixed Redis key
        """
        return f"{self.prefix}{key}"

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """
        Generate cache key from arguments

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hashed cache key
        """
        # Combine args and kwargs into a single string
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())  # Sort for consistency
        }

        key_str = json.dumps(key_data, sort_keys=True, default=str)

        # Hash for consistent key length
        return hashlib.sha256(key_str.encode()).hexdigest()

    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self._available

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to Redis

        Returns:
            True if reconnected successfully
        """
        if REDIS_AVAILABLE:
            return self._connect()
        return False
