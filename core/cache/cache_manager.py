"""
Cache Manager - Multi-layer cache orchestrator

Manages L1 (in-memory), L2 (Redis), and L3 (Database) cache layers.
Implements automatic cache warming and intelligent invalidation.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Callable, List
from datetime import datetime, timedelta
import asyncio

from .memory_cache import MemoryCache
from .redis_adapter import RedisAdapter


logger = logging.getLogger(__name__)


class CacheManager:
    """
    Multi-layer cache manager

    Cache hierarchy:
    L1 (Memory) → L2 (Redis) → L3 (Database) → API Call

    Features:
    - Automatic cache warming
    - TTL per layer
    - Cache invalidation strategies
    - Hit rate tracking
    - Semantic caching for LLM
    """

    def __init__(
        self,
        enable_l1: bool = True,
        enable_l2: bool = True,
        l1_size_mb: int = 100,
        l1_ttl: int = 3600,
        l2_size_mb: int = 1024,
        l2_ttl: int = 7200,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_db: int = 0,
    ):
        """
        Initialize cache manager

        Args:
            enable_l1: Enable L1 (in-memory) cache
            enable_l2: Enable L2 (Redis) cache
            l1_size_mb: L1 max size in MB
            l1_ttl: L1 default TTL in seconds
            l2_size_mb: L2 max size in MB
            l2_ttl: L2 default TTL in seconds
            redis_host: Redis host
            redis_port: Redis port
            redis_password: Redis password (optional)
            redis_db: Redis database number
        """
        self.enable_l1 = enable_l1
        self.enable_l2 = enable_l2

        # Initialize L1 (in-memory)
        self.l1: Optional[MemoryCache] = None
        if enable_l1:
            self.l1 = MemoryCache(max_size_mb=l1_size_mb, default_ttl=l1_ttl)
            logger.info(f"✅ L1 cache enabled (in-memory, {l1_size_mb}MB, TTL={l1_ttl}s)")

        # Initialize L2 (Redis)
        self.l2: Optional[RedisAdapter] = None
        if enable_l2:
            self.l2 = RedisAdapter(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                default_ttl=l2_ttl,
                max_size_mb=l2_size_mb,
            )
            if self.l2.is_available():
                logger.info(f"✅ L2 cache enabled (Redis, {l2_size_mb}MB, TTL={l2_ttl}s)")
            else:
                logger.warning("⚠️  L2 cache unavailable (Redis not connected)")
                self.l2 = None  # Disable if not available

        # Statistics
        self._stats = {
            "total_requests": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,  # Database hits (future)
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
        }

    async def get(
        self,
        key: str,
        fetch_fn: Optional[Callable] = None,
        ttl_l1: Optional[int] = None,
        ttl_l2: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Get value from cache (L1 → L2 → fetch_fn)

        Args:
            key: Cache key
            fetch_fn: Optional async function to fetch value if cache miss
            ttl_l1: Optional L1 TTL override
            ttl_l2: Optional L2 TTL override

        Returns:
            Cached value or result from fetch_fn or None
        """
        self._stats["total_requests"] += 1

        # Try L1 (in-memory) first
        if self.l1:
            value = self.l1.get(key)
            if value is not None:
                self._stats["l1_hits"] += 1
                logger.debug(f"✅ L1 cache hit: {key[:32]}...")
                return value

        # Try L2 (Redis)
        if self.l2:
            value = self.l2.get(key)
            if value is not None:
                self._stats["l2_hits"] += 1
                logger.debug(f"✅ L2 cache hit: {key[:32]}...")

                # Populate L1 with L2 value (cache warming)
                if self.l1:
                    self.l1.set(key, value, ttl=ttl_l1)

                return value

        # Cache miss - fetch value if fetch_fn provided
        if fetch_fn:
            logger.debug(f"❌ Cache miss: {key[:32]}... Fetching...")
            self._stats["misses"] += 1

            try:
                # Fetch value (async or sync)
                if asyncio.iscoroutinefunction(fetch_fn):
                    value = await fetch_fn()
                else:
                    value = fetch_fn()

                # Cache in all layers
                if value is not None:
                    await self.set(key, value, ttl_l1=ttl_l1, ttl_l2=ttl_l2)

                return value

            except Exception as e:
                logger.error(f"❌ Cache fetch error: {e}")
                return None

        # No fetch function - return None
        self._stats["misses"] += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_l1: Optional[int] = None,
        ttl_l2: Optional[int] = None,
    ) -> bool:
        """
        Set value in all cache layers

        Args:
            key: Cache key
            value: Value to cache
            ttl_l1: Optional L1 TTL override
            ttl_l2: Optional L2 TTL override

        Returns:
            True if successfully cached in at least one layer
        """
        self._stats["sets"] += 1
        success = False

        # Set in L1
        if self.l1:
            if self.l1.set(key, value, ttl=ttl_l1):
                success = True
                logger.debug(f"✅ L1 cached: {key[:32]}...")

        # Set in L2
        if self.l2:
            if self.l2.set(key, value, ttl=ttl_l2):
                success = True
                logger.debug(f"✅ L2 cached: {key[:32]}...")

        return success

    async def delete(self, key: str) -> bool:
        """
        Delete value from all cache layers

        Args:
            key: Cache key

        Returns:
            True if deleted from at least one layer
        """
        self._stats["invalidations"] += 1
        success = False

        # Delete from L1
        if self.l1:
            if self.l1.delete(key):
                success = True

        # Delete from L2
        if self.l2:
            if self.l2.delete(key):
                success = True

        logger.debug(f"🗑️  Invalidated: {key[:32]}...")
        return success

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in any cache layer

        Args:
            key: Cache key

        Returns:
            True if key exists in L1 or L2
        """
        # Check L1
        if self.l1 and self.l1.exists(key):
            return True

        # Check L2
        if self.l2 and self.l2.exists(key):
            return True

        return False

    async def clear(self) -> None:
        """Clear all cache layers"""
        if self.l1:
            self.l1.clear()
            logger.info("🧹 L1 cache cleared")

        if self.l2:
            self.l2.clear()
            logger.info("🧹 L2 cache cleared")

    async def warm_cache(self, entries: List[tuple]) -> int:
        """
        Warm cache with pre-computed entries

        Args:
            entries: List of (key, value, ttl_l1, ttl_l2) tuples

        Returns:
            Number of entries successfully cached
        """
        count = 0
        for entry in entries:
            key, value = entry[0], entry[1]
            ttl_l1 = entry[2] if len(entry) > 2 else None
            ttl_l2 = entry[3] if len(entry) > 3 else None

            if await self.set(key, value, ttl_l1=ttl_l1, ttl_l2=ttl_l2):
                count += 1

        logger.info(f"🔥 Cache warmed: {count}/{len(entries)} entries")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache stats from all layers
        """
        total_ops = self._stats["total_requests"]
        l1_hit_rate = (self._stats["l1_hits"] / total_ops * 100) if total_ops > 0 else 0
        l2_hit_rate = (self._stats["l2_hits"] / total_ops * 100) if total_ops > 0 else 0
        total_hit_rate = ((self._stats["l1_hits"] + self._stats["l2_hits"]) / total_ops * 100) if total_ops > 0 else 0

        stats = {
            "overview": {
                **self._stats,
                "l1_hit_rate": round(l1_hit_rate, 2),
                "l2_hit_rate": round(l2_hit_rate, 2),
                "total_hit_rate": round(total_hit_rate, 2),
            }
        }

        # Add L1 stats
        if self.l1:
            stats["l1"] = self.l1.get_stats()

        # Add L2 stats
        if self.l2:
            stats["l2"] = self.l2.get_stats()

        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters"""
        self._stats = {
            "total_requests": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
        }

        if self.l1:
            self.l1.reset_stats()

        if self.l2:
            self.l2.reset_stats()

    @staticmethod
    def make_cache_key(prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from prefix and arguments

        Args:
            prefix: Key prefix (e.g., "llm", "stt", "tts")
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hashed cache key with prefix
        """
        # Combine args and kwargs into a single string
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())  # Sort for consistency
        }

        key_str = json.dumps(key_data, sort_keys=True, default=str)

        # Hash for consistent key length
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()

        return f"{prefix}:{key_hash}"
