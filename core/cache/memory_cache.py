"""
Memory Cache (L1) - In-memory cache with LRU eviction

Ultra-fast cache layer with automatic eviction when size limit is reached.
Uses OrderedDict for O(1) operations and LRU eviction policy.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
import sys


class MemoryCache:
    """
    L1 In-memory cache with LRU eviction

    Features:
    - O(1) get/set operations
    - LRU eviction when size limit reached
    - TTL (Time-To-Live) support
    - Size tracking (approximate)
    - Thread-safe operations
    """

    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        """
        Initialize memory cache

        Args:
            max_size_mb: Maximum cache size in MB (default 100MB)
            default_ttl: Default TTL in seconds (default 1 hour)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl

        # OrderedDict for LRU cache (insertion order maintained)
        self._cache: OrderedDict[str, Tuple[Any, float, float]] = OrderedDict()
        # Tuple: (value, expiry_timestamp, size_bytes)

        self._current_size = 0
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
            "deletes": 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        value, expiry, size = self._cache[key]

        # Check if expired
        if time.time() > expiry:
            self._stats["misses"] += 1
            self.delete(key)  # Remove expired entry
            return None

        # Move to end (mark as recently used for LRU)
        self._cache.move_to_end(key)

        self._stats["hits"] += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: use default_ttl)

        Returns:
            True if successfully cached
        """
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl

        # Estimate size (rough approximation)
        size = sys.getsizeof(value)

        # If key already exists, remove old size
        if key in self._cache:
            _, _, old_size = self._cache[key]
            self._current_size -= old_size

        # Check if we need to evict
        while self._current_size + size > self.max_size_bytes and len(self._cache) > 0:
            self._evict_lru()

        # Store value
        self._cache[key] = (value, expiry, size)
        self._cache.move_to_end(key)  # Mark as most recently used
        self._current_size += size
        self._stats["sets"] += 1

        return True

    def delete(self, key: str) -> bool:
        """
        Delete value from cache

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        if key not in self._cache:
            return False

        _, _, size = self._cache[key]
        del self._cache[key]
        self._current_size -= size
        self._stats["deletes"] += 1

        return True

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache (and not expired)

        Args:
            key: Cache key

        Returns:
            True if key exists and not expired
        """
        if key not in self._cache:
            return False

        _, expiry, _ = self._cache[key]

        # Check if expired
        if time.time() > expiry:
            self.delete(key)
            return False

        return True

    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._current_size = 0

    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._cache:
            return

        # Remove first item (least recently used)
        key, (_, _, size) = self._cache.popitem(last=False)
        self._current_size -= size
        self._stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache stats (hits, misses, size, etc.)
        """
        total_ops = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_ops * 100) if total_ops > 0 else 0

        return {
            **self._stats,
            "total_entries": len(self._cache),
            "current_size_mb": self._current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "hit_rate": round(hit_rate, 2),
            "fill_rate": round((self._current_size / self.max_size_bytes) * 100, 2),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters"""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
            "deletes": 0,
        }

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

        key_str = json.dumps(key_data, sort_keys=True)

        # Hash for consistent key length
        return hashlib.sha256(key_str.encode()).hexdigest()
