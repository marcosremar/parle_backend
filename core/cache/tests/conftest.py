"""
Pytest fixtures for cache tests
"""

import pytest
from src.core.cache import MemoryCache, SemanticCache, CacheManager


@pytest.fixture
def memory_cache():
    """Create a MemoryCache instance for testing"""
    return MemoryCache(max_size_mb=10, default_ttl=60)


@pytest.fixture
def semantic_cache():
    """Create a SemanticCache instance for testing (semantic disabled)"""
    return SemanticCache(
        similarity_threshold=0.85,
        max_entries=100,
        enable_semantic=False  # Disable for faster testing
    )


@pytest.fixture
def cache_manager():
    """Create a CacheManager instance for testing"""
    return CacheManager(
        enable_l1=True,
        enable_l2=False,  # Disable Redis for unit tests
        l1_size_mb=10,
        l1_ttl=60,
    )
