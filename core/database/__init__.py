"""
Database Optimization Core Module
Provides connection pooling, query optimization, and performance monitoring
"""
from .query_optimizer import (
    QueryOptimizer,
    QueryMetrics,
    QueryStats,
    get_query_optimizer,
    track_query,
)

try:
    from .connection_pool import (
        ConnectionPoolManager,
        get_pool_manager,
    )
    CONNECTION_POOL_AVAILABLE = True
except ImportError:
    CONNECTION_POOL_AVAILABLE = False

__all__ = [
    "QueryOptimizer",
    "QueryMetrics",
    "QueryStats",
    "get_query_optimizer",
    "track_query",
]

if CONNECTION_POOL_AVAILABLE:
    __all__.extend([
        "ConnectionPoolManager",
        "get_pool_manager",
    ])
