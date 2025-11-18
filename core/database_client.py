"""
Database Client - Compatibility Wrapper

This module re-exports database client classes from their actual location
to maintain backward compatibility with code that imports from src.core.

NOTE: DatabaseType was removed - GenericDatabaseClient uses dict + FAISS automatically.
All public API classes and constants are re-exported for backward compatibility.
"""

# Re-export from actual location
from src.services.database.database_client import (
    UserDatabase,
    LoadStrategy,
    DatabaseClientFactory,
    GLOBAL_DATABASE,
    SYSTEM_DATABASE,
    _search_cache,
    _SEARCH_CACHE_TTL,
)

__all__ = [
    'UserDatabase',
    'LoadStrategy',
    'DatabaseClientFactory',
    'GLOBAL_DATABASE',
    'SYSTEM_DATABASE',
    '_search_cache',
    '_SEARCH_CACHE_TTL',
]
