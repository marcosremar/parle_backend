"""
Database Configuration for Orchestrator Service

Orchestrator needs fast access to user data for pipeline execution.
Uses eager loading (warm start) to keep active users in memory.
"""

from src.services.database.database_client import LoadStrategy

# Orchestrator Service Database Configuration
DATABASE_CONFIG = {
    "load_strategy": LoadStrategy.EAGER  # Warm start - pre-load active users
}

# Configuration rationale:
# - HYBRID: Needs both structured data (conversations) and semantic search (context)
# - EAGER: Pre-loads active users into memory for instant access
# - Critical for low-latency pipeline execution
# - Orchestrator maintains cache of active users

# Cache settings (used by service)
ACTIVE_USER_CACHE_SIZE = 100       # Max active users in memory
ACTIVE_USER_TTL_MINUTES = 30       # Evict after 30min of inactivity
PRELOAD_ON_STARTUP = True          # Pre-load recent users on service start
