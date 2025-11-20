"""
Database Configuration for Session Service

This file configures how Session Service uses the database.
Each service configures its own database usage independently.
"""

from src.services.database.database_client import LoadStrategy

# Session Service Database Configuration
DATABASE_CONFIG = {
    "load_strategy": LoadStrategy.LAZY  # Cold start - load on demand
}

# Configuration rationale:
# - LAZY: Sessions are ephemeral, no need to pre-load
# - Load on demand when user makes request
# - GenericDatabaseClient uses dict + FAISS automatically
