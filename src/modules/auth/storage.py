"""
In-memory Storage
Replace with database in production
"""

# Global stores
users_db: dict[str, dict] = {}
sessions_db: dict[str, dict] = {}
api_keys_db: dict[str, dict] = {}


def clear_all():
    """Clear all databases (for testing)"""
    users_db.clear()
    sessions_db.clear()
    api_keys_db.clear()


def get_stats() -> dict:
    """Get storage statistics"""
    return {
        "total_users": len(users_db),
        "total_sessions": len(sessions_db),
        "active_sessions": sum(1 for s in sessions_db.values() if s.get("is_active")),
        "total_api_keys": len(api_keys_db),
        "active_api_keys": sum(1 for k in api_keys_db.values() if k.get("is_active")),
    }
