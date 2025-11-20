"""
API Key Manager - API key lifecycle management
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import secrets

from ..storage import api_keys_db


def create_api_key(user_id: str, name: str, expires_in_days: int = 365) -> Dict:
    """
    Create a new API key for a user

    Args:
        user_id: User ID
        name: Name/description for the API key
        expires_in_days: Expiration time in days (default 365)

    Returns:
        API key dictionary with full key (only shown once)
    """
    api_key = secrets.token_urlsafe(32)
    key_id = f"key_{secrets.token_hex(8)}"

    api_key_data = {
        "key_id": key_id,
        "user_id": user_id,
        "name": name,
        "api_key": api_key,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(days=expires_in_days)).isoformat(),
        "is_active": True,
        "last_used": None
    }

    api_keys_db[api_key] = api_key_data
    return api_key_data


def get_api_key(api_key: str) -> Optional[Dict]:
    """Get API key data by key string"""
    return api_keys_db.get(api_key)


def is_api_key_valid(api_key: str) -> bool:
    """
    Check if API key is valid (exists, active, not expired)

    Args:
        api_key: API key string

    Returns:
        True if API key is valid
    """
    key_data = api_keys_db.get(api_key)
    if not key_data or not key_data.get("is_active"):
        return False

    expires_at = datetime.fromisoformat(key_data["expires_at"])
    if expires_at <= datetime.now():
        return False

    # Update last used timestamp
    key_data["last_used"] = datetime.now().isoformat()
    return True


def revoke_api_key(api_key: str) -> bool:
    """
    Revoke an API key

    Args:
        api_key: API key string to revoke

    Returns:
        True if key was revoked
    """
    key_data = api_keys_db.get(api_key)
    if key_data:
        key_data["is_active"] = False
        return True
    return False


def revoke_api_key_by_id(key_id: str) -> bool:
    """
    Revoke an API key by its ID

    Args:
        key_id: API key ID

    Returns:
        True if key was revoked
    """
    for key_data in api_keys_db.values():
        if key_data["key_id"] == key_id:
            key_data["is_active"] = False
            return True
    return False


def list_user_api_keys(user_id: str, active_only: bool = False) -> List[Dict]:
    """
    List all API keys for a user

    Args:
        user_id: User ID
        active_only: Only return active keys

    Returns:
        List of API key dictionaries (without full key)
    """
    keys = []
    for api_key, key_data in api_keys_db.items():
        if key_data["user_id"] == user_id:
            if active_only and not key_data["is_active"]:
                continue

            # Return key info without full API key (security)
            keys.append({
                "key_id": key_data["key_id"],
                "name": key_data["name"],
                "key_preview": f"{api_key[:8]}...{api_key[-4:]}",
                "created_at": key_data["created_at"],
                "expires_at": key_data["expires_at"],
                "is_active": key_data["is_active"],
                "last_used": key_data.get("last_used")
            })
    return keys


def cleanup_expired_api_keys() -> int:
    """
    Clean up expired API keys

    Returns:
        Number of keys cleaned up
    """
    now = datetime.now()
    count = 0

    for key_data in api_keys_db.values():
        if key_data["is_active"]:
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if expires_at <= now:
                key_data["is_active"] = False
                count += 1

    return count
