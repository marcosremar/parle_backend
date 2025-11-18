"""
User Manager - CRUD operations for users
"""
from fastapi import HTTPException, status
from typing import Dict, List, Optional
from datetime import datetime
import secrets

from ..storage import users_db
from .auth import hash_password, verify_password


def create_user(username: str, email: str, password: str, full_name: Optional[str] = None, is_admin: bool = False) -> Dict:
    """
    Create a new user

    Args:
        username: Unique username
        email: User email
        password: Plain text password (will be hashed)
        full_name: Optional full name
        is_admin: Whether user is admin

    Returns:
        Created user dictionary

    Raises:
        HTTPException: If username or email already exists
    """
    # Check if username or email already exists
    for user in users_db.values():
        if user["username"] == username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        if user["email"] == email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

    # Create new user
    user_id = f"user_{secrets.token_hex(8)}"
    user = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "full_name": full_name,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "preferences": {},
        "is_active": True,
        "is_admin": is_admin
    }

    users_db[user_id] = user
    return user


def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by ID"""
    return users_db.get(user_id)


def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username"""
    for user in users_db.values():
        if user["username"] == username:
            return user
    return None


def update_user(user_id: str, email: Optional[str] = None, full_name: Optional[str] = None,
                preferences: Optional[Dict] = None) -> Dict:
    """
    Update user profile

    Args:
        user_id: User ID to update
        email: New email (optional)
        full_name: New full name (optional)
        preferences: Preferences to merge (optional)

    Returns:
        Updated user dictionary

    Raises:
        HTTPException: If user not found
    """
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if email:
        user["email"] = email
    if full_name:
        user["full_name"] = full_name
    if preferences:
        user["preferences"].update(preferences)

    return user


def update_last_login(user_id: str) -> None:
    """Update user's last login timestamp"""
    user = users_db.get(user_id)
    if user:
        user["last_login"] = datetime.now().isoformat()


def list_all_users() -> List[Dict]:
    """List all users (admin function)"""
    return [
        {
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "is_active": user["is_active"],
            "is_admin": user.get("is_admin", False),
            "created_at": user["created_at"],
            "last_login": user["last_login"]
        }
        for user in users_db.values()
    ]


def change_password(user_id: str, old_password: str, new_password: str) -> bool:
    """
    Change user password

    Args:
        user_id: User ID
        old_password: Current password
        new_password: New password

    Returns:
        True if password changed successfully

    Raises:
        HTTPException: If user not found or old password incorrect
    """
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if not verify_password(old_password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
        )

    user["password_hash"] = hash_password(new_password)
    return True
