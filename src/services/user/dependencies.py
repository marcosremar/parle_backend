"""
FastAPI Dependencies for User Service
Handles authentication and authorization
"""
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict
from datetime import datetime

from .storage import users_db, sessions_db, api_keys_db

# Security
security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    Get current user from token (session token or API key)

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        User dictionary if authenticated

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials

    # Check if it's a session token
    if token in sessions_db:
        session = sessions_db[token]
        if session["is_active"]:
            expires_at = datetime.fromisoformat(session["expires_at"])
            if expires_at > datetime.now():
                user_id = session["user_id"]
                if user_id in users_db:
                    return users_db[user_id]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired"
        )

    # Check if it's an API key
    if token in api_keys_db:
        api_key = api_keys_db[token]
        if api_key["is_active"]:
            user_id = api_key["user_id"]
            if user_id in users_db:
                return users_db[user_id]

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials"
    )


def require_admin(current_user: Dict = Depends(get_current_user)) -> Dict:
    """
    Require user to have admin role

    Args:
        current_user: Current authenticated user

    Returns:
        User dictionary if user is admin

    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user
