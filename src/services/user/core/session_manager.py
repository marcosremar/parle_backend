"""
Session Manager - Session lifecycle management
"""
from typing import Dict, List
from datetime import datetime, timedelta
import secrets

from ..storage import sessions_db


def create_session(user_id: str, expires_in_hours: int = 24) -> Dict:
    """
    Create a new user session

    Args:
        user_id: User ID for the session
        expires_in_hours: Session expiration time (default 24h)

    Returns:
        Session dictionary
    """
    session_id = secrets.token_urlsafe(32)
    session = {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=expires_in_hours)).isoformat(),
        "is_active": True
    }
    sessions_db[session_id] = session
    return session


def get_session(session_id: str) -> Dict:
    """Get session by ID"""
    return sessions_db.get(session_id)


def is_session_valid(session_id: str) -> bool:
    """
    Check if session is valid (exists, active, not expired)

    Args:
        session_id: Session ID to check

    Returns:
        True if session is valid
    """
    session = sessions_db.get(session_id)
    if not session or not session.get("is_active"):
        return False

    expires_at = datetime.fromisoformat(session["expires_at"])
    return expires_at > datetime.now()


def invalidate_session(session_id: str) -> bool:
    """
    Invalidate a specific session

    Args:
        session_id: Session ID to invalidate

    Returns:
        True if session was invalidated
    """
    session = sessions_db.get(session_id)
    if session:
        session["is_active"] = False
        return True
    return False


def invalidate_user_sessions(user_id: str) -> int:
    """
    Invalidate all active sessions for a user

    Args:
        user_id: User ID

    Returns:
        Number of sessions invalidated
    """
    count = 0
    for session in sessions_db.values():
        if session["user_id"] == user_id and session["is_active"]:
            session["is_active"] = False
            count += 1
    return count


def list_user_sessions(user_id: str, active_only: bool = False) -> List[Dict]:
    """
    List all sessions for a user

    Args:
        user_id: User ID
        active_only: Only return active sessions

    Returns:
        List of session dictionaries
    """
    sessions = []
    for session in sessions_db.values():
        if session["user_id"] == user_id:
            if active_only and not session["is_active"]:
                continue
            sessions.append(session)
    return sessions


def cleanup_expired_sessions() -> int:
    """
    Clean up expired sessions

    Returns:
        Number of sessions cleaned up
    """
    now = datetime.now()
    count = 0
    expired_session_ids = []

    for session_id, session in sessions_db.items():
        if session["is_active"]:
            expires_at = datetime.fromisoformat(session["expires_at"])
            if expires_at <= now:
                expired_session_ids.append(session_id)
                count += 1

    # Remove expired sessions
    for session_id in expired_session_ids:
        sessions_db[session_id]["is_active"] = False

    return count
