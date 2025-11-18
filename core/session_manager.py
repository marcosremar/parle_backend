#!/usr/bin/env python3
"""
Session Manager - COMPATIBILITY WRAPPER (DEPRECATED)

⚠️  DEPRECATION NOTICE:
    This module is deprecated and will be removed in v6.0.
    Import from src.core.managers.session_manager instead.

    OLD: from src.core.session_manager import session_manager
    NEW: from src.core.managers.session_manager import session_manager

Migration Path:
    All functionality has been moved to src.core.managers.session_manager.
    This file only re-exports for backward compatibility.

    Please update your imports to use the new location:
    - Session (dataclass)
    - Message (dataclass)
    - SessionManager (class)
    - session_manager (singleton instance)
    - Helper functions: create_session, get_session, etc.
"""

import warnings

# Re-export all from main implementation
from src.core.managers.session_manager import (
    # Core classes
    SessionManager,
    Session,
    Message,

    # Singleton instance
    session_manager,

    # Helper functions
    create_session,
    get_session,
    get_or_create_session,
    add_interaction,
    get_context,
)

# Emit deprecation warning on import
warnings.warn(
    "Module 'src.core.session_manager' is deprecated. "
    "Import from 'src.core.managers.session_manager' instead. "
    "This compatibility wrapper will be removed in v6.0.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    # Core classes
    "SessionManager",
    "Session",
    "Message",

    # Singleton instance
    "session_manager",

    # Helper functions
    "create_session",
    "get_session",
    "get_or_create_session",
    "add_interaction",
    "get_context",
]
