"""
Shared dependencies for service endpoints.

Extracted from services.py to avoid duplication.
"""

from fastapi import HTTPException
from typing import Any

# ============================================================================
# Manager Dependency Injection
# ============================================================================

_manager = None

def set_manager(manager) -> Any:
    """Set the global manager instance."""
    global _manager
    _manager = manager

def get_manager() -> Any:
    """Dependency to get the manager instance."""
    if _manager is None:
        raise HTTPException(status_code=500, detail="Service manager not initialized")
    return _manager

# ============================================================================
# Input Validation - Security
# ============================================================================

def validate_service_id(service_id: str) -> str:
    """
    Validate service_id input to prevent path traversal and injection attacks.

    Args:
        service_id: Service identifier to validate

    Returns:
        Validated service_id

    Raises:
        HTTPException: If service_id contains malicious patterns
    """
    import re

    # Check for null or empty
    if not service_id or not service_id.strip():
        raise HTTPException(status_code=400, detail="service_id cannot be empty")

    # Check for path traversal patterns
    if ".." in service_id:
        raise HTTPException(status_code=400, detail="Invalid service_id: path traversal detected")

    # Check for absolute paths
    if service_id.startswith("/") or service_id.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid service_id: absolute paths not allowed")

    # Only allow alphanumeric, underscore, hyphen (standard service naming)
    if not re.match(r'^[a-zA-Z0-9_\-]+$', service_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid service_id: only alphanumeric characters, underscores, and hyphens allowed"
        )

    # Length check (reasonable limit)
    if len(service_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid service_id: too long (max 64 characters)")

    return service_id
