"""
CSRF (Cross-Site Request Forgery) protection for FastAPI
"""

import secrets

from fastapi import HTTPException, Request, status

# CSRF token storage (in production, use Redis or session storage)
_csrf_tokens: dict = {}


def generate_csrf_token() -> str:
    """
    Generate a secure CSRF token.

    Returns:
        Cryptographically secure random token (32 bytes, hex encoded)
    """
    return secrets.token_hex(32)


def validate_csrf_token(request: Request, token: str | None = None) -> bool:
    """
    Validate CSRF token from request.

    For state-changing operations (POST, PUT, DELETE, PATCH), validates
    that the CSRF token matches the one in the session/cookie.

    Args:
        request: FastAPI Request object
        token: CSRF token from request header or form data

    Returns:
        True if token is valid, False otherwise

    Note:
        FastAPI already provides CSRF protection for JSON APIs via
        SameSite cookies and CORS. This is additional protection for
        form-based submissions.
    """
    # For JSON APIs, CSRF is handled by CORS and SameSite cookies
    # Only validate for form submissions
    content_type = request.headers.get("Content-Type", "")

    if "application/json" in content_type:
        # JSON APIs are protected by CORS
        return True

    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        # Form submissions need CSRF token
        if not token:
            # Try to get from header
            token = request.headers.get("X-CSRF-Token")

        if not token:
            return False

        # Get token from session (in production, use proper session storage)
        session_token = _csrf_tokens.get(request.cookies.get("session_id"))

        if not session_token:
            return False

        # Compare tokens
        return secrets.compare_digest(token, session_token)

    # GET, HEAD, OPTIONS don't need CSRF protection
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return True

    # Default: require CSRF for state-changing operations
    return False


def csrf_protect(request: Request, token: str | None = None):
    """
    CSRF protection decorator/middleware helper.

    Validates CSRF token for state-changing operations.

    Args:
        request: FastAPI Request object
        token: CSRF token from request

    Raises:
        HTTPException: 403 if CSRF token is invalid

    Note:
        This is a helper function. In practice, FastAPI's CORS and
        SameSite cookies provide sufficient CSRF protection for JSON APIs.
        Use this only for form-based endpoints.
    """
    if not validate_csrf_token(request, token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing CSRF token"
        )


# CSRF token generator endpoint (for forms)
def get_csrf_token(session_id: str) -> str:
    """
    Get or generate CSRF token for a session.

    Args:
        session_id: Session identifier

    Returns:
        CSRF token for the session
    """
    if session_id not in _csrf_tokens:
        _csrf_tokens[session_id] = generate_csrf_token()

    return _csrf_tokens[session_id]
