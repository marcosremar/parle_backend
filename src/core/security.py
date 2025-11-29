"""
Security utilities for API endpoints
Includes rate limiting, input validation, and sanitization
"""

from functools import wraps
from typing import Callable, Any, Dict, Optional, List
from fastapi import Request, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger
from src.core.config import get_config

config = get_config()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove sensitive data from logging
    
    Args:
        data: Dictionary to sanitize
        
    Returns:
        Sanitized dictionary with sensitive values redacted
    """
    sensitive_keys = ['api_key', 'password', 'token', 'secret', 'auth', 'authorization', 'jwt']
    sanitized = data.copy()
    
    for key in sensitive_keys:
        if key.lower() in [k.lower() for k in sanitized.keys()]:
            # Find the actual key (case-insensitive)
            actual_key = next((k for k in sanitized.keys() if k.lower() == key.lower()), None)
            if actual_key:
                sanitized[actual_key] = "***REDACTED***"
    
    # Also check nested dictionaries
    for key, value in sanitized.items():
        if isinstance(value, dict):
            sanitized[key] = sanitize_for_logging(value)
        elif isinstance(value, str) and len(value) > 100:
            # Truncate long strings that might contain sensitive data
            sanitized[key] = value[:50] + "...[truncated]"
    
    return sanitized


def validate_upload_size(content_length: Optional[int], max_size_mb: Optional[int] = None) -> None:
    """
    Validate upload size
    
    Args:
        content_length: Content length in bytes
        max_size_mb: Maximum size in MB (defaults to config value)
        
    Raises:
        HTTPException: If file size exceeds limit
    """
    if content_length is None:
        return  # Can't validate without content length
    
    max_size = (max_size_mb or config.server.max_upload_size_mb) * 1024 * 1024
    
    if content_length > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({content_length / 1024 / 1024:.2f}MB) exceeds maximum allowed size ({max_size_mb or config.server.max_upload_size_mb}MB)"
        )


def validate_content_type(content_type: Optional[str], allowed_types: List[str]) -> None:
    """
    Validate content type
    
    Args:
        content_type: Content type header value
        allowed_types: List of allowed content types (e.g., ['audio/', 'application/json'])
        
    Raises:
        HTTPException: If content type is not allowed
    """
    if not content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content-Type header is required"
        )
    
    if not any(content_type.startswith(allowed) for allowed in allowed_types):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Content-Type '{content_type}' not allowed. Allowed types: {', '.join(allowed_types)}"
        )


def rate_limit_decorator(limit: str = "10/minute"):
    """
    Decorator for rate limiting endpoints
    
    Args:
        limit: Rate limit string (e.g., "10/minute", "100/hour")
        
    Usage:
        @rate_limit_decorator("5/minute")
        async def my_endpoint(request: Request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find Request object in args or kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get('request')
            
            if request:
                # Get app state for limiter
                app = request.app
                if hasattr(app.state, 'limiter'):
                    limiter = app.state.limiter
                    # Apply rate limit using limiter
                    try:
                        # Use limiter's limit method
                        limited_func = limiter.limit(limit)(func)
                        return await limited_func(*args, **kwargs)
                    except RateLimitExceeded:
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Rate limit exceeded: {limit}"
                        )
            
            # No request or limiter, execute normally
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def safe_log_error(message: str, error: Exception, context: Optional[Dict] = None, request: Optional[Request] = None):
    """
    Log error with sanitized context
    
    Args:
        message: Error message
        error: Exception object
        context: Additional context (will be sanitized)
        request: Request object (will extract safe info)
    """
    safe_context = sanitize_for_logging(context or {})
    
    # Add request info if available
    if request:
        safe_context.update({
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else None,
        })
    
    logger.error(
        f"{message}: {type(error).__name__}",
        exc_info=error,
        extra=safe_context
    )
