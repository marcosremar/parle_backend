"""
Middleware components for FastAPI applications

Includes:
- Rate Limiting: Prevent API abuse
- Input Validation: Validate and sanitize requests
- Authentication: JWT token validation
- CORS: Cross-origin resource sharing
"""

from .rate_limiting import (
    RateLimitMiddleware,
    RateLimitConfig,
    RateLimiter,
    RateLimitStrategy,
    get_rate_limiter
)

from .input_validation import (
    InputValidationMiddleware,
    ValidationConfig,
    InputValidator,
    ValidationLevel,
    get_input_validator
)

from .authentication import (
    AuthenticationMiddleware,
    AuthConfig,
    JWTAuth,
    UserRole,
    get_jwt_auth
)

__all__ = [
    # Rate Limiting
    "RateLimitMiddleware",
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitStrategy",
    "get_rate_limiter",

    # Input Validation
    "InputValidationMiddleware",
    "ValidationConfig",
    "InputValidator",
    "ValidationLevel",
    "get_input_validator",

    # Authentication
    "AuthenticationMiddleware",
    "AuthConfig",
    "JWTAuth",
    "UserRole",
    "get_jwt_auth",
]
