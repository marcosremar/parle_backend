"""
Security module for Ultravox Pipeline.

This module provides enhanced security features:
- JWT Manager: Improved token management with refresh tokens, blacklist, RBAC
- Rate Limiter: Advanced rate limiting with Redis support
- Input Sanitizer: XSS, SQL injection, CSRF protection
- Secrets Manager: HashiCorp Vault / AWS Secrets Manager integration
- Security Audit Logger: Centralized audit trail

Author: Ultravox Team
Version: 1.0.0
"""

from .jwt_manager import JWTManager, JWTConfig, TokenPair, TokenType, UserRole, Permission
from .rate_limiter import (
    AdvancedRateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitBackend,
    RateLimitScope,
)
from .input_sanitizer import InputSanitizer, SanitizationConfig, SanitizationLevel
from .secrets_manager import SecretsManager, SecretsBackend, SecretConfig
from .audit_logger import SecurityAuditLogger, AuditEvent, AuditEventType, AuditSeverity
from .security_middleware import SecurityMiddleware, create_security_middleware

__all__ = [
    # JWT Management
    "JWTManager",
    "JWTConfig",
    "TokenPair",
    "TokenType",
    "UserRole",
    "Permission",
    # Rate Limiting
    "AdvancedRateLimiter",
    "RateLimitConfig",
    "RateLimitStrategy",
    "RateLimitBackend",
    "RateLimitScope",
    # Input Sanitization
    "InputSanitizer",
    "SanitizationConfig",
    "SanitizationLevel",
    # Secrets Management
    "SecretsManager",
    "SecretsBackend",
    "SecretConfig",
    # Audit Logging
    "SecurityAuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    # Middleware
    "SecurityMiddleware",
    "create_security_middleware",
]
