"""
Integrated Security Middleware combining all security components.

This middleware provides a single point to enable all security features:
- JWT authentication
- Rate limiting
- Input sanitization
- CSRF protection
- Security audit logging

Author: Ultravox Team
Version: 1.0.0
"""

import time
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .audit_logger import AuditEventType, AuditSeverity, SecurityAuditLogger
from .input_sanitizer import InputSanitizer
from .jwt_manager import JWTManager, UserRole
from .rate_limiter import AdvancedRateLimiter, RateLimitScope


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Integrated security middleware.

    Example:
        >>> from fastapi import FastAPI
        >>> from src.core.security import SecurityMiddleware, SecurityConfig
        >>>
        >>> app = FastAPI()
        >>> config = SecurityConfig()
        >>> app.add_middleware(SecurityMiddleware, config=config)
    """

    def __init__(
        self,
        app,
        jwt_manager: Optional[JWTManager] = None,
        rate_limiter: Optional[AdvancedRateLimiter] = None,
        input_sanitizer: Optional[InputSanitizer] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
        public_paths: Optional[list] = None,
        require_auth: bool = True,
        require_https: bool = False,
    ):
        """
        Initialize security middleware.

        Args:
            app: FastAPI application
            jwt_manager: JWT manager instance
            rate_limiter: Rate limiter instance
            input_sanitizer: Input sanitizer instance
            audit_logger: Audit logger instance
            public_paths: List of paths that don't require authentication
            require_auth: Require authentication by default
            require_https: Require HTTPS for all requests
        """
        super().__init__(app)
        self.jwt_manager = jwt_manager
        self.rate_limiter = rate_limiter
        self.input_sanitizer = input_sanitizer
        self.audit_logger = audit_logger
        self.public_paths = public_paths or ["/health", "/docs", "/openapi.json"]
        self.require_auth = require_auth
        self.require_https = require_https

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request through security layers.

        Security checks in order:
        1. HTTPS enforcement
        2. Rate limiting
        3. Input sanitization
        4. Authentication
        5. CSRF validation
        6. Audit logging
        """
        start_time = time.time()
        user_id = None
        user_role = None
        trace_id = request.headers.get("X-Trace-ID", "")

        try:
            # 1. HTTPS enforcement
            if self.require_https and request.url.scheme != "https":
                if self.audit_logger:
                    self.audit_logger.log_event(
                        event_type=AuditEventType.SECURITY_CSRF_VIOLATION,
                        action="Request over HTTP rejected",
                        severity=AuditSeverity.WARNING,
                        ip_address=self._get_client_ip(request),
                        trace_id=trace_id,
                    )
                return JSONResponse(
                    status_code=403,
                    content={"error": "HTTPS required"},
                )

            # 2. Rate limiting
            if self.rate_limiter:
                client_ip = self._get_client_ip(request)
                rate_limit_result = self.rate_limiter.check_limit(
                    identifier=client_ip,
                    scope=RateLimitScope.IP,
                    endpoint=request.url.path,
                )

                if not rate_limit_result.allowed:
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            event_type=AuditEventType.SECURITY_RATE_LIMITED,
                            action=f"Rate limit exceeded for IP {client_ip}",
                            severity=AuditSeverity.WARNING,
                            ip_address=client_ip,
                            resource=request.url.path,
                            trace_id=trace_id,
                        )

                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "retry_after": rate_limit_result.retry_after,
                        },
                        headers={
                            "Retry-After": str(rate_limit_result.retry_after or 60),
                            "X-RateLimit-Limit": str(
                                self.rate_limiter.config.requests_per_minute
                            ),
                            "X-RateLimit-Remaining": str(rate_limit_result.remaining),
                            "X-RateLimit-Reset": str(int(rate_limit_result.reset_at)),
                        },
                    )

                # Apply throttle delay if configured
                if rate_limit_result.throttle_delay_ms:
                    import asyncio

                    await asyncio.sleep(rate_limit_result.throttle_delay_ms / 1000)

            # 3. Input sanitization
            if self.input_sanitizer:
                # Check for XSS in query parameters
                for key, value in request.query_params.items():
                    if self.input_sanitizer.detect_xss(value):
                        if self.audit_logger:
                            self.audit_logger.log_event(
                                event_type=AuditEventType.SECURITY_XSS_DETECTED,
                                action=f"XSS detected in query parameter: {key}",
                                severity=AuditSeverity.CRITICAL,
                                ip_address=self._get_client_ip(request),
                                resource=request.url.path,
                                success=False,
                                trace_id=trace_id,
                            )

                        return JSONResponse(
                            status_code=400,
                            content={"error": "Invalid input detected"},
                        )

                    # Check for SQL injection
                    if self.input_sanitizer.detect_sql_injection(value):
                        if self.audit_logger:
                            self.audit_logger.log_event(
                                event_type=AuditEventType.SECURITY_SQL_INJECTION_DETECTED,
                                action=f"SQL injection detected in query parameter: {key}",
                                severity=AuditSeverity.CRITICAL,
                                ip_address=self._get_client_ip(request),
                                resource=request.url.path,
                                success=False,
                                trace_id=trace_id,
                            )

                        return JSONResponse(
                            status_code=400,
                            content={"error": "Invalid input detected"},
                        )

            # 4. Authentication
            is_public = any(
                request.url.path.startswith(path) for path in self.public_paths
            )

            if self.require_auth and not is_public and self.jwt_manager:
                auth_header = request.headers.get("Authorization", "")

                if not auth_header.startswith("Bearer "):
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            event_type=AuditEventType.AUTHZ_ACCESS_DENIED,
                            action="Missing authentication token",
                            severity=AuditSeverity.WARNING,
                            ip_address=self._get_client_ip(request),
                            resource=request.url.path,
                            success=False,
                            trace_id=trace_id,
                        )

                    return JSONResponse(
                        status_code=401,
                        content={"error": "Authentication required"},
                    )

                token = auth_header.split(" ")[1]

                try:
                    # Verify token
                    payload = self.jwt_manager.verify_access_token(token)
                    user_id = payload.get("sub")
                    user_role = payload.get("role")

                    # Attach to request state
                    request.state.user_id = user_id
                    request.state.user_role = user_role
                    request.state.token_payload = payload

                    # Log successful auth
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            event_type=AuditEventType.AUTHZ_ACCESS_GRANTED,
                            action="Request authenticated successfully",
                            severity=AuditSeverity.INFO,
                            user_id=user_id,
                            user_role=user_role,
                            ip_address=self._get_client_ip(request),
                            resource=request.url.path,
                            trace_id=trace_id,
                        )

                except Exception as e:
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            event_type=AuditEventType.AUTHZ_ACCESS_DENIED,
                            action="Invalid authentication token",
                            severity=AuditSeverity.WARNING,
                            ip_address=self._get_client_ip(request),
                            resource=request.url.path,
                            success=False,
                            error_message=str(e),
                            trace_id=trace_id,
                        )

                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid or expired token"},
                    )

            # 5. CSRF validation for state-changing methods
            if (
                self.input_sanitizer
                and request.method in ["POST", "PUT", "DELETE", "PATCH"]
                and not is_public
            ):
                csrf_token = request.headers.get("X-CSRF-Token", "")

                if user_id and not self.input_sanitizer.validate_csrf_token(
                    csrf_token, user_id
                ):
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            event_type=AuditEventType.SECURITY_CSRF_VIOLATION,
                            action="CSRF token validation failed",
                            severity=AuditSeverity.CRITICAL,
                            user_id=user_id,
                            ip_address=self._get_client_ip(request),
                            resource=request.url.path,
                            success=False,
                            trace_id=trace_id,
                        )

                    return JSONResponse(
                        status_code=403,
                        content={"error": "CSRF validation failed"},
                    )

            # Process request
            response = await call_next(request)

            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            if self.require_https:
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )

            # Log successful request
            if self.audit_logger and response.status_code < 400:
                duration_ms = (time.time() - start_time) * 1000
                self.audit_logger.log_event(
                    event_type=AuditEventType.DATA_READ
                    if request.method == "GET"
                    else AuditEventType.DATA_UPDATED,
                    action=f"{request.method} {request.url.path}",
                    severity=AuditSeverity.INFO,
                    user_id=user_id,
                    user_role=user_role,
                    ip_address=self._get_client_ip(request),
                    resource=request.url.path,
                    success=True,
                    trace_id=trace_id,
                    metadata={
                        "duration_ms": duration_ms,
                        "status_code": response.status_code,
                    },
                )

            return response

        except Exception as e:
            # Log error
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type=AuditEventType.SERVICE_ERROR,
                    action=f"Request processing error: {str(e)}",
                    severity=AuditSeverity.ERROR,
                    user_id=user_id,
                    ip_address=self._get_client_ip(request),
                    resource=request.url.path,
                    success=False,
                    error_message=str(e),
                    trace_id=trace_id,
                )

            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check X-Forwarded-For header (from load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP in chain
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to client host
        if request.client:
            return request.client.host

        return "unknown"


def create_security_middleware(
    jwt_config: Optional[dict] = None,
    rate_limit_config: Optional[dict] = None,
    sanitization_config: Optional[dict] = None,
    audit_config: Optional[dict] = None,
    **kwargs,
) -> SecurityMiddleware:
    """
    Factory function to create fully configured security middleware.

    Example:
        >>> middleware = create_security_middleware(
        ...     jwt_config={"secret_key": "my-secret", "algorithm": "HS256"},
        ...     rate_limit_config={"requests_per_minute": 60},
        ...     require_auth=True
        ... )
    """
    from .audit_logger import AuditLogConfig
    from .input_sanitizer import SanitizationConfig
    from .jwt_manager import JWTConfig
    from .rate_limiter import RateLimitConfig

    # Initialize components
    jwt_manager = None
    if jwt_config:
        jwt_manager = JWTManager(JWTConfig(**jwt_config))

    rate_limiter = None
    if rate_limit_config:
        rate_limiter = AdvancedRateLimiter(RateLimitConfig(**rate_limit_config))

    input_sanitizer = None
    if sanitization_config:
        input_sanitizer = InputSanitizer(SanitizationConfig(**sanitization_config))

    audit_logger = None
    if audit_config:
        audit_logger = SecurityAuditLogger(AuditLogConfig(**audit_config))

    return SecurityMiddleware(
        app=None,  # Will be set by FastAPI
        jwt_manager=jwt_manager,
        rate_limiter=rate_limiter,
        input_sanitizer=input_sanitizer,
        audit_logger=audit_logger,
        **kwargs,
    )
