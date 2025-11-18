#!/usr/bin/env python3
"""
Custom Exceptions for Ultravox Pipeline
Hierarchical exception system with structured error context
"""

import json
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""

    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    PROCESSING = "processing"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Detailed error context for debugging and tracking"""

    error_id: str = field(default_factory=lambda: f"err_{int(time.time() * 1000)}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    system_info: Optional[Dict[str, Any]] = None


class UltravoxError(Exception):
    """Base exception for Ultravox Pipeline"""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        retry_after: Optional[int] = None,
        recoverable: bool = True,
        **kwargs,
    ):
        super().__init__(message)

        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or f"{category.value.upper()}_ERROR"
        self.context = context or ErrorContext()
        self.retry_after = retry_after
        self.recoverable = recoverable
        self.details = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to structured dictionary"""
        return {
            "error": {
                "id": self.context.error_id,
                "code": self.error_code,
                "message": self.message,
                "category": self.category.value,
                "severity": self.severity.value,
                "timestamp": self.context.timestamp.isoformat(),
                "recoverable": self.recoverable,
                "retry_after": self.retry_after,
                "context": {
                    "correlation_id": self.context.correlation_id,
                    "session_id": self.context.session_id,
                    "user_id": self.context.user_id,
                    "component": self.context.component,
                    "operation": self.context.operation,
                },
                "details": self.details,
            }
        }

    def to_json(self) -> str:
        """Convert error to JSON string"""
        return json.dumps(self.to_dict(), default=str)


# === VALIDATION ERRORS ===


class ValidationError(UltravoxError):
    """Input validation error"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            error_code="VALIDATION_ERROR",
            **kwargs,
        )
        if field:
            self.details["field"] = field


# === AUTHENTICATION & AUTHORIZATION ERRORS ===


class AuthenticationError(UltravoxError):
    """Authentication required or failed"""

    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            error_code="AUTH_ERROR",
            recoverable=False,
            **kwargs,
        )


class AuthorizationError(UltravoxError):
    """Authorization/permission denied"""

    def __init__(self, message: str = "Permission denied", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            error_code="AUTHORIZATION_ERROR",
            recoverable=False,
            **kwargs,
        )


# === RATE LIMITING ERRORS ===


class RateLimitError(UltravoxError):
    """Rate limit exceeded"""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            error_code="RATE_LIMIT_EXCEEDED",
            retry_after=retry_after,
            **kwargs,
        )


# === SERVICE AVAILABILITY ERRORS ===


class ServiceUnavailableError(UltravoxError):
    """Service temporarily unavailable"""

    def __init__(
        self, service: str, message: Optional[str] = None, retry_after: int = 30, **kwargs
    ):
        message = message or f"Service {service} is currently unavailable"
        super().__init__(
            message=message,
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            severity=ErrorSeverity.HIGH,
            error_code="SERVICE_UNAVAILABLE",
            retry_after=retry_after,
            **kwargs,
        )
        self.details["service"] = service


class CircuitBreakerOpenError(ServiceUnavailableError):
    """Circuit breaker is open - service protection active"""

    def __init__(self, service: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(
            service=service,
            message=f"Circuit breaker open for {service}",
            retry_after=retry_after,
            **kwargs,
        )
        self.error_code = "CIRCUIT_BREAKER_OPEN"


# === PROCESSING ERRORS ===


class ProcessingError(UltravoxError):
    """General processing error"""

    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            error_code="PROCESSING_ERROR",
            **kwargs,
        )
        if component:
            self.details["component"] = component


class TimeoutError(UltravoxError):
    """Operation timeout"""

    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout_seconds}s",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.HIGH,
            error_code="TIMEOUT_ERROR",
            retry_after=int(timeout_seconds),
            **kwargs,
        )
        self.details.update({"operation": operation, "timeout_seconds": timeout_seconds})


# === RESOURCE ERRORS ===


class ResourceExhaustedError(UltravoxError):
    """Resource exhausted (memory, GPU, disk, etc)"""

    def __init__(self, resource: str, message: Optional[str] = None, **kwargs):
        message = message or f"Resource '{resource}' exhausted"
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE_EXHAUSTED,
            severity=ErrorSeverity.CRITICAL,
            error_code="RESOURCE_EXHAUSTED",
            retry_after=120,
            **kwargs,
        )
        self.details["resource"] = resource


# === NETWORK ERRORS ===


class NetworkError(UltravoxError):
    """Network connectivity error"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            error_code="NETWORK_ERROR",
            retry_after=30,
            **kwargs,
        )


# === CONFIGURATION ERRORS ===


class ConfigurationError(UltravoxError):
    """Configuration error - invalid settings"""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            error_code="CONFIGURATION_ERROR",
            recoverable=False,
            **kwargs,
        )
        if config_key:
            self.details["config_key"] = config_key


# === ERROR CONVERSION UTILITIES ===


def convert_exception_to_ultravox_error(
    error: Exception, context: Optional[ErrorContext] = None
) -> UltravoxError:
    """
    Convert generic exception to UltravoxError
    Intelligent mapping based on exception type and message
    """
    error_type = type(error).__name__
    message = str(error)

    # Already an UltravoxError
    if isinstance(error, UltravoxError):
        if context:
            error.context = context
        return error

    # Timeout errors
    if "timeout" in message.lower() or "TimeoutError" in error_type:
        return TimeoutError("timeout", 30.0, context=context)

    # Connection/Network errors
    if any(
        keyword in message.lower()
        for keyword in ["connection", "network", "unreachable", "refused"]
    ):
        return NetworkError(message, context=context)

    # Resource exhaustion
    if "memory" in message.lower() or "CUDA" in message or "OOM" in message:
        return ResourceExhaustedError("gpu_memory", message, context=context)

    # Validation errors
    if "validation" in message.lower() or "ValueError" in error_type:
        return ValidationError(message, context=context)

    # Rate limiting
    if "rate" in message.lower() and "limit" in message.lower():
        return RateLimitError(message, context=context)

    # Authentication
    if any(keyword in message.lower() for keyword in ["auth", "unauthorized", "token"]):
        return AuthenticationError(message, context=context)

    # Default: generic error
    return UltravoxError(
        message=message,
        category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.MEDIUM,
        context=context,
        original_exception=error_type,
    )
