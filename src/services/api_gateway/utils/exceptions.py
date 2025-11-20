"""
Custom exception hierarchy for Ultravox Pipeline.

Provides specific exception types to replace broad `except Exception` handlers
throughout the codebase. Improves error diagnosis and handling.

Exception Hierarchy:
    UltravoxError (base)
    ├── ServiceError
    │   ├── ServiceUnavailableError
    │   ├── ServiceTimeoutError
    │   └── ServiceInitializationError
    ├── CommunicationError
    │   ├── NetworkError
    │   ├── ProtocolError
    │   └── SerializationError
    ├── ValidationError
    │   ├── RequestValidationError
    │   ├── ResponseValidationError
    │   └── ConfigurationError
    ├── SecurityError
    │   ├── AuthenticationError
    │   ├── AuthorizationError
    │   └── RateLimitError
    ├── ResourceError
    │   ├── GPUNotAvailableError
    │   ├── MemoryError
    │   └── StorageError
    └── AIError
        ├── LLMError
        ├── STTError
        └── TTSError

Usage:
    from src.core.exceptions import ServiceUnavailableError, ValidationError

    try:
        result = await call_service()
    except aiohttp.ClientError as e:
        raise ServiceUnavailableError("LLM service") from e
    except pydantic.ValidationError as e:
        raise RequestValidationError("Invalid audio data") from e
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Dict


# ============================================================================
# Enums and Context
# ============================================================================


class ErrorSeverity(Enum):
    """
    Error severity levels.

    Used to categorize errors by impact and urgency.
    """

    LOW = "low"  # Minor issues, degraded functionality
    MEDIUM = "medium"  # Significant issues, partial failure
    HIGH = "high"  # Critical issues, major failure
    CRITICAL = "critical"  # System-wide failure, immediate action required


@dataclass
class ErrorContext:
    """
    Rich context for errors to aid debugging and monitoring.

    Provides detailed information about the error including timing,
    correlation IDs, and system state.

    Attributes:
        error_id: Unique identifier for this error instance
        timestamp: When the error occurred
        correlation_id: Request correlation ID (for distributed tracing)
        trace_id: OpenTelemetry trace ID (32 hex chars)
        span_id: OpenTelemetry span ID (16 hex chars)
        session_id: User session ID (if applicable)
        user_id: User identifier (if applicable)
        component: Component where error occurred
        operation: Operation being performed
        request_data: Request data (sanitized)
        system_info: System state at time of error
    """

    error_id: str = field(default_factory=lambda: f"err_{int(datetime.utcnow().timestamp() * 1000)}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    system_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "component": self.component,
            "operation": self.operation,
            "request_data": self.request_data,
            "system_info": self.system_info,
        }


# ============================================================================
# Base Exception
# ============================================================================


class UltravoxError(Exception):
    """
    Base exception for all Ultravox Pipeline errors.

    All custom exceptions should inherit from this class.
    This allows catching all Ultravox-specific errors with a single except clause.

    Attributes:
        message: Error message
        details: Additional error context (optional)
        original_error: Original exception that caused this error (optional)
        severity: Error severity level
        error_code: Machine-readable error code
        context: Rich error context
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
    ):
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        self.severity = severity
        self.error_code = error_code or "ULTRAVOX_ERROR"
        self.context = context or ErrorContext()
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.original_error:
            parts.append(
                f"Caused by: {type(self.original_error).__name__}: {self.original_error}"
            )
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for structured logging/responses.

        Returns:
            Dictionary with error information

        Example:
            error = ServiceUnavailableError("llm")
            error_dict = error.to_dict()
            # {
            #     "error": {
            #         "code": "SERVICE_UNAVAILABLE",
            #         "message": "Service 'llm' is unavailable",
            #         "severity": "high",
            #         "type": "ServiceUnavailableError",
            #         "retryable": true,
            #         "context": {...}
            #     }
            # }
        """
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "severity": self.severity.value,
                "type": type(self).__name__,
                "retryable": self.details.get("retryable", False),
                "details": self.details,
                "context": self.context.to_dict() if self.context else None,
                "original_error": (
                    type(self.original_error).__name__ if self.original_error else None
                ),
            }
        }

    def to_json(self) -> str:
        """
        Convert exception to JSON string.

        Returns:
            JSON string with error information

        Example:
            error = ServiceUnavailableError("llm")
            json_str = error.to_json()
        """
        return json.dumps(self.to_dict(), default=str)


# ============================================================================
# Service Errors
# ============================================================================


class ServiceError(UltravoxError):
    """Base class for service-related errors."""

    pass


class ServiceUnavailableError(ServiceError):
    """
    Service is not available or not responding.

    Raised when:
    - Service is down or unreachable
    - Network connection fails
    - Service discovery fails to locate service

    This is a retryable error.
    """

    def __init__(self, service_name: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Service '{service_name}' is unavailable",
            details={"service": service_name, "retryable": True},
            original_error=original_error,
        )


class ServiceTimeoutError(ServiceError):
    """
    Service request timed out.

    Raised when:
    - Service takes too long to respond
    - Deadline exceeded

    This is a retryable error.
    """

    def __init__(
        self, service_name: str, timeout_ms: int, original_error: Optional[Exception] = None
    ):
        super().__init__(
            f"Service '{service_name}' timed out after {timeout_ms}ms",
            details={"service": service_name, "timeout_ms": timeout_ms, "retryable": True},
            original_error=original_error,
        )


class ServiceInitializationError(ServiceError):
    """
    Service failed to initialize.

    Raised when:
    - Service startup fails
    - Required resources unavailable
    - Configuration invalid

    This is NOT retryable without intervention.
    """

    def __init__(self, service_name: str, reason: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Failed to initialize service '{service_name}': {reason}",
            details={"service": service_name, "reason": reason, "retryable": False},
            original_error=original_error,
        )


# ============================================================================
# Communication Errors
# ============================================================================


class CommunicationError(UltravoxError):
    """Base class for inter-service communication errors."""

    pass


class NetworkError(CommunicationError):
    """
    Network-level error.

    Raised when:
    - Connection refused
    - Connection reset
    - DNS resolution fails

    This is a retryable error.
    """

    def __init__(self, endpoint: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Network error connecting to {endpoint}",
            details={"endpoint": endpoint, "retryable": True},
            original_error=original_error,
        )


class ProtocolError(CommunicationError):
    """
    Protocol-level error.

    Raised when:
    - Invalid HTTP status code
    - gRPC error
    - ZeroMQ socket error

    May be retryable depending on the error.
    """

    def __init__(
        self,
        protocol: str,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {"protocol": protocol}
        if status_code:
            details["status_code"] = status_code
            details["retryable"] = status_code >= 500  # Server errors are retryable

        super().__init__(
            f"Protocol error ({protocol})" + (f" - status {status_code}" if status_code else ""),
            details=details,
            original_error=original_error,
        )


class SerializationError(CommunicationError):
    """
    Data serialization/deserialization error.

    Raised when:
    - JSON encoding/decoding fails
    - Protobuf serialization fails
    - Invalid data format

    This is NOT retryable without fixing the data.
    """

    def __init__(self, data_format: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Serialization error ({data_format})",
            details={"format": data_format, "retryable": False},
            original_error=original_error,
        )


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(UltravoxError):
    """Base class for validation errors."""

    pass


class RequestValidationError(ValidationError):
    """
    Request validation failed.

    Raised when:
    - Pydantic validation fails on request
    - Required fields missing
    - Invalid field values

    This is NOT retryable (client error).
    """

    def __init__(self, field: str, reason: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Invalid request field '{field}': {reason}",
            details={"field": field, "reason": reason, "retryable": False},
            original_error=original_error,
        )


class ResponseValidationError(ValidationError):
    """
    Response validation failed.

    Raised when:
    - Service returns invalid data
    - Pydantic validation fails on response
    - Data integrity check fails

    This may be retryable (server error).
    """

    def __init__(self, service: str, reason: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Invalid response from '{service}': {reason}",
            details={"service": service, "reason": reason, "retryable": True},
            original_error=original_error,
        )


class ConfigurationError(ValidationError):
    """
    Configuration validation failed.

    Raised when:
    - Missing required config
    - Invalid config values
    - Config file malformed

    This is NOT retryable without fixing config.
    """

    def __init__(self, config_key: str, reason: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Invalid configuration '{config_key}': {reason}",
            details={"config_key": config_key, "reason": reason, "retryable": False},
            original_error=original_error,
        )


# ============================================================================
# Security Errors
# ============================================================================


class SecurityError(UltravoxError):
    """Base class for security-related errors."""

    pass


class AuthenticationError(SecurityError):
    """
    Authentication failed.

    Raised when:
    - Missing or invalid credentials
    - Token expired
    - Authentication service unavailable

    This is NOT retryable without new credentials.
    """

    def __init__(
        self,
        reason: str = "Authentication required",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            f"Authentication failed: {reason}",
            details={"reason": reason, "retryable": False},
            original_error=original_error,
            severity=ErrorSeverity.HIGH,
            error_code="AUTHENTICATION_FAILED",
        )


class AuthorizationError(SecurityError):
    """
    Authorization failed.

    Raised when:
    - User lacks required permissions
    - Resource access denied
    - Role/scope insufficient

    This is NOT retryable without permission changes.
    """

    def __init__(
        self,
        resource: str,
        action: str,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            f"Access denied: Cannot {action} {resource}",
            details={
                "resource": resource,
                "action": action,
                "retryable": False,
            },
            original_error=original_error,
            severity=ErrorSeverity.HIGH,
            error_code="AUTHORIZATION_DENIED",
        )


class RateLimitError(SecurityError):
    """
    Rate limit exceeded.

    Raised when:
    - Too many requests
    - Quota exceeded
    - Throttling applied

    This is retryable after retry_after seconds.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: int = 60,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            details={
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after,
                "retryable": True,
            },
            original_error=original_error,
            severity=ErrorSeverity.MEDIUM,
            error_code="RATE_LIMIT_EXCEEDED",
        )


# ============================================================================
# Resource Errors
# ============================================================================


class ResourceError(UltravoxError):
    """Base class for resource-related errors."""

    pass


class GPUNotAvailableError(ResourceError):
    """
    GPU not available.

    Raised when:
    - No GPU detected
    - GPU out of memory
    - CUDA error

    May be retryable if temporary (e.g., OOM).
    """

    def __init__(self, reason: str, retryable: bool = False):
        super().__init__(
            f"GPU not available: {reason}", details={"reason": reason, "retryable": retryable}
        )


class MemoryError(ResourceError):
    """
    Memory allocation failed.

    Raised when:
    - Out of RAM
    - Memory limit exceeded

    May be retryable after cleanup.
    """

    def __init__(self, required_mb: Optional[int] = None, retryable: bool = True):
        details = {"retryable": retryable}
        if required_mb:
            details["required_mb"] = required_mb

        super().__init__(
            "Out of memory" + (f" (needed {required_mb}MB)" if required_mb else ""),
            details=details,
        )


class StorageError(ResourceError):
    """
    Storage operation failed.

    Raised when:
    - Disk full
    - File not found
    - Permission denied

    Retryable depends on the error.
    """

    def __init__(self, operation: str, path: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Storage error during {operation}: {path}",
            details={"operation": operation, "path": path},
            original_error=original_error,
        )


# ============================================================================
# AI Service Errors
# ============================================================================


class AIError(UltravoxError):
    """Base class for AI service errors."""

    pass


class LLMError(AIError):
    """
    LLM (Large Language Model) error.

    Raised when:
    - LLM inference fails
    - Invalid prompt
    - Model not loaded

    May be retryable depending on the error.
    """

    def __init__(
        self, model: str, reason: str, retryable: bool = True, original_error: Optional[Exception] = None
    ):
        super().__init__(
            f"LLM error ({model}): {reason}",
            details={"model": model, "reason": reason, "retryable": retryable},
            original_error=original_error,
        )


class STTError(AIError):
    """
    Speech-to-Text error.

    Raised when:
    - Transcription fails
    - Invalid audio format
    - Model not loaded

    May be retryable depending on the error.
    """

    def __init__(
        self, model: str, reason: str, retryable: bool = True, original_error: Optional[Exception] = None
    ):
        super().__init__(
            f"STT error ({model}): {reason}",
            details={"model": model, "reason": reason, "retryable": retryable},
            original_error=original_error,
        )


class TTSError(AIError):
    """
    Text-to-Speech error.

    Raised when:
    - Synthesis fails
    - Invalid text input
    - Model not loaded

    May be retryable depending on the error.
    """

    def __init__(
        self, model: str, reason: str, retryable: bool = True, original_error: Optional[Exception] = None
    ):
        super().__init__(
            f"TTS error ({model}): {reason}",
            details={"model": model, "reason": reason, "retryable": retryable},
            original_error=original_error,
        )


# ============================================================================
# Helper Functions
# ============================================================================


def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if error is retryable, False otherwise

    Example:
        try:
            result = await call_service()
        except Exception as e:
            if is_retryable(e):
                # Retry logic
                pass
            else:
                # Fail fast
                raise
    """
    if isinstance(error, UltravoxError):
        return error.details.get("retryable", False)

    # Network errors are generally retryable
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    # Default: not retryable
    return False


def wrap_exception(
    error: Exception, service_name: Optional[str] = None, operation: Optional[str] = None
) -> UltravoxError:
    """
    Wrap a generic exception into an appropriate UltravoxError.

    Args:
        error: Original exception
        service_name: Name of service (if applicable)
        operation: Operation being performed (if applicable)

    Returns:
        Wrapped UltravoxError

    Example:
        try:
            response = await http_client.get(url)
        except aiohttp.ClientError as e:
            raise wrap_exception(e, service_name="llm", operation="inference")
    """
    import aiohttp

    # Network errors
    if isinstance(error, (aiohttp.ClientError, ConnectionError)):
        if service_name:
            return ServiceUnavailableError(service_name, error)
        return NetworkError(operation or "unknown", error)

    # Timeout errors
    if isinstance(error, (aiohttp.ServerTimeoutError, TimeoutError, asyncio.TimeoutError)):
        if service_name:
            return ServiceTimeoutError(service_name, 0, error)
        return NetworkError(operation or "unknown", error)

    # Validation errors
    try:
        import pydantic

        if isinstance(error, pydantic.ValidationError):
            return RequestValidationError("unknown", str(error), error)
    except ImportError:
        pass

    # Default: wrap as generic service error
    return ServiceError(f"Error during {operation or 'operation'}: {error}", original_error=error)
