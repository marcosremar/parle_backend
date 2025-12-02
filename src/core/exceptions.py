"""
Core Exceptions Module
Shared exception hierarchy for the application.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

try:
    import orjson as json
    def json_dumps(obj, default=str):
        return json.dumps(obj, default=default).decode("utf-8")
except ImportError:
    import json
    def json_dumps(obj, default=str):
        return json.dumps(obj, default=default)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    error_id: str = field(default_factory=lambda: f"err_{int(datetime.now(timezone.utc).timestamp() * 1000)}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    component: str | None = None
    operation: str | None = None
    request_data: dict[str, Any] | None = None
    system_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
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


class UltravoxError(Exception):
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: str | None = None,
        context: ErrorContext | None = None,
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
            parts.append(f"Caused by: {type(self.original_error).__name__}: {self.original_error}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "severity": self.severity.value,
                "type": type(self).__name__,
                "retryable": self.details.get("retryable", False),
                "details": self.details,
                "context": self.context.to_dict() if self.context else None,
                "original_error": (type(self.original_error).__name__ if self.original_error else None),
            }
        }

    def to_json(self) -> str:
        return json_dumps(self.to_dict(), default=str)


class ServiceError(UltravoxError):
    pass


class ServiceUnavailableError(ServiceError):
    def __init__(self, service_name: str, original_error: Exception | None = None):
        super().__init__(
            f"Service '{service_name}' is unavailable",
            details={"service": service_name, "retryable": True},
            original_error=original_error,
        )


class ServiceTimeoutError(ServiceError):
    def __init__(self, service_name: str, timeout_ms: int, original_error: Exception | None = None):
        super().__init__(
            f"Service '{service_name}' timed out after {timeout_ms}ms",
            details={"service": service_name, "timeout_ms": timeout_ms, "retryable": True},
            original_error=original_error,
        )


class CommunicationError(UltravoxError):
    pass


class NetworkError(CommunicationError):
    def __init__(self, endpoint: str, original_error: Exception | None = None):
        super().__init__(
            f"Network error connecting to {endpoint}",
            details={"endpoint": endpoint, "retryable": True},
            original_error=original_error,
        )


class ValidationError(UltravoxError):
    pass


class RequestValidationError(ValidationError):
    def __init__(self, field: str, reason: str, original_error: Exception | None = None):
        super().__init__(
            f"Invalid request field '{field}': {reason}",
            details={"field": field, "reason": reason, "retryable": False},
            original_error=original_error,
        )


def wrap_exception(
    error: Exception, service_name: str | None = None, operation: str | None = None
) -> UltravoxError:
    import aiohttp
    import asyncio

    if isinstance(error, (aiohttp.ClientError, ConnectionError)):
        if service_name:
            return ServiceUnavailableError(service_name, error)
        return NetworkError(operation or "unknown", error)

    if isinstance(error, (aiohttp.ServerTimeoutError, TimeoutError, asyncio.TimeoutError)):
        if service_name:
            return ServiceTimeoutError(service_name, 0, error)
        return NetworkError(operation or "unknown", error)

    try:
        import pydantic
        if isinstance(error, pydantic.ValidationError):
            return RequestValidationError("unknown", str(error), error)
    except ImportError:
        pass

    return ServiceError(f"Error during {operation or 'operation'}: {error}", original_error=error)
