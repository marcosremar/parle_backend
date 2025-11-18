"""
FastAPI Error Handler Middleware - Centralized error handling for FastAPI apps.

Provides middleware that intercepts all exceptions and converts them to
appropriate HTTP responses with proper status codes and error context.

Features:
- Automatic conversion of UltravoxError to HTTP responses
- OpenTelemetry integration (trace_id in responses)
- Structured error responses
- Automatic logging with trace context

Usage:
    from fastapi import FastAPI
    from src.core.error_handling.middleware import setup_error_handling

    app = FastAPI()
    setup_error_handling(app)

    # Now all exceptions are automatically handled
    @app.post("/process")
    async def process_audio(audio: bytes):
        # Any exception raised here will be caught and converted
        result = await service.process(audio)
        return result
"""

from typing import Union, Dict, Any, Optional
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from loguru import logger

from src.core.exceptions import (
    UltravoxError,
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    ServiceInitializationError,
    CommunicationError,
    ValidationError as UltravoxValidationError,
    RequestValidationError as UltravoxRequestValidationError,
    ResponseValidationError,
    ConfigurationError,
    ResourceError,
    GPUNotAvailableError,
    MemoryError as UltravoxMemoryError,
    StorageError,
    AIError,
)
from .telemetry import add_error_to_span, log_error_with_context, get_trace_context


# Map exception types to HTTP status codes
EXCEPTION_STATUS_MAP: Dict[type, int] = {
    # Validation errors → 400 Bad Request
    UltravoxValidationError: status.HTTP_400_BAD_REQUEST,
    UltravoxRequestValidationError: status.HTTP_400_BAD_REQUEST,
    ConfigurationError: status.HTTP_400_BAD_REQUEST,
    # Service errors → 503 Service Unavailable or 504 Gateway Timeout
    ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ServiceTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
    ServiceInitializationError: status.HTTP_503_SERVICE_UNAVAILABLE,
    # Communication errors → 502 Bad Gateway or 503
    CommunicationError: status.HTTP_502_BAD_GATEWAY,
    # Resource errors → 507 Insufficient Storage or 503
    ResourceError: status.HTTP_507_INSUFFICIENT_STORAGE,
    GPUNotAvailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    UltravoxMemoryError: status.HTTP_507_INSUFFICIENT_STORAGE,
    StorageError: status.HTTP_507_INSUFFICIENT_STORAGE,
    # AI errors → 500 Internal Server Error
    AIError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    # Response validation → 502 Bad Gateway (upstream returned bad data)
    ResponseValidationError: status.HTTP_502_BAD_GATEWAY,
    # Generic service error → 500
    ServiceError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    # Generic Ultravox error → 500
    UltravoxError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def get_status_code_for_exception(error: Exception) -> int:
    """
    Get appropriate HTTP status code for exception.

    Args:
        error: Exception to map

    Returns:
        HTTP status code

    Example:
        status_code = get_status_code_for_exception(ServiceUnavailableError("llm"))
        # Returns: 503
    """
    # Try to find exact match first
    error_type = type(error)
    if error_type in EXCEPTION_STATUS_MAP:
        return EXCEPTION_STATUS_MAP[error_type]

    # Try parent classes (MRO)
    for base_class in error_type.__mro__:
        if base_class in EXCEPTION_STATUS_MAP:
            return EXCEPTION_STATUS_MAP[base_class]

    # Default to 500 Internal Server Error
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def create_error_response(
    error: Exception,
    status_code: int,
    include_details: bool = True,
    include_trace_id: bool = True,
) -> Dict[str, Any]:
    """
    Create standardized error response dictionary.

    Args:
        error: Exception to convert
        status_code: HTTP status code
        include_details: Include error details (default: True)
        include_trace_id: Include trace_id from OpenTelemetry (default: True)

    Returns:
        Dictionary with error response

    Example:
        response = create_error_response(
            ServiceUnavailableError("llm"),
            503
        )
        # Returns: {
        #     "error": {
        #         "code": "SERVICE_UNAVAILABLE",
        #         "message": "Service 'llm' is unavailable",
        #         "type": "ServiceUnavailableError",
        #         "trace_id": "abc123..."
        #     }
        # }
    """
    response: Dict[str, Any] = {
        "error": {
            "code": getattr(error, "error_code", "INTERNAL_ERROR"),
            "message": str(error),
            "type": type(error).__name__,
        }
    }

    # Add UltravoxError details
    if isinstance(error, UltravoxError):
        if include_details and error.details:
            response["error"]["details"] = error.details

        # Add retryable flag
        response["error"]["retryable"] = error.details.get("retryable", False)

        # Add retry_after if available
        if "retry_after" in error.details:
            response["error"]["retry_after"] = error.details["retry_after"]

    # Add OpenTelemetry trace context
    if include_trace_id:
        trace_context = get_trace_context()
        if trace_context:
            response["error"]["trace_id"] = trace_context["trace_id"]
            response["error"]["span_id"] = trace_context["span_id"]

    return response


class ErrorHandlerMiddleware:
    """
    FastAPI middleware for centralized error handling.

    Intercepts all exceptions and converts them to appropriate HTTP responses.
    """

    def __init__(
        self,
        app: FastAPI,
        include_details: bool = True,
        include_trace_id: bool = True,
        log_errors: bool = True,
    ):
        """
        Initialize error handler middleware.

        Args:
            app: FastAPI application
            include_details: Include error details in response (default: True)
            include_trace_id: Include OpenTelemetry trace_id (default: True)
            log_errors: Log errors automatically (default: True)
        """
        self.app = app
        self.include_details = include_details
        self.include_trace_id = include_trace_id
        self.log_errors = log_errors

    async def __call__(self, request: Request, call_next):
        """
        Process request and handle any exceptions.

        Args:
            request: FastAPI request
            call_next: Next middleware in chain

        Returns:
            Response (normal or error response)
        """
        try:
            response = await call_next(request)
            return response

        except Exception as error:
            return await self.handle_error(request, error)

    async def handle_error(self, request: Request, error: Exception) -> JSONResponse:
        """
        Handle exception and create error response.

        Args:
            request: FastAPI request
            error: Exception to handle

        Returns:
            JSON error response
        """
        # Add error to OpenTelemetry span
        add_error_to_span(error)

        # Log error with context
        if self.log_errors:
            log_error_with_context(
                error,
                level="error",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "client": request.client.host if request.client else "unknown",
                },
            )

        # Get appropriate status code
        status_code = get_status_code_for_exception(error)

        # Create error response
        response_data = create_error_response(
            error,
            status_code,
            include_details=self.include_details,
            include_trace_id=self.include_trace_id,
        )

        # Return JSON response
        return JSONResponse(status_code=status_code, content=response_data)


async def ultravox_exception_handler(request: Request, exc: UltravoxError) -> JSONResponse:
    """
    Exception handler for UltravoxError.

    Registers with FastAPI using app.add_exception_handler().

    Args:
        request: FastAPI request
        exc: UltravoxError exception

    Returns:
        JSON error response
    """
    # Add error to span
    add_error_to_span(exc)

    # Log error
    log_error_with_context(
        exc,
        level="error",
        extra={
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Get status code
    status_code = get_status_code_for_exception(exc)

    # Create response
    response_data = create_error_response(exc, status_code)

    return JSONResponse(status_code=status_code, content=response_data)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Exception handler for Pydantic validation errors.

    Args:
        request: FastAPI request
        exc: Pydantic validation error

    Returns:
        JSON error response
    """
    # Convert to UltravoxRequestValidationError
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    field = ".".join(str(loc) for loc in first_error.get("loc", []))
    message = first_error.get("msg", "Validation error")

    ultravox_error = UltravoxRequestValidationError(field=field, reason=message)

    # Add to span
    add_error_to_span(ultravox_error)

    # Log error
    log_error_with_context(
        ultravox_error,
        level="warning",
        extra={
            "path": request.url.path,
            "method": request.method,
            "validation_errors": errors,
        },
    )

    # Create response
    response_data = create_error_response(ultravox_error, status.HTTP_400_BAD_REQUEST)
    response_data["error"]["validation_errors"] = errors

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=response_data)


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Exception handler for Starlette HTTP exceptions.

    Args:
        request: FastAPI request
        exc: HTTP exception

    Returns:
        JSON error response
    """
    # Log error
    logger.warning(
        f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}"
    )

    # Create simple error response
    response_data = {
        "error": {
            "code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "type": "HTTPException",
        }
    }

    # Add trace context if available
    trace_context = get_trace_context()
    if trace_context:
        response_data["error"]["trace_id"] = trace_context["trace_id"]

    return JSONResponse(status_code=exc.status_code, content=response_data)


def setup_error_handling(
    app: FastAPI,
    include_details: bool = True,
    include_trace_id: bool = True,
    log_errors: bool = True,
) -> None:
    """
    Setup error handling for FastAPI application.

    Registers exception handlers for all error types.

    Args:
        app: FastAPI application
        include_details: Include error details in responses (default: True)
        include_trace_id: Include OpenTelemetry trace_id (default: True)
        log_errors: Log errors automatically (default: True)

    Example:
        from fastapi import FastAPI
        from src.core.error_handling.middleware import setup_error_handling

        app = FastAPI()
        setup_error_handling(app)

        @app.post("/process")
        async def process_data(data: dict):
            # Any exception here will be handled automatically
            if not data:
                raise RequestValidationError("data", "Data cannot be empty")
            return {"status": "ok"}
    """
    # Register exception handlers
    app.add_exception_handler(UltravoxError, ultravox_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    logger.info(
        "✅ Error handling configured "
        f"(details={include_details}, trace_id={include_trace_id}, logging={log_errors})"
    )


# Utility function to add to existing FastAPI apps
def add_error_handling_to_app(app: FastAPI) -> None:
    """
    Add error handling to existing FastAPI app.

    Convenience function that calls setup_error_handling with defaults.

    Args:
        app: FastAPI application

    Example:
        from fastapi import FastAPI
        from src.core.error_handling.middleware import add_error_handling_to_app

        app = FastAPI()
        add_error_handling_to_app(app)
    """
    setup_error_handling(app, include_details=True, include_trace_id=True, log_errors=True)
