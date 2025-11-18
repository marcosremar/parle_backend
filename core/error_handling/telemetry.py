"""
Telemetry Integration for Error Handling.

Integrates error handling with OpenTelemetry for distributed tracing.
Adds error information to spans and provides structured logging with trace context.

Usage:
    from src.core.error_handling.telemetry import add_error_to_span, log_error_with_context
    from src.core.exceptions import ServiceUnavailableError

    try:
        result = await call_service()
    except ServiceUnavailableError as e:
        add_error_to_span(e)
        log_error_with_context(e, extra={"service": "llm"})
        raise
"""

import sys
import traceback
from typing import Optional, Dict, Any
from loguru import logger

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from src.core.exceptions import UltravoxError


def add_error_to_span(
    error: Exception,
    span: Optional["trace.Span"] = None,
    record_exception: bool = True,
) -> None:
    """
    Add error information to OpenTelemetry span.

    Adds error attributes and optionally records the full exception
    in the span for distributed tracing.

    Args:
        error: Exception to record
        span: Specific span to use (default: current span)
        record_exception: Record full exception details (default: True)

    Example:
        try:
            result = await call_service()
        except Exception as e:
            add_error_to_span(e)
            raise
    """
    if not OTEL_AVAILABLE:
        return

    # Get current span if not provided
    if span is None:
        span = trace.get_current_span()

    # Check if span is valid (not NoOp span)
    if not span.is_recording():
        return

    # Set span status to ERROR
    span.set_status(Status(StatusCode.ERROR, str(error)))

    # Add error attributes
    span.set_attribute("error", True)
    span.set_attribute("error.type", type(error).__name__)
    span.set_attribute("error.message", str(error))

    # Add UltravoxError-specific attributes
    if isinstance(error, UltravoxError):
        span.set_attribute("error.retryable", error.details.get("retryable", False))
        if error.details:
            for key, value in error.details.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"error.details.{key}", value)

    # Record full exception with traceback
    if record_exception:
        span.record_exception(error)


def log_error_with_context(
    error: Exception,
    level: str = "error",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log error with OpenTelemetry trace context.

    Adds trace_id and span_id to log for correlation with traces.

    Args:
        error: Exception to log
        level: Log level (debug, info, warning, error, critical)
        extra: Additional context to include in log

    Example:
        try:
            result = await call_service()
        except Exception as e:
            log_error_with_context(
                e,
                level="error",
                extra={"service": "llm", "operation": "generate"}
            )
            raise
    """
    # Build log context
    log_context: Dict[str, Any] = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    # Add OpenTelemetry trace context if available
    if OTEL_AVAILABLE:
        span = trace.get_current_span()
        if span.is_recording():
            span_context = span.get_span_context()
            log_context["trace_id"] = format(span_context.trace_id, "032x")
            log_context["span_id"] = format(span_context.span_id, "016x")

    # Add UltravoxError details
    if isinstance(error, UltravoxError):
        log_context["error_retryable"] = error.details.get("retryable", False)
        if error.details:
            log_context.update({f"detail_{k}": v for k, v in error.details.items()})

    # Add extra context
    if extra:
        log_context.update(extra)

    # Get traceback
    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    log_context["traceback"] = tb

    # Log with appropriate level
    log_func = getattr(logger, level.lower(), logger.error)
    log_func(f"Error: {str(error)}", **log_context)


def get_trace_context() -> Optional[Dict[str, str]]:
    """
    Get current OpenTelemetry trace context.

    Returns trace_id and span_id for the current span.

    Returns:
        Dictionary with trace_id and span_id, or None if no active span

    Example:
        context = get_trace_context()
        if context:
            logger.info(f"Request {context['trace_id']}: Processing...")
    """
    if not OTEL_AVAILABLE:
        return None

    span = trace.get_current_span()
    if not span.is_recording():
        return None

    span_context = span.get_span_context()
    return {
        "trace_id": format(span_context.trace_id, "032x"),
        "span_id": format(span_context.span_id, "016x"),
    }


def create_error_span_attributes(error: Exception) -> Dict[str, Any]:
    """
    Create span attributes dictionary from exception.

    Useful for adding error context to spans in batch.

    Args:
        error: Exception to extract attributes from

    Returns:
        Dictionary of span attributes

    Example:
        span.set_attributes(create_error_span_attributes(error))
    """
    attributes = {
        "error": True,
        "error.type": type(error).__name__,
        "error.message": str(error),
    }

    # Add UltravoxError-specific attributes
    if isinstance(error, UltravoxError):
        attributes["error.retryable"] = error.details.get("retryable", False)
        if error.details:
            for key, value in error.details.items():
                if isinstance(value, (str, int, float, bool)):
                    attributes[f"error.details.{key}"] = value

    return attributes


def wrap_span_with_error_handling(span_name: str):
    """
    Decorator to create a span with automatic error handling.

    Creates an OpenTelemetry span and automatically adds error info if exception occurs.

    Args:
        span_name: Name for the span

    Example:
        @wrap_span_with_error_handling("process_audio")
        async def process_audio(audio: bytes) -> str:
            return await transcribe(audio)
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return await func(*args, **kwargs)

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    add_error_to_span(e, span=span)
                    log_error_with_context(e, extra={"span": span_name})
                    raise

        return wrapper

    return decorator


# Context manager for error tracking in spans
class ErrorTrackedSpan:
    """
    Context manager for creating spans with automatic error tracking.

    Example:
        async with ErrorTrackedSpan("process_request") as span:
            result = await process()
            span.set_attribute("result_size", len(result))
    """

    def __init__(self, span_name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Initialize error-tracked span.

        Args:
            span_name: Name for the span
            attributes: Initial attributes to set on span
        """
        self.span_name = span_name
        self.attributes = attributes or {}
        self.span = None
        self.tracer = None

    def __enter__(self):
        """Enter context - create span."""
        if OTEL_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            self.span = self.tracer.start_span(self.span_name)

            # Set initial attributes
            if self.attributes:
                self.span.set_attributes(self.attributes)

            # Make this span current
            self._token = trace.set_span_in_context(self.span).__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - handle errors and close span."""
        if self.span:
            if exc_type is not None:
                # Error occurred
                add_error_to_span(exc_val, span=self.span)
                log_error_with_context(exc_val, extra={"span": self.span_name})
            else:
                # Success
                self.span.set_status(Status(StatusCode.OK))

            # End span
            self.span.end()

            # Restore context
            if hasattr(self, "_token"):
                self._token.__exit__(exc_type, exc_val, exc_tb)

        return False  # Don't suppress exceptions

    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute on span."""
        if self.span and self.span.is_recording():
            self.span.set_attribute(key, value)

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple attributes on span."""
        if self.span and self.span.is_recording():
            self.span.set_attributes(attributes)


# Utility for legacy code that doesn't use async/await
def log_exception_to_span(error: Exception) -> None:
    """
    Convenience function to log exception to current span and logger.

    Combines add_error_to_span() and log_error_with_context() in one call.

    Args:
        error: Exception to log

    Example:
        try:
            result = some_operation()
        except Exception as e:
            log_exception_to_span(e)
            raise
    """
    add_error_to_span(error)
    log_error_with_context(error)
