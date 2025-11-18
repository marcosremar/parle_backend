"""
Error Handling Module - Consolidated error handling for Ultravox Pipeline.

This module provides:
- Circuit breaker pattern for resilience
- Decorators for automatic error handling
- FastAPI middleware for centralized error responses
- OpenTelemetry integration for error tracking

Public API:
    from src.core.error_handling import (
        # Circuit Breaker
        CircuitBreaker,
        CircuitBreakerState,

        # Decorators
        with_circuit_breaker,
        with_retry,
        with_timeout,
        handle_errors,

        # Middleware
        ErrorHandlerMiddleware,
        setup_error_handling,

        # Telemetry
        add_error_to_span,
        log_error_with_context
    )

Usage Example:
    from src.core.error_handling import with_retry, with_circuit_breaker
    from src.core.exceptions import ServiceUnavailableError

    @with_circuit_breaker(name="external_api", failure_threshold=5)
    @with_retry(max_attempts=3, backoff_ms=1000)
    async def call_external_api(data: dict) -> dict:
        # Your code here
        pass

Note:
    All exceptions should be imported from src.core.exceptions, not from here.
    This module only provides error handling utilities, not the exceptions themselves.
"""

# Circuit Breaker
from .circuit_breaker import CircuitBreaker, CircuitBreakerState

# Decorators
from .decorators import (
    with_circuit_breaker,
    with_retry,
    with_timeout,
    handle_errors,
    validate_input,
)

# Middleware
from .middleware import ErrorHandlerMiddleware, setup_error_handling

# Telemetry integration
from .telemetry import add_error_to_span, log_error_with_context

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    # Decorators
    "with_circuit_breaker",
    "with_retry",
    "with_timeout",
    "handle_errors",
    "validate_input",
    # Middleware
    "ErrorHandlerMiddleware",
    "setup_error_handling",
    # Telemetry
    "add_error_to_span",
    "log_error_with_context",
]
