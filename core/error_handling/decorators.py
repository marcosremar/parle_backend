"""
Error Handling Decorators - Automatic error handling, retries, and circuit breaking.

Provides decorators for common error handling patterns:
- @with_circuit_breaker: Circuit breaker pattern
- @with_retry: Automatic retries with exponential backoff
- @with_timeout: Timeout enforcement
- @handle_errors: Automatic exception conversion and logging
- @validate_input: Input validation

Usage:
    from src.core.error_handling.decorators import (
        with_circuit_breaker,
        with_retry,
        with_timeout,
        handle_errors
    )

    @handle_errors(component="llm_service")
    @with_circuit_breaker(name="llm_api", failure_threshold=5)
    @with_retry(max_attempts=3, backoff_ms=1000)
    @with_timeout(timeout_ms=5000)
    async def call_llm(prompt: str) -> str:
        # Your code here
        pass
"""

import asyncio
import functools
import time
from typing import Callable, Optional, Any, TypeVar, Type
from loguru import logger

from src.core.exceptions import (
    UltravoxError,
    ServiceTimeoutError,
    wrap_exception,
    is_retryable,
)
from .circuit_breaker import get_circuit_breaker

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    success_threshold: int = 2,
    expected_exception: Type[Exception] = Exception,
) -> Callable[[F], F]:
    """
    Decorator to apply circuit breaker pattern to a function.

    Prevents cascading failures by stopping requests to failing services.

    Args:
        name: Circuit breaker identifier (shared across all decorators with same name)
        failure_threshold: Number of failures before opening circuit (default: 5)
        recovery_timeout: Seconds to wait before testing recovery (default: 60)
        success_threshold: Successes needed in HALF_OPEN to close (default: 2)
        expected_exception: Exception type to monitor (default: Exception)

    Returns:
        Decorated function

    Example:
        @with_circuit_breaker(name="llm_service", failure_threshold=5)
        async def call_llm(prompt: str) -> str:
            return await llm_client.generate(prompt)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            circuit_breaker = get_circuit_breaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold,
                expected_exception=expected_exception,
            )

            async with circuit_breaker:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def with_retry(
    max_attempts: int = 3,
    backoff_ms: int = 1000,
    max_backoff_ms: int = 30000,
    exponential: bool = True,
    retry_on: Optional[tuple[Type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """
    Decorator to add automatic retry logic with exponential backoff.

    Retries failed operations with configurable backoff strategy.
    Only retries exceptions that are marked as retryable.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff_ms: Initial backoff time in milliseconds (default: 1000)
        max_backoff_ms: Maximum backoff time in milliseconds (default: 30000)
        exponential: Use exponential backoff (default: True)
        retry_on: Tuple of exception types to retry (default: retryable exceptions)

    Returns:
        Decorated function

    Example:
        @with_retry(max_attempts=3, backoff_ms=1000)
        async def call_api(url: str) -> dict:
            return await http_client.get(url)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception: Optional[Exception] = None
            current_backoff = backoff_ms

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    should_retry = False
                    if retry_on:
                        should_retry = isinstance(e, retry_on)
                    else:
                        # Use is_retryable from exceptions module
                        should_retry = is_retryable(e)

                    if not should_retry or attempt >= max_attempts:
                        # Don't retry or max attempts reached
                        logger.error(
                            f"❌ {func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise

                    # Calculate backoff
                    if exponential:
                        current_backoff = min(backoff_ms * (2 ** (attempt - 1)), max_backoff_ms)
                    else:
                        current_backoff = backoff_ms

                    logger.warning(
                        f"⚠️  {func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_backoff}ms..."
                    )

                    # Wait before retry
                    await asyncio.sleep(current_backoff / 1000)

            # Should never reach here, but just in case
            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


def with_timeout(timeout_ms: int) -> Callable[[F], F]:
    """
    Decorator to enforce timeout on async functions.

    Raises ServiceTimeoutError if function exceeds timeout.

    Args:
        timeout_ms: Timeout in milliseconds

    Returns:
        Decorated function

    Example:
        @with_timeout(timeout_ms=5000)
        async def slow_operation() -> str:
            await asyncio.sleep(10)  # Will timeout
            return "Done"
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_ms / 1000
                )
            except asyncio.TimeoutError as e:
                logger.error(f"⏱️  {func.__name__} timed out after {timeout_ms}ms")
                raise ServiceTimeoutError(
                    service_name=func.__name__, timeout_ms=timeout_ms, original_error=e
                )

        return wrapper  # type: ignore

    return decorator


def handle_errors(
    component: str,
    operation: Optional[str] = None,
    wrap_exceptions: bool = True,
    log_errors: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for centralized error handling.

    Automatically wraps exceptions into UltravoxError hierarchy,
    adds context, and logs errors.

    Args:
        component: Component name for error context
        operation: Operation name (defaults to function name)
        wrap_exceptions: Wrap non-UltravoxError exceptions (default: True)
        log_errors: Log errors automatically (default: True)

    Returns:
        Decorated function

    Example:
        @handle_errors(component="orchestrator", operation="process_audio")
        async def process_audio(audio_data: bytes) -> str:
            return await transcribe(audio_data)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation or func.__name__

            try:
                return await func(*args, **kwargs)

            except UltravoxError:
                # Already an UltravoxError, just log and re-raise
                if log_errors:
                    logger.error(f"❌ Error in {component}.{op_name}")
                raise

            except Exception as e:
                # Wrap in UltravoxError
                if log_errors:
                    logger.error(
                        f"❌ Unexpected error in {component}.{op_name}: {e}",
                        exc_info=True,
                    )

                if wrap_exceptions:
                    wrapped = wrap_exception(e, service_name=component, operation=op_name)
                    raise wrapped
                else:
                    raise

        return wrapper  # type: ignore

    return decorator


def validate_input(
    validation_func: Callable[[Any], bool],
    error_message: str,
    check_args: bool = True,
    check_kwargs: Optional[tuple[str, ...]] = None,
) -> Callable[[F], F]:
    """
    Decorator for input validation.

    Validates function arguments using a custom validation function.

    Args:
        validation_func: Function that returns True if input is valid
        error_message: Error message if validation fails
        check_args: Validate positional arguments (default: True)
        check_kwargs: Tuple of kwarg keys to validate (default: None = all)

    Returns:
        Decorated function

    Example:
        def is_valid_audio(data):
            return isinstance(data, bytes) and len(data) > 0

        @validate_input(is_valid_audio, "Invalid audio data")
        async def process_audio(audio: bytes) -> str:
            return await transcribe(audio)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from src.core.exceptions import RequestValidationError

            # Validate args
            if check_args:
                for i, arg in enumerate(args):
                    if not validation_func(arg):
                        raise RequestValidationError(
                            field=f"arg[{i}]", reason=error_message
                        )

            # Validate kwargs
            if check_kwargs:
                for key in check_kwargs:
                    if key in kwargs and not validation_func(kwargs[key]):
                        raise RequestValidationError(field=key, reason=error_message)
            elif check_kwargs is None:
                # Check all kwargs
                for key, value in kwargs.items():
                    if not validation_func(value):
                        raise RequestValidationError(field=key, reason=error_message)

            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Convenience decorators for common validations
def validate_audio(func: F) -> F:
    """
    Decorator to validate audio data.

    Example:
        @validate_audio
        async def process_audio(audio: bytes) -> str:
            return await transcribe(audio)
    """

    def is_valid_audio(data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, bytes):
            return len(data) > 44  # Minimum WAV header
        # Add numpy array check if needed
        return False

    return validate_input(
        validation_func=is_valid_audio,
        error_message="Invalid audio data",
        check_args=True,
    )(func)


def validate_text(max_length: int = 10000) -> Callable[[F], F]:
    """
    Decorator to validate text input.

    Args:
        max_length: Maximum text length (default: 10000)

    Example:
        @validate_text(max_length=5000)
        async def process_text(text: str) -> str:
            return await generate_response(text)
    """

    def is_valid_text(data: Any) -> bool:
        return isinstance(data, str) and 0 < len(data.strip()) <= max_length

    def decorator(func: F) -> F:
        return validate_input(
            validation_func=is_valid_text,
            error_message=f"Invalid text (must be 1-{max_length} chars)",
            check_args=True,
        )(func)

    return decorator


def validate_session_id(func: F) -> F:
    """
    Decorator to validate session ID.

    Example:
        @validate_session_id
        async def get_session(session_id: str) -> dict:
            return await load_session(session_id)
    """

    def is_valid_session_id(data: Any) -> bool:
        return isinstance(data, str) and 0 < len(data) <= 128

    return validate_input(
        validation_func=is_valid_session_id,
        error_message="Invalid session ID",
        check_args=True,
    )(func)


# Stacking decorators example:
"""
Best practice for stacking decorators (order matters!):

@handle_errors(component="my_service")           # ← Outermost: catches all errors
@with_circuit_breaker(name="external_api")       # ← Circuit breaker
@with_retry(max_attempts=3)                      # ← Retry logic
@with_timeout(timeout_ms=5000)                   # ← Timeout enforcement
@validate_audio                                  # ← Input validation
async def process_audio(audio: bytes) -> str:   # ← Innermost: your function
    return await external_api.transcribe(audio)

Order explanation:
1. Input validation happens first (fail fast)
2. Timeout wraps the actual function call
3. Retry wraps timeout (retries on timeout)
4. Circuit breaker wraps retry (prevents retry if circuit is open)
5. Error handling wraps everything (catches and logs all errors)
"""
