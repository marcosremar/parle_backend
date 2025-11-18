"""
Unified Error Handling System for Ultravox Pipeline (DRY v5.2)

Consolidates:
- src/core/exceptions.py (exception hierarchy)
- src/core/error_handler.py (CircuitBreaker, decorators)

Provides:
- Typed exception hierarchy (ServiceError, CommunicationError, etc.)
- FastAPI exception handlers (auto-convert to HTTP responses)
- Decorators for automatic error handling (@handle_service_errors, @with_circuit_breaker)
- Circuit breaker for cascade failure protection
- Utilities for wrapping generic exceptions

Usage:
    from src.core.error_handling import (
        ServiceUnavailableError,
        handle_service_errors,
        setup_error_handlers,
        wrap_exception
    )

    # In routes.py:
    app = FastAPI()
    setup_error_handlers(app)  # Auto-convert exceptions to HTTP

    # In service methods:
    @handle_service_errors(service_name="llm", operation="inference")
    async def process_request(self, data):
        # Exceptions auto-wrapped and converted to HTTP responses
        result = await external_api_call()
        return result
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Type
from dataclasses import dataclass
from functools import wraps

# Import base exceptions from exceptions.py
from src.core.exceptions import (
    UltravoxError,
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    ServiceInitializationError,
    CommunicationError,
    NetworkError,
    ProtocolError,
    SerializationError,
    ValidationError as UltravoxValidationError,
    RequestValidationError,
    ResponseValidationError,
    ConfigurationError,
    ResourceError,
    GPUNotAvailableError,
    MemoryError as UltravoxMemoryError,
    StorageError,
    AIError,
    LLMError,
    STTError,
    TTSError,
    is_retryable,
    wrap_exception
)

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError as FastAPIValidationError
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Circuit Breaker (from error_handler.py)
# ============================================================================

@dataclass
class CircuitBreakerState:
    """Estado do circuit breaker"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    success_count: int = 0


class CircuitBreaker:
    """
    Circuit Breaker para proteger contra falhas em cascata

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        async with breaker:
            result = await risky_operation()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception,
        name: Optional[str] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "circuit_breaker"
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Context manager entry"""
        async with self._lock:
            if self.state.state == "OPEN":
                if self._should_attempt_reset():
                    self.state.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise ServiceUnavailableError(
                        service_name=self.name,
                        original_error=Exception(f"Circuit breaker {self.name} is OPEN")
                    )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        async with self._lock:
            if exc_type is None:
                # Success
                if self.state.state == "HALF_OPEN":
                    self.state.state = "CLOSED"
                    self.state.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after successful recovery")
                elif self.state.state == "CLOSED":
                    self.state.success_count += 1
            elif issubclass(exc_type, self.expected_exception):
                # Expected failure
                self.state.failure_count += 1
                self.state.last_failure_time = datetime.utcnow()

                if self.state.failure_count >= self.failure_threshold:
                    self.state.state = "OPEN"
                    logger.warning(f"Circuit breaker {self.name} opened after {self.state.failure_count} failures")

        return False  # Don't suppress exceptions

    def _should_attempt_reset(self) -> bool:
        """Verifica se deve tentar reset do circuit breaker"""
        if self.state.last_failure_time is None:
            return True

        time_since_failure = datetime.utcnow() - self.state.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout

    def get_state(self) -> Dict[str, Any]:
        """Retorna estado atual do circuit breaker"""
        return {
            "name": self.name,
            "state": self.state.state,
            "failure_count": self.state.failure_count,
            "success_count": self.state.success_count,
            "last_failure_time": self.state.last_failure_time.isoformat() if self.state.last_failure_time else None,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }


# ============================================================================
# Global Circuit Breaker Registry
# ============================================================================

_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Get or create circuit breaker

    Args:
        name: Circuit breaker name
        **kwargs: CircuitBreaker constructor arguments

    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]


# ============================================================================
# Decorators
# ============================================================================

def with_circuit_breaker(name: str, **cb_kwargs):
    """
    Decorator to apply circuit breaker protection

    Usage:
        @with_circuit_breaker("external_api", failure_threshold=3)
        async def call_external_api():
            return await api_call()
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(name, **cb_kwargs)
            async with breaker:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def handle_service_errors(
    service_name: str,
    operation: Optional[str] = None,
    log_errors: bool = True
):
    """
    Decorator for automatic error handling and wrapping

    Automatically wraps generic exceptions into typed UltravoxError exceptions.
    Works with FastAPI - exceptions are auto-converted to HTTP responses.

    Usage:
        @handle_service_errors(service_name="llm", operation="inference")
        async def process_llm_request(self, prompt: str):
            # Any exception auto-wrapped and logged
            response = await self.llm_client.generate(prompt)
            return response

    Args:
        service_name: Name of the service (for error context)
        operation: Operation being performed (defaults to function name)
        log_errors: Whether to log errors (default: True)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation or func.__name__

            try:
                return await func(*args, **kwargs)
            except UltravoxError:
                # Already typed, re-raise as-is
                raise
            except Exception as e:
                # Wrap generic exception
                wrapped = wrap_exception(e, service_name=service_name, operation=op_name)

                if log_errors:
                    logger.error(
                        f"Error in {service_name}.{op_name}: {wrapped.message}",
                        error_type=type(wrapped).__name__,
                        retryable=wrapped.details.get("retryable", False)
                    )

                raise wrapped from e

        return wrapper
    return decorator


def validate_request(**validators):
    """
    Decorator for request validation

    Usage:
        @validate_request(audio=is_valid_audio, text=is_valid_text)
        async def process_request(self, audio: bytes, text: str):
            # Validation happens automatically before function runs
            pass

    Args:
        **validators: Mapping of parameter name to validator function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Validate kwargs
            for param_name, validator_func in validators.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not validator_func(value):
                        raise RequestValidationError(
                            field=param_name,
                            reason=f"Validation failed for {param_name}"
                        )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# FastAPI Exception Handlers
# ============================================================================

def setup_error_handlers(app: 'FastAPI') -> None:
    """
    Setup FastAPI exception handlers for automatic error conversion

    This eliminates the need for manual try/except HTTPException in routes!
    Just raise typed exceptions and they're auto-converted to HTTP responses.

    Usage:
        from fastapi import FastAPI
        from src.core.error_handling import setup_error_handlers

        app = FastAPI()
        setup_error_handlers(app)  # Add once, works everywhere!

        # In routes - NO try/except needed!
        @app.post("/process")
        async def process(data: dict):
            if not data.get("audio"):
                raise RequestValidationError("audio", "Audio is required")
            # Auto-converted to HTTP 400!

    Args:
        app: FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping error handler setup")
        return

    @app.exception_handler(UltravoxError)
    async def ultravox_error_handler(request: Request, exc: UltravoxError):
        """Convert UltravoxError to HTTP response"""
        status_code = _get_http_status_code(exc)

        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": exc.message,
                    "details": exc.details,
                    "retryable": exc.details.get("retryable", False)
                }
            }
        )

    @app.exception_handler(FastAPIValidationError)
    async def fastapi_validation_error_handler(request: Request, exc: FastAPIValidationError):
        """Convert Pydantic validation errors to consistent format"""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "details": {
                        "errors": exc.errors()
                    },
                    "retryable": False
                }
            }
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        """Fallback handler for unexpected errors"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "details": {},
                    "retryable": False
                }
            }
        )

    logger.info("✅ Error handlers registered with FastAPI")


def _get_http_status_code(exc: UltravoxError) -> int:
    """
    Map UltravoxError to HTTP status code

    Args:
        exc: UltravoxError instance

    Returns:
        HTTP status code (400-599)
    """
    # Validation errors -> 400 Bad Request
    if isinstance(exc, (UltravoxValidationError, RequestValidationError, ConfigurationError)):
        return 400

    # Service unavailable -> 503
    if isinstance(exc, ServiceUnavailableError):
        return 503

    # Timeout -> 504 Gateway Timeout
    if isinstance(exc, ServiceTimeoutError):
        return 504

    # Resource errors -> 507 Insufficient Storage OR 503
    if isinstance(exc, (GPUNotAvailableError, UltravoxMemoryError, StorageError)):
        return 507

    # Network/communication errors -> 502 Bad Gateway
    if isinstance(exc, (NetworkError, ProtocolError)):
        return 502

    # AI service errors -> 500 Internal Server Error
    if isinstance(exc, (LLMError, STTError, TTSError)):
        return 500

    # Default -> 500
    return 500


# ============================================================================
# Validation Utilities
# ============================================================================

def is_valid_audio(audio_data) -> bool:
    """
    Validate audio data

    Supports:
    - numpy arrays (float32, int16)
    - bytes (WAV format)
    - base64 strings

    Args:
        audio_data: Audio data in various formats

    Returns:
        True if valid, False otherwise
    """
    if audio_data is None:
        return False

    # numpy array
    try:
        import numpy as np
        if isinstance(audio_data, np.ndarray):
            return len(audio_data) > 0 and audio_data.dtype in [np.float32, np.int16]
    except ImportError:
        pass

    # bytes (WAV file)
    if isinstance(audio_data, bytes):
        return len(audio_data) > 44  # Minimum WAV header size

    # base64 string
    if isinstance(audio_data, str):
        try:
            import base64
            decoded = base64.b64decode(audio_data)
            return len(decoded) > 44
        except Exception:
            return False

    return False


def is_valid_text(text) -> bool:
    """
    Validate text input

    Args:
        text: Text string

    Returns:
        True if valid, False otherwise
    """
    return isinstance(text, str) and 0 < len(text.strip()) <= 10000


def is_valid_session_id(session_id) -> bool:
    """
    Validate session ID

    Args:
        session_id: Session ID string

    Returns:
        True if valid, False otherwise
    """
    return isinstance(session_id, str) and 0 < len(session_id) <= 128


def is_valid_language(language) -> bool:
    """
    Validate language code

    Args:
        language: ISO 639-1 language code

    Returns:
        True if valid, False otherwise
    """
    valid_languages = {'pt', 'en', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'zh', 'ru'}
    return isinstance(language, str) and language.lower() in valid_languages


# ============================================================================
# Export all public APIs
# ============================================================================

__all__ = [
    # Base exceptions
    'UltravoxError',
    'ServiceError',
    'ServiceUnavailableError',
    'ServiceTimeoutError',
    'ServiceInitializationError',
    'CommunicationError',
    'NetworkError',
    'ProtocolError',
    'SerializationError',
    'UltravoxValidationError',
    'RequestValidationError',
    'ResponseValidationError',
    'ConfigurationError',
    'ResourceError',
    'GPUNotAvailableError',
    'UltravoxMemoryError',
    'StorageError',
    'AIError',
    'LLMError',
    'STTError',
    'TTSError',

    # Circuit breaker
    'CircuitBreaker',
    'CircuitBreakerState',
    'get_circuit_breaker',

    # Decorators
    'with_circuit_breaker',
    'handle_service_errors',
    'validate_request',

    # FastAPI integration
    'setup_error_handlers',

    # Utilities
    'is_retryable',
    'wrap_exception',
    'is_valid_audio',
    'is_valid_text',
    'is_valid_session_id',
    'is_valid_language',
]
