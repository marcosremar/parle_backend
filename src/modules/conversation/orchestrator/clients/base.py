"""
Base Service Client

Common functionality for all service clients.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import Enum
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any

import aiohttp

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import exception utilities for retry logic
try:
    from ..utils.exceptions import is_retryable
except ImportError:

    def is_retryable(error: Exception) -> bool:
        return isinstance(error, (ConnectionError, TimeoutError, aiohttp.ClientError))


logger = logging.getLogger(__name__)


class Priority(str, Enum):
    """Request priority levels"""

    REALTIME = "realtime"  # WebRTC, low-latency required
    NORMAL = "normal"  # Standard API calls
    DEBUG = "debug"  # Testing/debugging, prefer JSON


class ServiceClientError(Exception):
    """Base exception for service client errors"""


class BaseServiceClient:
    """Base HTTP client with common functionality using direct HTTP calls or direct module calls"""

    def __init__(
        self,
        service_name: str,
        base_url: str | None = None,
        is_module_service: bool = False,
        max_retries: int | None = None,
        base_backoff: float | None = None,
        timeout: float | None = None,
        use_circuit_breaker: bool = True,
    ) -> None:
        """
        Initialize base service client.

        Args:
            service_name: Name of the service
            base_url: Optional base URL override
            is_module_service: Whether service runs in-process
            max_retries: Maximum retry attempts
            base_backoff: Base backoff time in seconds
            timeout: Request timeout in seconds
            use_circuit_breaker: Whether to use circuit breaker
        """
        self.service_name = service_name
        self.is_module_service = is_module_service  # Set this first before calling _get_service_url
        # Only get URL for external services (module services use direct calls)
        service_url = (
            base_url or self._get_service_url(service_name) if not is_module_service else None
        )
        self.base_url = service_url or ""
        self.session: aiohttp.ClientSession | None = None

        # For module services, always use direct calls (no HTTP needed)
        self.direct_module = None
        self._module_initialized = False

        if self.is_module_service:
            try:
                from src.modules import module_factory

                self.direct_module = module_factory.create(service_name)
                logger.debug(f"✅ {service_name} client will use direct module calls (no HTTP)")
            except Exception as e:
                logger.warning(
                    f"⚠️  Could not create direct module for {service_name}: {e}, will use HTTP fallback"
                )
                self.direct_module = None

        # Import constants once at the top of initialization
        from ..constants import (
            DEFAULT_BASE_BACKOFF,
            DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            DEFAULT_MAX_RETRIES,
            DEFAULT_TIMEOUT,
        )

        # Retry configuration
        self.max_retries = max_retries or int(
            os.getenv(f"{service_name.upper()}_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
        )
        self.base_backoff = base_backoff or float(
            os.getenv(f"{service_name.upper()}_BASE_BACKOFF", str(DEFAULT_BASE_BACKOFF))
        )

        # Timeout configuration
        if timeout is not None:
            self.default_timeout = timeout
        else:
            try:
                # Use default timeout from config or environment
                env_timeout = os.getenv(f"{service_name.upper()}_TIMEOUT")
                self.default_timeout = float(env_timeout) if env_timeout else DEFAULT_TIMEOUT
            except Exception:
                env_timeout = os.getenv(f"{service_name.upper()}_TIMEOUT")
                self.default_timeout = float(env_timeout) if env_timeout else DEFAULT_TIMEOUT

        # Circuit breaker configuration
        self.use_circuit_breaker = use_circuit_breaker
        self.circuit_breaker = None
        if self.use_circuit_breaker:
            try:
                from src.core.circuit_breaker import CircuitBreakerConfig, get_circuit_breaker

                # Use default circuit breaker config
                cb_config = CircuitBreakerConfig(
                    failure_threshold=DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                    recovery_timeout=DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                    timeout=int(self.default_timeout),
                )

                self.circuit_breaker = get_circuit_breaker(service_name, cb_config)
            except ImportError:
                logger.warning(f"Circuit breaker not available for {service_name}")
                self.use_circuit_breaker = False

    async def _ensure_module_initialized(self) -> None:
        """
        Helper method to ensure module is initialized (lazy initialization).
        Reduces code duplication across clients.

        Raises:
            ServiceClientError: If module initialization fails
        """
        if not self.direct_module:
            raise ServiceClientError(
                f"{self.service_name} module not available (is_module_service={self.is_module_service})"
            )

        if self._module_initialized:
            return

        if hasattr(self.direct_module, "initialize"):
            try:
                await self.direct_module.initialize()
                self._module_initialized = True
                logger.debug(f"✅ {self.service_name} module initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {self.service_name} module: {e}")
                raise ServiceClientError(
                    f"{self.service_name} module initialization failed: {e}"
                ) from e

    def _get_service_url(self, service_name: str) -> str | None:
        """Get service URL from environment (only for external services)"""
        if self.is_module_service:
            # Module services use direct calls, no URL needed
            return None

        env_var = f"{service_name.upper()}_SERVICE_URL"
        # Only external services need URLs (websocket, webrtc, etc.)
        external_defaults: dict[str, str] = {
            "conversation_store": "http://localhost:8800",  # May be external
            "conversation_history": "http://localhost:8501",  # May be external
            "websocket": "http://localhost:8022",
            "webrtc": "http://localhost:8090",
            "rest_polling": "http://localhost:8701",
        }
        return os.getenv(env_var, external_defaults.get(service_name, "http://localhost:8000"))

    async def initialize(self, session: aiohttp.ClientSession | None = None) -> None:
        """Initialize with shared aiohttp session (only needed for HTTP services)"""
        # Module services use direct calls, no HTTP session needed
        if self.is_module_service:
            return

        # HTTP services require session
        if session is None:
            raise ValueError(f"{self.service_name} requires HTTP session (not a module service)")
        self.session = session
        logger.info(
            f"✅ {self.service_name} client initialized with HTTP (max_retries={self.max_retries})"
        )

    async def _retry_with_backoff(
        self, operation: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """
        Retry operation with exponential backoff.

        Args:
            operation: Async function to retry
            operation_name: Name of operation for logging
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of operation

        Raises:
            ServiceClientError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e

                # Don't retry if error is not retryable
                if not is_retryable(e):
                    raise

                # Don't retry on last attempt
                if attempt == self.max_retries:
                    break

                # Exponential backoff with jitter
                backoff = self.base_backoff * (2**attempt)
                jitter = random.uniform(0, backoff * 0.1)  # 10% jitter
                wait_time = backoff + jitter

                logger.warning(
                    f"⚠️ {self.service_name} {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )

                await asyncio.sleep(wait_time)

        # All retries exhausted
        logger.error(
            f"❌ {self.service_name} {operation_name} failed after {self.max_retries + 1} attempts: {last_error}"
        )
        try:
            from ..utils.exceptions import wrap_exception

            wrapped_error = wrap_exception(
                last_error, service_name=self.service_name, operation=operation_name
            )
            raise wrapped_error
        except ImportError:
            raise ServiceClientError(
                f"{self.service_name} {operation_name} failed after {self.max_retries + 1} attempts: {last_error}"
            ) from last_error

    def _require_session(self) -> None:
        """Helper to ensure HTTP session is available"""
        if not self.session:
            raise ServiceClientError(f"{self.service_name}: HTTP session not initialized")

    def _handle_module_error(self, operation: str, error: Exception) -> None:
        """
        Helper to handle module errors consistently.
        Logs error and raises ServiceClientError with context.

        Args:
            operation: Name of the operation that failed
            error: The exception that occurred

        Raises:
            ServiceClientError: Always raises with descriptive message
        """
        logger.error(f"❌ {self.service_name} {operation} failed: {error}")
        raise ServiceClientError(f"{self.service_name} {operation} failed: {error}") from error

    async def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        """Generic GET request using direct HTTP with retry logic"""
        self._require_session()

        url = f"{self.base_url}{path}"
        timeout_value = timeout if timeout is not None else self.default_timeout

        async def _do_get():
            async with self.session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout_value)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=error_text,
                        )
                    raise ServiceClientError(
                        f"{self.service_name} GET {path} failed ({resp.status}): {error_text}"
                    )

        return await self._retry_with_backoff(_do_get, f"GET {path}")

    async def _post(
        self,
        path: str,
        data: Any = None,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Generic POST request using direct HTTP with retry logic"""
        self._require_session()

        url = f"{self.base_url}{path}"
        timeout_value = timeout if timeout is not None else self.default_timeout

        async def _do_post():
            async with self.session.post(
                url,
                data=data,
                json=json_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_value),
            ) as resp:
                if resp.status == 200:
                    try:
                        return await resp.json()
                    except Exception:
                        return await resp.read()
                else:
                    error_text = await resp.text()
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=error_text,
                        )
                    raise ServiceClientError(
                        f"{self.service_name} POST {path} failed ({resp.status}): {error_text}"
                    )

        return await self._retry_with_backoff(_do_post, f"POST {path}")

    async def _put(
        self, path: str, json_data: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        """Generic PUT request with retry logic"""
        self._require_session()

        url = f"{self.base_url}{path}"
        timeout_value = timeout if timeout is not None else self.default_timeout

        async def _do_put():
            async with self.session.put(
                url, json=json_data, timeout=aiohttp.ClientTimeout(total=timeout_value)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=error_text,
                        )
                    raise ServiceClientError(
                        f"{self.service_name} PUT {path} failed ({resp.status}): {error_text}"
                    )

        return await self._retry_with_backoff(_do_put, f"PUT {path}")

    async def health_check(self) -> bool:
        """
        Check if service is healthy via HTTP GET /health check.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Use default health check timeout
            from ..constants import DEFAULT_HEALTH_CHECK_TIMEOUT

            health_timeout = float(
                os.getenv(
                    f"{self.service_name.upper()}_HEALTH_TIMEOUT", str(DEFAULT_HEALTH_CHECK_TIMEOUT)
                )
            )

            result = await self._get("/health", timeout=health_timeout)
            return result.get("status") in ["healthy", "ok", "running"]
        except Exception as e:
            logger.debug(f"HTTP health check failed for {self.service_name}: {e}")
            return False
