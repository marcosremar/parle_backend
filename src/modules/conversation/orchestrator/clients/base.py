"""
Base Service Client

Common functionality for all service clients.
"""

from __future__ import annotations

import aiohttp
import asyncio
import logging
import os
import random
from typing import Dict, Any, Optional, Callable
from enum import Enum
from pathlib import Path
import sys

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
    REALTIME = "realtime"      # WebRTC, low-latency required
    NORMAL = "normal"          # Standard API calls
    DEBUG = "debug"            # Testing/debugging, prefer JSON


class ServiceClientError(Exception):
    """Base exception for service client errors"""
    pass


class BaseServiceClient:
    """Base HTTP client with common functionality using direct HTTP calls or direct module calls"""

    def __init__(
        self,
        service_name: str,
        base_url: Optional[str] = None,
        is_module_service: bool = False,
        max_retries: Optional[int] = None,
        base_backoff: Optional[float] = None,
        timeout: Optional[float] = None,
        use_circuit_breaker: bool = True
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
        # Only get URL for external services (module services use direct calls)
        service_url = base_url or self._get_service_url(service_name) if not is_module_service else None
        self.base_url = service_url or ""
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_module_service = is_module_service
        
        # For module services, always use direct calls (no HTTP needed)
        self.direct_module = None
        self._module_initialized = False
        
        if self.is_module_service:
            try:
                from src.modules import module_factory
                self.direct_module = module_factory.create(service_name)
                logger.debug(f"✅ {service_name} client will use direct module calls (no HTTP)")
            except Exception as e:
                logger.warning(f"⚠️  Could not create direct module for {service_name}: {e}, will use HTTP fallback")
                self.direct_module = None
        
        # Retry configuration
        self.max_retries = max_retries or int(os.getenv(f"{service_name.upper()}_MAX_RETRIES", "3"))
        self.base_backoff = base_backoff or float(os.getenv(f"{service_name.upper()}_BASE_BACKOFF", "1.0"))
        
        # Timeout configuration
        if timeout is not None:
            self.default_timeout = timeout
        else:
            try:
                from src.core.config import get_config
                config = get_config()
                # Use default timeout from config or environment
                env_timeout = os.getenv(f"{service_name.upper()}_TIMEOUT")
                self.default_timeout = float(env_timeout) if env_timeout else 30.0
            except Exception:
                env_timeout = os.getenv(f"{service_name.upper()}_TIMEOUT")
                self.default_timeout = float(env_timeout) if env_timeout else 30.0
        
        # Circuit breaker configuration
        self.use_circuit_breaker = use_circuit_breaker
        self.circuit_breaker = None
        if self.use_circuit_breaker:
            try:
                from src.core.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
                # Use default circuit breaker config
                cb_config = CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=30.0,  # Default recovery timeout
                    timeout=int(self.default_timeout)
                )
                
                self.circuit_breaker = get_circuit_breaker(service_name, cb_config)
            except ImportError:
                logger.warning(f"Circuit breaker not available for {service_name}")
                self.use_circuit_breaker = False

    def _get_service_url(self, service_name: str) -> Optional[str]:
        """Get service URL from environment (only for external services)"""
        if self.is_module_service:
            # Module services use direct calls, no URL needed
            return None
        
        env_var = f"{service_name.upper()}_SERVICE_URL"
        # Only external services need URLs (websocket, webrtc, etc.)
        external_defaults: Dict[str, str] = {
            "conversation_store": "http://localhost:8800",  # May be external
            "conversation_history": "http://localhost:8501",  # May be external
            "websocket": "http://localhost:8022",
            "webrtc": "http://localhost:8090",
            "rest_polling": "http://localhost:8701",
        }
        return os.getenv(env_var, external_defaults.get(service_name, f"http://localhost:8000"))

    async def initialize(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        """Initialize with shared aiohttp session (only needed for HTTP services)"""
        # Module services use direct calls, no HTTP session needed
        if self.is_module_service:
            return
        
        if not self.is_module_service:
            if session is None:
                raise ValueError(f"{self.service_name} requires HTTP session (not a module service)")
            self.session = session
            logger.info(f"✅ {self.service_name} client initialized with HTTP (max_retries={self.max_retries})")
        else:
            # Module services don't need HTTP session
            if self.direct_module:
                logger.info(f"✅ {self.service_name} client initialized with direct module calls")
            else:
                # Fallback: if module creation failed, we might need HTTP
                if session:
                    self.session = session
                    logger.warning(f"⚠️  {self.service_name} using HTTP fallback (module not available)")

    async def _retry_with_backoff(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
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
                backoff = self.base_backoff * (2 ** attempt)
                jitter = random.uniform(0, backoff * 0.1)  # 10% jitter
                wait_time = backoff + jitter
                
                logger.warning(
                    f"⚠️ {self.service_name} {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )
                
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        logger.error(f"❌ {self.service_name} {operation_name} failed after {self.max_retries + 1} attempts: {last_error}")
        try:
            from ..utils.exceptions import wrap_exception
            wrapped_error = wrap_exception(last_error, service_name=self.service_name, operation=operation_name)
            raise wrapped_error
        except ImportError:
            raise ServiceClientError(
                f"{self.service_name} {operation_name} failed after {self.max_retries + 1} attempts: {last_error}"
            ) from last_error

    async def _get(self, path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Generic GET request using direct HTTP with retry logic"""
        if not self.session:
            raise ServiceClientError(f"{self.service_name}: Session not initialized")

        url = f"{self.base_url}{path}"
        timeout_value = timeout if timeout is not None else self.default_timeout
        
        async def _do_get():
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_value)) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=error_text
                        )
                    raise ServiceClientError(f"{self.service_name} GET {path} failed ({resp.status}): {error_text}")
        
        return await self._retry_with_backoff(_do_get, f"GET {path}")

    async def _post(
        self,
        path: str,
        data: Any = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """Generic POST request using direct HTTP with retry logic"""
        if not self.session:
            raise ServiceClientError(f"{self.service_name}: Session not initialized")

        url = f"{self.base_url}{path}"
        timeout_value = timeout if timeout is not None else self.default_timeout
        
        async def _do_post():
            async with self.session.post(
                url,
                data=data,
                json=json_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_value)
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
                            message=error_text
                        )
                    raise ServiceClientError(f"{self.service_name} POST {path} failed ({resp.status}): {error_text}")
        
        return await self._retry_with_backoff(_do_post, f"POST {path}")

    async def _put(self, path: str, json_data: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Generic PUT request with retry logic"""
        if not self.session:
            raise ServiceClientError(f"{self.service_name}: Session not initialized")

        url = f"{self.base_url}{path}"
        timeout_value = timeout if timeout is not None else self.default_timeout
        
        async def _do_put():
            async with self.session.put(
                url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=timeout_value)
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
                            message=error_text
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
            health_timeout = float(os.getenv(f"{self.service_name.upper()}_HEALTH_TIMEOUT", "2.0"))
            
            result = await self._get("/health", timeout=health_timeout)
            return result.get("status") in ["healthy", "ok", "running"]
        except Exception as e:
            logger.debug(f"HTTP health check failed for {self.service_name}: {e}")
            return False
