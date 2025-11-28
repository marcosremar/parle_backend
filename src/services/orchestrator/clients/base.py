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
    from src.services.orchestrator.utils.exceptions import is_retryable
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
        self.base_url = base_url or self._get_service_url(service_name)
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_module_service = is_module_service
        
        # Check if running in monolith mode
        self.monolith_mode = os.getenv("MONOLITH_MODE", "false").lower() == "true"
        self.direct_module = None
        
        # In monolith mode, try to get direct module
        if self.monolith_mode:
            try:
                from src.modules import create
                self.direct_module = create(service_name)
                logger.debug(f"✅ {service_name} client using direct module (monolith mode)")
            except Exception as e:
                logger.warning(f"⚠️  Could not create direct module for {service_name}: {e}, falling back to HTTP")
                self.monolith_mode = False
        
        # Retry configuration
        self.max_retries = max_retries or int(os.getenv(f"{service_name.upper()}_MAX_RETRIES", "3"))
        self.base_backoff = base_backoff or float(os.getenv(f"{service_name.upper()}_BASE_BACKOFF", "1.0"))
        
        # Timeout configuration
        if timeout is not None:
            self.default_timeout = timeout
        else:
            try:
                from config.settings import get_settings
                settings = get_settings()
                timeout_map = {
                    "stt": settings.timeouts.stt,
                    "llm": settings.timeouts.llm,
                    "tts": settings.timeouts.tts,
                    "session": settings.timeouts.session,
                    "conversation_store": settings.timeouts.conversation_store,
                    "scenarios": settings.timeouts.scenarios,
                    "file_storage": settings.timeouts.file_storage,
                    "database": settings.timeouts.database,
                    "websocket": settings.timeouts.websocket,
                    "webrtc": settings.timeouts.webrtc,
                }
                self.default_timeout = timeout_map.get(service_name, settings.timeouts.default)
            except Exception:
                env_timeout = os.getenv(f"{service_name.upper()}_TIMEOUT")
                self.default_timeout = float(env_timeout) if env_timeout else 30.0
        
        # Circuit breaker configuration
        self.use_circuit_breaker = use_circuit_breaker
        self.circuit_breaker = None
        if self.use_circuit_breaker:
            try:
                from src.core.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
                try:
                    from config.settings import get_settings
                    settings = get_settings()
                    cb_config = CircuitBreakerConfig(
                        failure_threshold=3,
                        recovery_timeout=settings.pipeline_failover.recovery_timeout,
                        timeout=int(self.default_timeout)
                    )
                except Exception:
                    cb_config = CircuitBreakerConfig(timeout=int(self.default_timeout))
                
                self.circuit_breaker = get_circuit_breaker(service_name, cb_config)
            except ImportError:
                logger.warning(f"Circuit breaker not available for {service_name}")
                self.use_circuit_breaker = False

    def _get_service_url(self, service_name: str) -> str:
        """Get service URL from environment or default ports."""
        env_var = f"{service_name.upper()}_SERVICE_URL"
        default_ports: Dict[str, str] = {
            "stt": "http://localhost:8099",
            "tts": "http://localhost:8103",
            "llm": "http://localhost:8110",
            "orchestrator": "http://localhost:8500",
            "conversation_store": "http://localhost:8800",
            "conversation_history": "http://localhost:8501",
            "session": "http://localhost:8600",
            "scenarios": "http://localhost:8700",
        }
        return os.getenv(env_var, default_ports.get(service_name, f"http://localhost:8000"))

    async def initialize(self, session: aiohttp.ClientSession) -> None:
        """Initialize with shared aiohttp session"""
        self.session = session
        logger.info(f"✅ {self.service_name} client initialized with HTTP (max_retries={self.max_retries})")

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
            from src.services.orchestrator.utils.exceptions import wrap_exception, ServiceUnavailableError
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
            try:
                from config.settings import get_settings
                settings = get_settings()
                health_timeout = settings.timeouts.health_check
            except Exception:
                health_timeout = 2.0
            
            result = await self._get("/health", timeout=health_timeout)
            return result.get("status") in ["healthy", "ok", "running"]
        except Exception as e:
            logger.debug(f"HTTP health check failed for {self.service_name}: {e}")
            return False
