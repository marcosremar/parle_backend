"""
Base Controller
Abstract base class for all controllers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter"""
    enabled: bool = False
    max_requests: int = 100  # Max requests per window
    window_seconds: int = 60  # Time window in seconds


class RateLimiter:
    """Simple in-memory rate limiter using sliding window"""

    def __init__(self, config: RateLimiterConfig):
        self.config = config
        self.requests: Dict[str, deque] = {}  # session_id -> deque of timestamps

    def check_rate_limit(self, session_id: str) -> Dict[str, Any]:
        """
        Check if request should be rate limited

        Args:
            session_id: Session/user identifier

        Returns:
            Dict with 'allowed' bool and optional 'retry_after' seconds
        """
        if not self.config.enabled:
            return {"allowed": True}

        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.window_seconds)

        # Initialize request queue for this session
        if session_id not in self.requests:
            self.requests[session_id] = deque()

        # Remove old requests outside the window
        request_queue = self.requests[session_id]
        while request_queue and request_queue[0] < window_start:
            request_queue.popleft()

        # Check if limit exceeded
        if len(request_queue) >= self.config.max_requests:
            # Calculate retry_after (when oldest request will expire)
            oldest = request_queue[0]
            retry_after = (oldest + timedelta(seconds=self.config.window_seconds) - now).total_seconds()

            return {
                "allowed": False,
                "retry_after": max(1, int(retry_after)),
                "current_count": len(request_queue),
                "max_requests": self.config.max_requests,
                "window_seconds": self.config.window_seconds
            }

        # Add current request
        request_queue.append(now)

        return {
            "allowed": True,
            "current_count": len(request_queue),
            "max_requests": self.config.max_requests,
            "window_seconds": self.config.window_seconds
        }


class BaseController(ABC):
    """
    Base controller class with common functionality
    All controllers should inherit from this class
    """

    def __init__(self, rate_limiter_config: Optional[RateLimiterConfig] = None):
        """
        Initialize base controller

        Args:
            rate_limiter_config: Optional rate limiter configuration
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = {}

        # Initialize rate limiter (disabled by default)
        if rate_limiter_config is None:
            rate_limiter_config = RateLimiterConfig(enabled=False)
        self.rate_limiter = RateLimiter(rate_limiter_config)

    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for handling requests
        Implements common logic like validation, metrics, error handling, rate limiting

        Args:
            request_data: Incoming request data

        Returns:
            Response dictionary
        """
        start_time = time.time()
        request_id = request_data.get('request_id', f"req_{int(time.time()*1000)}")

        try:
            # Log incoming request
            self.logger.info(f"📥 Request {request_id}: {request_data.get('type', 'unknown')}")

            # Check rate limit
            session_id = request_data.get('session_id', 'default')
            rate_check = self.rate_limiter.check_rate_limit(session_id)

            if not rate_check["allowed"]:
                self.logger.warning(
                    f"🚫 Rate limit exceeded for session {session_id}: "
                    f"{rate_check['current_count']}/{rate_check['max_requests']} "
                    f"requests in {rate_check['window_seconds']}s window"
                )
                return {
                    'success': False,
                    'error': (
                        f"Rate limit exceeded. "
                        f"Max {rate_check['max_requests']} requests per {rate_check['window_seconds']}s. "
                        f"Retry after {rate_check['retry_after']}s."
                    ),
                    'rate_limit': rate_check,
                    'request_id': request_id,
                    'metrics': {
                        'controller_ms': (time.time() - start_time) * 1000
                    }
                }

            # Validate request
            validation_result = await self.validate_request(request_data)
            if not validation_result['valid']:
                return self._error_response(
                    error=validation_result['error'],
                    request_id=request_id
                )

            # Process request (implemented by child classes)
            result = await self.process_request(request_data)

            # Add metrics
            elapsed_ms = (time.time() - start_time) * 1000
            result['metrics'] = {
                **result.get('metrics', {}),
                'controller_ms': elapsed_ms,
                'request_id': request_id
            }

            # Log success
            self.logger.info(f"✅ Request {request_id} completed in {elapsed_ms:.0f}ms")

            return result

        except Exception as e:
            self.logger.error(f"❌ Request {request_id} failed: {e}")
            return self._error_response(
                error=str(e),
                request_id=request_id,
                elapsed_ms=(time.time() - start_time) * 1000
            )

    @abstractmethod
    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate incoming request
        Must be implemented by child classes

        Args:
            request_data: Request to validate

        Returns:
            Dictionary with 'valid' boolean and optional 'error' message
        """
        pass

    @abstractmethod
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the validated request
        Must be implemented by child classes

        Args:
            request_data: Validated request data

        Returns:
            Response dictionary
        """
        pass

    def _error_response(self,
                       error: str,
                       request_id: str,
                       elapsed_ms: float = 0) -> Dict[str, Any]:
        """
        Create standardized error response

        Args:
            error: Error message
            request_id: Request identifier
            elapsed_ms: Time taken before error

        Returns:
            Error response dictionary
        """
        return {
            'success': False,
            'error': error,
            'request_id': request_id,
            'metrics': {
                'controller_ms': elapsed_ms
            }
        }

    def _success_response(self,
                         data: Any,
                         request_id: str,
                         metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create standardized success response

        Args:
            data: Response data
            request_id: Request identifier
            metrics: Optional performance metrics

        Returns:
            Success response dictionary
        """
        return {
            'success': True,
            'data': data,
            'request_id': request_id,
            'metrics': metrics or {}
        }