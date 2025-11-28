"""
Rate Limiter - Sliding Window Implementation
Provides rate limiting for API Gateway using sliding window algorithm
"""
import time
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger

try:
    from src.services.api_gateway.utils.exceptions import RateLimitError
except ImportError:
    try:
        from src.services.orchestrator.utils.exceptions import RateLimitError
    except ImportError:
        # Fallback exception
        class RateLimitError(Exception):
            def __init__(self, message: str, limit: int = 0, window_seconds: int = 0):
                self.message = message
                self.limit = limit
                self.window_seconds = window_seconds
                super().__init__(message)


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter
    
    Tracks requests in a sliding time window and enforces rate limits
    per identifier (IP address or user ID)
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        window_size_seconds: int = 60
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Max requests per minute per identifier
            requests_per_hour: Max requests per hour per identifier
            window_size_seconds: Size of sliding window in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.window_size_seconds = window_size_seconds
        
        # Store request timestamps per identifier
        # Structure: {identifier: deque([timestamp1, timestamp2, ...])}
        self.request_timestamps: Dict[str, deque] = defaultdict(deque)
        
        # Cleanup old entries periodically
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
        logger.info(
            f"✅ Rate Limiter initialized: {requests_per_minute} req/min, "
            f"{requests_per_hour} req/hour"
        )
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old entries to prevent memory leak"""
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # Keep last hour
        identifiers_to_remove = []
        
        for identifier, timestamps in self.request_timestamps.items():
            # Remove timestamps older than cutoff
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()
            
            # Remove identifier if no timestamps left
            if not timestamps:
                identifiers_to_remove.append(identifier)
        
        for identifier in identifiers_to_remove:
            del self.request_timestamps[identifier]
        
        self.last_cleanup = current_time
    
    def _count_requests_in_window(
        self,
        timestamps: deque,
        current_time: float,
        window_seconds: int
    ) -> int:
        """Count requests in the specified time window"""
        cutoff_time = current_time - window_seconds
        
        # Remove timestamps outside window
        while timestamps and timestamps[0] < cutoff_time:
            timestamps.popleft()
        
        return len(timestamps)
    
    def check_rate_limit(
        self,
        identifier: str,
        current_time: Optional[float] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limits
        
        Args:
            identifier: Unique identifier (IP address or user ID)
            current_time: Current timestamp (for testing)
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
            rate_limit_info contains:
                - remaining: Remaining requests in current window
                - reset_after: Seconds until window resets
                - limit: Current limit being checked
        """
        if current_time is None:
            current_time = time.time()
        
        # Cleanup old entries periodically
        self._cleanup_old_entries(current_time)
        
        # Get or create timestamps deque for this identifier
        timestamps = self.request_timestamps[identifier]
        
        # Check per-minute limit
        requests_in_minute = self._count_requests_in_window(
            timestamps, current_time, 60
        )
        
        if requests_in_minute >= self.requests_per_minute:
            reset_after = 60 - (current_time % 60)
            return False, {
                "remaining": 0,
                "reset_after": int(reset_after),
                "limit": self.requests_per_minute,
                "window": "minute"
            }
        
        # Check per-hour limit
        requests_in_hour = self._count_requests_in_window(
            timestamps, current_time, 3600
        )
        
        if requests_in_hour >= self.requests_per_hour:
            reset_after = 3600 - (current_time % 3600)
            return False, {
                "remaining": 0,
                "reset_after": int(reset_after),
                "limit": self.requests_per_hour,
                "window": "hour"
            }
        
        # Request is allowed - add timestamp
        timestamps.append(current_time)
        
        # Calculate remaining requests
        remaining_minute = max(0, self.requests_per_minute - requests_in_minute - 1)
        remaining_hour = max(0, self.requests_per_hour - requests_in_hour - 1)
        
        # Use the more restrictive limit
        remaining = min(remaining_minute, remaining_hour)
        reset_after = min(60 - (current_time % 60), 3600 - (current_time % 3600))
        
        return True, {
            "remaining": remaining,
            "reset_after": int(reset_after),
            "limit": self.requests_per_minute if remaining == remaining_minute else self.requests_per_hour,
            "window": "minute" if remaining == remaining_minute else "hour"
        }
    
    def get_rate_limit_headers(self, rate_limit_info: Dict[str, int]) -> Dict[str, str]:
        """
        Generate rate limit headers for HTTP response
        
        Args:
            rate_limit_info: Result from check_rate_limit
            
        Returns:
            Dictionary of header names to values
        """
        return {
            "X-RateLimit-Limit": str(rate_limit_info["limit"]),
            "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
            "X-RateLimit-Reset": str(int(time.time()) + rate_limit_info["reset_after"]),
            "X-RateLimit-Window": rate_limit_info["window"]
        }


# Global rate limiter instance
_rate_limiter: Optional[SlidingWindowRateLimiter] = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    
    if _rate_limiter is None:
        import os
        requests_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        requests_per_hour = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
        
        _rate_limiter = SlidingWindowRateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour
        )
    
    return _rate_limiter


async def check_rate_limit_middleware(request, call_next):
    """
    FastAPI middleware for rate limiting
    
    Usage:
        @app.middleware("http")
        async def rate_limit_middleware(request, call_next):
            return await check_rate_limit_middleware(request, call_next)
    """
    from fastapi import Request, Response
    
    # Get identifier (IP address or user ID from token)
    identifier = request.client.host if request.client else "unknown"
    
    # Try to get user ID from JWT token if available
    try:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Extract user ID from token (simplified - in production, verify token)
            # For now, use IP + token hash as identifier
            token = auth_header.split(" ")[1]
            identifier = f"{identifier}:{hash(token) % 10000}"
    except Exception:
        pass  # Fall back to IP address
    
    # Check rate limit
    rate_limiter = get_rate_limiter()
    is_allowed, rate_limit_info = rate_limiter.check_rate_limit(identifier)
    
    if not is_allowed:
        # Rate limit exceeded
        headers = rate_limiter.get_rate_limit_headers(rate_limit_info)
        
        raise RateLimitError(
            f"Rate limit exceeded: {rate_limit_info['limit']} requests per {rate_limit_info['window']}",
            limit=rate_limit_info["limit"],
            window_seconds=60 if rate_limit_info["window"] == "minute" else 3600
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    
    # Add headers
    headers = rate_limiter.get_rate_limit_headers(rate_limit_info)
    for header_name, header_value in headers.items():
        response.headers[header_name] = header_value
    
    return response

