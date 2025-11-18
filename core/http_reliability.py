"""
HTTP Reliability Utilities

Provides helpers for robust HTTP communication with:
- Automatic retries with exponential backoff
- Request timeouts
- Circuit breaker pattern
- Error logging
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, TypeVar
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Simple circuit breaker for external service calls"""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """Initialize circuit breaker
        
        Args:
            name: Service name for logging
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying again
            success_threshold: Successes in half-open before closing
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
    
    async def call(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """Execute function with circuit breaker protection"""
        
        # Check if we should try to recover
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                logger.warning(f"Circuit breaker '{self.name}' is OPEN - rejecting request")
                return None
        
        # Try to execute
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success
            self.failure_count = 0
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            logger.warning(f"Circuit breaker '{self.name}' failure #{self.failure_count}: {e}")
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker '{self.name}' is now OPEN")
            
            return None
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery"""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout


# Global circuit breakers for common services
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get or create circuit breaker for service"""
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = CircuitBreaker(service_name)
    return _circuit_breakers[service_name]


async def http_call_with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    timeout: float = 10.0,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **kwargs
) -> Optional[Any]:
    """Execute HTTP call with automatic retries and circuit breaker
    
    Args:
        func: Async function to call
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier (2.0 = double each time)
        timeout: Request timeout in seconds
        circuit_breaker: Optional circuit breaker to use
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Result from function or None if all retries failed
    """
    
    # Use circuit breaker if provided
    if circuit_breaker:
        return await circuit_breaker.call(func, *args, **kwargs)
    
    # Standard retry logic
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # Add timeout to kwargs if not present
            if 'timeout' not in kwargs:
                kwargs['timeout'] = timeout
            
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout
            )
            return result
        
        except asyncio.TimeoutError as e:
            last_exception = e
            logger.warning(f"HTTP call timeout (attempt {attempt + 1}/{max_retries + 1}): {e}")
        
        except Exception as e:
            last_exception = e
            logger.warning(f"HTTP call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
        
        # Wait before retry (exponential backoff)
        if attempt < max_retries:
            wait_time = backoff_factor ** attempt
            logger.info(f"Retrying in {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
    
    # All retries failed
    logger.error(f"HTTP call failed after {max_retries + 1} attempts: {last_exception}")
    return None


# Decorator for easy use in async functions
def with_http_reliability(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    timeout: float = 10.0,
    circuit_breaker_name: Optional[str] = None
):
    """Decorator to add HTTP reliability to async functions
    
    Usage:
        @with_http_reliability(max_retries=3, timeout=5.0)
        async def call_external_api():
            return await session.get("http://api.example.com/data")
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cb = None
            if circuit_breaker_name:
                cb = get_circuit_breaker(circuit_breaker_name)
            
            return await http_call_with_retry(
                func,
                *args,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                timeout=timeout,
                circuit_breaker=cb,
                **kwargs
            )
        return wrapper
    return decorator

