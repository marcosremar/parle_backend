"""
Generic Circuit Breaker for Service Resilience
Can be used for any service (STT, TTS, LLM, etc.)
"""
import time
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Circuit open - using fallback or failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 3       # Failures before opening circuit
    recovery_timeout: int = 30       # Seconds before retry
    half_open_max_calls: int = 1     # Test calls in half-open state
    success_threshold: int = 1       # Successes needed to close from half-open
    timeout: int = 10                # Timeout for service calls


class GenericCircuitBreaker:
    """
    Generic circuit breaker for any service
    
    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, failing fast
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(self, service_name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker
        
        Args:
            service_name: Name of the service (for logging)
            config: Circuit breaker configuration
        """
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.success_count_in_half_open = 0
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0

        logger.info(
            f"🔌 Circuit breaker initialized for {service_name}: "
            f"threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout}s"
        )

    async def call(
        self,
        service_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call service function with circuit breaker protection
        
        Args:
            service_fn: Service function to call
            *args, **kwargs: Arguments to pass to service function
            
        Returns:
            Service result
            
        Raises:
            Exception if circuit is open or service fails
        """
        self.total_calls += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count_in_half_open = 0
                logger.info(f"🔄 Circuit breaker HALF_OPEN for {self.service_name} - testing recovery")
            else:
                # Circuit still open, fail fast
                time_until_retry = self._time_until_retry()
                logger.warning(
                    f"🚨 Circuit OPEN for {self.service_name} - failing fast "
                    f"(retry in {time_until_retry:.1f}s)"
                )
                raise Exception(
                    f"Circuit breaker is OPEN for {self.service_name}. "
                    f"Retry in {time_until_retry:.1f}s"
                )

        # Try service call
        try:
            import asyncio
            result = await asyncio.wait_for(
                service_fn(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Success!
            self._on_success()
            self.total_successes += 1
            return result

        except asyncio.TimeoutError:
            logger.error(f"⏱️  {self.service_name} timeout after {self.config.timeout}s")
            self._on_failure()
            raise Exception(f"{self.service_name} timeout after {self.config.timeout}s")

        except Exception as e:
            logger.error(f"❌ {self.service_name} failed: {type(e).__name__}: {e}")
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count_in_half_open += 1

            if self.success_count_in_half_open >= self.config.success_threshold:
                # Recovery successful
                logger.info(f"✅ {self.service_name} recovered - closing circuit")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count_in_half_open = 0
        else:
            # Reset failure count on success
            if self.failure_count > 0:
                logger.debug(
                    f"✅ {self.service_name} success - resetting failure count "
                    f"(was {self.failure_count})"
                )
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test - back to OPEN
            logger.warning(f"❌ {self.service_name} still failing - circuit back to OPEN")
            self.state = CircuitState.OPEN
            self.success_count_in_half_open = 0

        elif self.failure_count >= self.config.failure_threshold:
            # Too many failures - open circuit
            logger.warning(
                f"🚨 Circuit breaker OPEN for {self.service_name} "
                f"after {self.failure_count} failures"
            )
            self.state = CircuitState.OPEN

        else:
            logger.warning(
                f"⚠️  {self.service_name} failure "
                f"{self.failure_count}/{self.config.failure_threshold}"
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.state != CircuitState.OPEN:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout

    def _time_until_retry(self) -> float:
        """Get seconds until retry attempt"""
        if self.state != CircuitState.OPEN:
            return 0

        time_since_failure = time.time() - self.last_failure_time
        return max(0, self.config.recovery_timeout - time_since_failure)

    def get_state(self) -> Dict[str, Any]:
        """
        Get current circuit breaker state
        
        Returns:
            State information with metrics
        """
        return {
            "service": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "time_since_failure": (
                time.time() - self.last_failure_time 
                if self.last_failure_time > 0 else None
            ),
            "time_until_retry": (
                self._time_until_retry() 
                if self.state == CircuitState.OPEN else None
            ),
            "metrics": {
                "total_calls": self.total_calls,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "success_rate": (
                    self.total_successes / self.total_calls 
                    if self.total_calls > 0 else 0
                )
            }
        }

    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        logger.info(f"🔄 Manually resetting circuit breaker for {self.service_name} to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count_in_half_open = 0


# Global circuit breakers registry
_circuit_breakers: Dict[str, GenericCircuitBreaker] = {}


def get_circuit_breaker(
    service_name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> GenericCircuitBreaker:
    """
    Get or create circuit breaker for a service
    
    Args:
        service_name: Name of the service
        config: Optional circuit breaker configuration
        
    Returns:
        Circuit breaker instance
    """
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = GenericCircuitBreaker(service_name, config)
    
    return _circuit_breakers[service_name]


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """
    Get state of all circuit breakers
    
    Returns:
        Dictionary of service name -> circuit breaker state
    """
    return {
        name: cb.get_state()
        for name, cb in _circuit_breakers.items()
    }

