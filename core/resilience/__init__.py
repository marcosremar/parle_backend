"""
Resilience patterns for fault-tolerant distributed systems

Includes:
- Circuit Breaker: Prevent cascading failures
- Retry Policy: Automatic retry with exponential backoff
- Timeout: Configurable request timeouts
- Bulkhead: Resource isolation
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerError,
    CircuitState,
    get_circuit_breaker_registry
)

from .retry_policy import (
    RetryPolicy,
    RetryPolicyConfig,
    RetryPolicyRegistry,
    RetryExhaustedError,
    RetryStrategy,
    get_retry_policy_registry
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitBreakerError",
    "CircuitState",
    "get_circuit_breaker_registry",

    # Retry Policy
    "RetryPolicy",
    "RetryPolicyConfig",
    "RetryPolicyRegistry",
    "RetryExhaustedError",
    "RetryStrategy",
    "get_retry_policy_registry",
]
