"""
Communication Interfaces - Formal contracts for communication components.

SOLID Principles Applied:
- Interface Segregation Principle (ISP): Multiple small, focused interfaces
- Dependency Inversion Principle (DIP): Depend on abstractions, not concretions

Each interface represents a single responsibility in the communication layer.
"""

from typing import Protocol, Dict, Any, Optional, AsyncIterator, Callable, List
from enum import Enum


# ============================================================================
# Protocol Enums
# ============================================================================

class CommunicationProtocol(str, Enum):
    """Available communication protocols (ordered by preference)."""
    DIRECT = "direct"          # In-process call (zero overhead)
    ZEROMQ = "zeromq"          # Ultra-fast IPC (0.01ms, 410k msg/s)
    GRPC = "grpc"              # Efficient RPC (~7ms)
    HTTP_BINARY = "http_binary"  # Binary over HTTP
    HTTP_JSON = "json"         # JSON over HTTP (fallback)


class Priority(str, Enum):
    """Request priority levels for protocol selection."""
    REALTIME = "realtime"  # Low-latency required (WebRTC, voice)
    NORMAL = "normal"      # Standard API calls
    DEBUG = "debug"        # Testing/debugging (prefer human-readable)


# ============================================================================
# Core Interfaces
# ============================================================================

class IProtocolSelector(Protocol):
    """
    Interface for protocol selection logic.

    Responsibilities:
    - Select optimal protocol based on context (service, data type, priority)
    - Provide fallback protocol when primary fails
    - Track protocol usage metrics

    SOLID: Single Responsibility - Only protocol selection logic
    """

    def select_protocol(
        self,
        service_name: str,
        data_size: int,
        priority: Priority,
        data_type: str,
        is_internal: bool = False
    ) -> CommunicationProtocol:
        """
        Select optimal protocol for a service call.

        Args:
            service_name: Target service identifier
            data_size: Payload size in bytes
            priority: Request priority level
            data_type: Type of data ('audio', 'text', 'json')
            is_internal: Whether service is in-process

        Returns:
            Selected protocol enum
        """
        ...

    def get_fallback_protocol(
        self,
        failed_protocol: CommunicationProtocol,
        service_name: str
    ) -> CommunicationProtocol:
        """
        Get fallback protocol when primary fails.

        Args:
            failed_protocol: Protocol that failed
            service_name: Target service

        Returns:
            Fallback protocol to try
        """
        ...

    def set_preference(
        self,
        service_name: str,
        primary: CommunicationProtocol,
        fallback: CommunicationProtocol
    ) -> None:
        """Set protocol preference for a specific service."""
        ...


class IServiceResolver(Protocol):
    """
    Interface for service URL resolution.

    Responsibilities:
    - Resolve service URLs (internal vs external)
    - Cache resolved URLs for performance
    - Integrate with service discovery
    - Handle port changes and invalidation

    SOLID: Single Responsibility - Only URL resolution
    """

    def resolve_url(
        self,
        service_name: str,
        endpoint_path: str = "",
        is_internal: bool = False
    ) -> str:
        """
        Resolve full URL for a service endpoint.

        Args:
            service_name: Service identifier
            endpoint_path: API endpoint (e.g., '/generate')
            is_internal: Whether service runs in-process

        Returns:
            Full URL (e.g., 'http://localhost:8100/generate')
        """
        ...

    def invalidate_cache(self, service_name: Optional[str] = None) -> None:
        """
        Invalidate cached URLs.

        Args:
            service_name: Service to invalidate, or None for all
        """
        ...

    def update_port(self, service_name: str, new_port: int) -> Optional[int]:
        """
        Update port for a service and invalidate cache.

        Args:
            service_name: Service identifier
            new_port: New port number

        Returns:
            Previous port or None
        """
        ...

    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Discover service via Service Discovery.

        Args:
            service_name: Service to discover

        Returns:
            Service info dict or None if not found
        """
        ...


class IResilienceManager(Protocol):
    """
    Interface for resilience patterns.

    Responsibilities:
    - Circuit breaker pattern
    - Retry with exponential backoff
    - Timeout handling
    - Failure tracking

    SOLID: Single Responsibility - Only resilience logic
    """

    async def execute_with_resilience(
        self,
        fn: Callable,
        service_name: str,
        timeout: float = 30.0,
        retries: Optional[int] = None,
        enable_circuit_breaker: bool = True
    ) -> Any:
        """
        Execute function with resilience patterns.

        Args:
            fn: Async function to execute
            service_name: Service name (for circuit breaker)
            timeout: Timeout in seconds
            retries: Number of retries (None = use default)
            enable_circuit_breaker: Enable circuit breaker

        Returns:
            Result from function

        Raises:
            CircuitBreakerError: Circuit is open
            RetryExhaustedError: All retries failed
        """
        ...

    def reset_circuit_breaker(self, service_name: str) -> None:
        """Manually reset circuit breaker for a service."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get resilience statistics (circuit breakers, retries)."""
        ...


class IMetricsCollector(Protocol):
    """
    Interface for communication metrics tracking.

    Responsibilities:
    - Record successful calls (latency, protocol used)
    - Record failed calls (error type, latency)
    - Provide aggregated metrics
    - Clear metrics on demand

    SOLID: Single Responsibility - Only metrics tracking
    """

    def record_success(
        self,
        service_name: str,
        protocol: CommunicationProtocol,
        latency_ms: float
    ) -> None:
        """Record successful service call."""
        ...

    def record_failure(
        self,
        service_name: str,
        protocol: CommunicationProtocol,
        latency_ms: float,
        error: str
    ) -> None:
        """Record failed service call."""
        ...

    def get_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.

        Args:
            service_name: Specific service, or None for all

        Returns:
            Metrics dict with success rate, avg latency, etc.
        """
        ...

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report for all services."""
        ...

    def clear_metrics(self, service_name: Optional[str] = None) -> None:
        """Clear metrics for a service or all services."""
        ...


class IServiceRegistry(Protocol):
    """
    Interface for in-process service registration.

    Responsibilities:
    - Register services for direct calls
    - Unregister services
    - Check service availability
    - Manage service lifecycle

    SOLID: Single Responsibility - Only service registration
    """

    def register_service(
        self,
        service_name: str,
        service_instance: Any,
        parent_service: Optional[str] = None
    ) -> None:
        """
        Register an in-process service for direct calls.

        Args:
            service_name: Service identifier
            service_instance: Service instance (must have callable endpoints)
            parent_service: Optional parent composite service
        """
        ...

    def unregister_service(self, service_name: str) -> None:
        """Unregister a service."""
        ...

    def is_registered(self, service_name: str) -> bool:
        """Check if service is registered."""
        ...

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get registered service instance."""
        ...

    def get_all_services(self) -> Dict[str, Any]:
        """Get all registered services."""
        ...


class IServiceDiscoveryClient(Protocol):
    """
    Interface for service discovery integration.

    Responsibilities:
    - Lookup services via Service Manager
    - Cache discovery results
    - Handle discovery failures gracefully

    SOLID: Single Responsibility - Only service discovery
    """

    async def lookup_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Lookup service via Service Discovery.

        Args:
            service_name: Service to lookup

        Returns:
            Service info (host, port, health, etc.) or None
        """
        ...

    def invalidate_discovery_cache(self, service_name: Optional[str] = None) -> None:
        """Invalidate discovery cache."""
        ...


class IActivityTracker(Protocol):
    """
    Interface for service activity tracking (for auto-scaling).

    Responsibilities:
    - Track service usage
    - Provide activity statistics
    - Trigger auto-scaling based on activity

    SOLID: Single Responsibility - Only activity tracking
    """

    def record_activity(self, service_name: str) -> None:
        """Record activity for a service (for auto-scaling)."""
        ...

    def get_activity_stats(self, service_name: str) -> Dict[str, Any]:
        """Get activity statistics for a service."""
        ...

    def get_all_activity(self) -> Dict[str, Dict[str, Any]]:
        """Get activity for all services."""
        ...


# ============================================================================
# Composite Interface (Facade)
# ============================================================================

class ICommunicationManager(Protocol):
    """
    High-level interface for service communication (Facade pattern).

    This is the main interface that services interact with.
    It orchestrates all the specialized components above.

    SOLID:
    - Facade Pattern: Simple interface to complex subsystem
    - Dependency Inversion: Depends on abstractions (IProtocolSelector, etc.)
    """

    async def call_service(
        self,
        service_name: str,
        endpoint_path: str,
        method: str = 'POST',
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        enable_resilience: bool = True
    ) -> Dict[str, Any]:
        """Generic service call with automatic URL resolution."""
        ...

    async def call_audio_service(
        self,
        service_name: str,
        audio_data: bytes,
        sample_rate: int = 16000,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Priority = Priority.NORMAL,
        force_protocol: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call service with audio data."""
        ...

    async def call_text_service(
        self,
        service_name: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Priority = Priority.NORMAL,
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call service with text data."""
        ...

    async def stream_request(
        self,
        service_name: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.REALTIME,
        endpoint: Optional[str] = None,
        chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Stream response from service."""
        ...

    async def stream_json_events(
        self,
        service_name: str,
        action: str,
        payload: Dict[str, Any],
        endpoint: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream JSON events from service."""
        ...

    def register_internal_service(
        self,
        service_name: str,
        service_instance: Any
    ) -> None:
        """Register in-process service for direct calls."""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics."""
        ...

    async def initialize(self) -> None:
        """Initialize communication manager."""
        ...

    async def cleanup(self) -> None:
        """Cleanup resources."""
        ...


# ============================================================================
# Type Aliases
# ============================================================================

ServiceMetrics = Dict[str, Any]
"""Type alias for service metrics dictionary."""

ServiceInfo = Dict[str, Any]
"""Type alias for service discovery info."""

ProtocolPreferences = Dict[str, CommunicationProtocol]
"""Type alias for protocol preferences."""
