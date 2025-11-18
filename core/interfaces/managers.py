"""
Manager Interfaces (Dependency Inversion Principle)

This module defines interfaces for all system managers (infrastructure components).
Following the Dependency Inversion Principle, services depend on these abstractions
rather than concrete implementations.

Interface Design:
- Use Protocol for flexible structural typing
- Use ABC for strict contract enforcement where needed
- All Protocol interfaces are runtime_checkable
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Session Management Interfaces
# ============================================================================


@runtime_checkable
class ISessionManager(Protocol):
    """
    Interface for Session Management

    Manages user sessions, conversation context, and message history.
    Implementations: In-memory cache, Redis, database-backed, etc.
    """

    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:  # Returns Session object
        """
        Create a new session

        Args:
            user_id: User identifier (optional, auto-generated if None)
            metadata: Session metadata (device, location, etc.)

        Returns:
            Session object
        """
        ...

    def get_session(self, session_id: str) -> Optional[Any]:
        """
        Retrieve session by ID

        Args:
            session_id: Session identifier

        Returns:
            Session object if found, None otherwise
        """
        ...

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Any:
        """
        Get existing session or create new one

        Args:
            session_id: Session identifier (optional)
            user_id: User identifier (optional)

        Returns:
            Session object
        """
        ...

    def add_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        audio_duration_ms: Optional[float] = None,
        voice_used: Optional[str] = None
    ) -> bool:
        """
        Add a complete interaction to session history

        Args:
            session_id: Session identifier
            user_message: User's message/transcription
            assistant_response: Assistant's response
            audio_duration_ms: Audio duration (optional)
            voice_used: Voice identifier used for TTS (optional)

        Returns:
            True if interaction added successfully
        """
        ...

    def get_session_context(
        self,
        session_id: str,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get formatted conversation context for LLM

        Args:
            session_id: Session identifier
            max_messages: Maximum messages to include

        Returns:
            List of message dictionaries (role, content)
        """
        ...

    def cleanup_expired_sessions(self) -> None:
        """Remove expired/inactive sessions"""
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get session manager statistics

        Returns:
            Dictionary with metrics (total_sessions, active_sessions, etc.)
        """
        ...


# ============================================================================
# Profile Management Interfaces
# ============================================================================


@runtime_checkable
class IProfileManager(Protocol):
    """
    Interface for Profile Management

    Manages execution profiles for different environments (dev-local, main-dev, etc.)
    Controls which services are enabled/disabled based on profile.
    """

    def load_profiles(self, profiles_path: Optional[str] = None) -> None:
        """
        Load profiles from YAML configuration

        Args:
            profiles_path: Path to profiles.yaml (optional)
        """
        ...

    def get_profile(self, profile_name: str) -> Optional[Any]:
        """
        Get profile by name

        Args:
            profile_name: Profile identifier (e.g., "dev-local", "main-prod")

        Returns:
            Profile object if found
        """
        ...

    def set_active_profile(self, profile_name: str) -> bool:
        """
        Set active profile

        Args:
            profile_name: Profile to activate

        Returns:
            True if profile activated successfully
        """
        ...

    def get_active_profile(self) -> Optional[Any]:
        """
        Get currently active profile

        Returns:
            Active Profile object
        """
        ...

    def is_service_enabled(self, service_id: str) -> bool:
        """
        Check if service is enabled in active profile

        Args:
            service_id: Service identifier

        Returns:
            True if service is enabled
        """
        ...

    def get_service_overrides(self, service_id: str) -> Dict[str, Any]:
        """
        Get profile-specific overrides for a service

        Args:
            service_id: Service identifier

        Returns:
            Dictionary of configuration overrides
        """
        ...

    def validate_profile(self, profile_name: str) -> Any:
        """
        Validate profile configuration

        Args:
            profile_name: Profile to validate

        Returns:
            ValidationResult object
        """
        ...


# ============================================================================
# GPU Management Interfaces
# ============================================================================


class IGPUManager(ABC):
    """
    Interface for GPU Management (Abstract Base Class)

    Manages GPU resources, memory allocation, and model deployment.
    Uses ABC for stricter contract enforcement due to critical nature.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize GPU manager and detect hardware

        Detects GPU model, compute capability, memory, etc.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if GPU is available

        Returns:
            True if CUDA/GPU is available
        """
        pass

    @abstractmethod
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information

        Returns:
            Dictionary with GPU details (name, memory, compute_cap, etc.)
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage

        Returns:
            Dictionary with memory stats (used_mb, free_mb, total_mb, etc.)
        """
        pass

    @abstractmethod
    async def allocate_memory(
        self,
        service_id: str,
        required_mb: int
    ) -> bool:
        """
        Allocate GPU memory for a service

        Args:
            service_id: Service requesting memory
            required_mb: Memory required in MB

        Returns:
            True if allocation successful
        """
        pass

    @abstractmethod
    async def release_memory(self, service_id: str) -> bool:
        """
        Release GPU memory allocated to a service

        Args:
            service_id: Service releasing memory

        Returns:
            True if release successful
        """
        pass

    @abstractmethod
    def get_recommended_backend(self, model_name: str) -> str:
        """
        Get recommended backend for a model

        Args:
            model_name: Model identifier

        Returns:
            Backend name (e.g., "vllm", "vulkan", "transformers")
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup GPU resources and reset state
        """
        pass


# ============================================================================
# Metrics/Observability Interfaces
# ============================================================================


@runtime_checkable
class IMetricsCollector(Protocol):
    """
    Interface for Metrics Collection

    Collects and reports system metrics (counters, gauges, histograms).
    Implementations: Prometheus, StatsD, CloudWatch, etc.
    """

    def increment(
        self,
        metric_name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric

        Args:
            metric_name: Metric identifier
            value: Increment value (default: 1)
            tags: Metric tags/labels
        """
        ...

    def gauge(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set a gauge metric

        Args:
            metric_name: Metric identifier
            value: Gauge value
            tags: Metric tags/labels
        """
        ...

    def histogram(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a histogram value

        Args:
            metric_name: Metric identifier
            value: Value to record
            tags: Metric tags/labels
        """
        ...

    def timing(
        self,
        metric_name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a timing/duration

        Args:
            metric_name: Metric identifier
            duration_ms: Duration in milliseconds
            tags: Metric tags/labels
        """
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot

        Returns:
            Dictionary of all metrics and their values
        """
        ...

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state"""
        ...


# ============================================================================
# Communication Manager Interface
# ============================================================================


@runtime_checkable
class ICommunicationManager(Protocol):
    """
    Interface for Service Communication

    Manages inter-service communication with automatic protocol selection
    (ZeroMQ → gRPC → HTTP) and circuit breaker/retry logic.
    """

    async def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Call another service

        Args:
            service_name: Target service identifier
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            data: Request payload
            timeout: Request timeout in seconds
            **kwargs: Additional parameters

        Returns:
            Service response (type depends on endpoint)
        """
        ...

    async def register_service(
        self,
        service_id: str,
        host: str,
        port: int,
        protocol: str = "http"
    ) -> None:
        """
        Register a service in the service registry

        Args:
            service_id: Service identifier
            host: Service host
            port: Service port
            protocol: Communication protocol
        """
        ...

    async def unregister_service(self, service_id: str) -> None:
        """
        Unregister a service

        Args:
            service_id: Service to unregister
        """
        ...

    def get_service_url(self, service_name: str) -> Optional[str]:
        """
        Get service URL

        Args:
            service_name: Service identifier

        Returns:
            Service URL if registered
        """
        ...

    async def health_check(self, service_name: str) -> bool:
        """
        Check if service is healthy

        Args:
            service_name: Service to check

        Returns:
            True if service is healthy
        """
        ...


# ============================================================================
# Benchmark Manager Interface
# ============================================================================


@runtime_checkable
class IBenchmarkManager(Protocol):
    """
    Interface for Performance Benchmarking

    Manages performance benchmarks and optimization experiments.
    """

    async def run_benchmark(
        self,
        benchmark_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a performance benchmark

        Args:
            benchmark_name: Benchmark identifier
            config: Benchmark configuration

        Returns:
            Benchmark results
        """
        ...

    def get_benchmark_results(
        self,
        benchmark_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get stored benchmark results

        Args:
            benchmark_name: Benchmark identifier

        Returns:
            Results if available
        """
        ...

    def compare_benchmarks(
        self,
        baseline: str,
        comparison: str
    ) -> Dict[str, Any]:
        """
        Compare two benchmark results

        Args:
            baseline: Baseline benchmark name
            comparison: Comparison benchmark name

        Returns:
            Comparison metrics (speedup, regression, etc.)
        """
        ...


# ============================================================================
# Export all interfaces
# ============================================================================

__all__ = [
    # Session & Profile
    "ISessionManager",
    "IProfileManager",
    # Infrastructure
    "IGPUManager",
    "IMetricsCollector",
    "ICommunicationManager",
    "IBenchmarkManager",
]
