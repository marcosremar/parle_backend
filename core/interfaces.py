"""
Interface definitions for the unified pipeline
Using Protocol for better type hints and flexibility

This file contains two types of interfaces:
1. Domain interfaces (IAudioProcessor, ITextToSpeech, etc.) - Business logic
2. Infrastructure interfaces (ILogger, IMetricsCollector, etc.) - Cross-cutting concerns
"""

from typing import Protocol, Optional, Dict, Any, List, runtime_checkable
from enum import Enum
import numpy as np
import logging


class IAudioProcessor(Protocol):
    """Interface for audio processing (STT/Understanding)"""

    def initialize(self) -> None:
        """Initialize the audio processor"""
        ...

    def process(self,
               audio: np.ndarray,
               sample_rate: int,
               session_id: str,
               context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process audio and return text response

        Args:
            audio: Audio data as numpy array (float32)
            sample_rate: Sample rate in Hz (typically 16000)
            session_id: Unique session identifier
            context: Optional context dictionary

        Returns:
            Text response from the model
        """
        ...

    def cleanup(self) -> None:
        """Cleanup resources"""
        ...


class ITextToSpeech(Protocol):
    """Interface for text-to-speech synthesis"""

    def initialize(self) -> None:
        """Initialize the TTS engine"""
        ...

    def synthesize(self,
                  text: str,
                  voice_id: Optional[str] = None,
                  speed: float = 1.0) -> bytes:
        """
        Synthesize text to audio

        Args:
            text: Text to synthesize
            voice_id: Optional voice identifier
            speed: Speed multiplier (1.0 = normal)

        Returns:
            Audio data as bytes (PCM float32)
        """
        ...

    def get_available_voices(self) -> List[Dict[str, str]]:
        """Get list of available voices"""
        ...

    def cleanup(self) -> None:
        """Cleanup resources"""
        ...


class IMemoryStore(Protocol):
    """Interface for conversation memory storage"""

    def initialize(self) -> None:
        """Initialize the memory store"""
        ...

    def get_context(self,
                   session_id: str,
                   max_messages: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get conversation context for a session

        Args:
            session_id: Unique session identifier
            max_messages: Maximum number of messages to retrieve

        Returns:
            List of messages or None if no history
        """
        ...

    def save_interaction(self,
                        session_id: str,
                        user_input: Optional[str],
                        assistant_response: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save an interaction to memory

        Args:
            session_id: Unique session identifier
            user_input: User's input (text from STT)
            assistant_response: Assistant's response
            metadata: Optional metadata (timestamps, etc)
        """
        ...

    def clear_session(self, session_id: str) -> None:
        """Clear all memory for a session"""
        ...

    def cleanup_old_sessions(self, max_age_seconds: int = 1800) -> int:
        """
        Clean up old sessions

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of sessions cleaned up
        """
        ...


class IWebRTCHandler(Protocol):
    """Interface for WebRTC connection handling"""
    
    async def initialize(self) -> None:
        """Initialize WebRTC handler"""
        ...
    
    async def handle_offer(self, 
                          offer: Dict[str, Any],
                          session_id: str) -> Dict[str, Any]:
        """
        Handle WebRTC offer and return answer
        
        Args:
            offer: WebRTC offer from client
            session_id: Unique session identifier
            
        Returns:
            WebRTC answer to send back to client
        """
        ...
    
    async def handle_ice_candidate(self,
                                  candidate: Dict[str, Any],
                                  session_id: str) -> None:
        """
        Handle ICE candidate from client
        
        Args:
            candidate: ICE candidate information
            session_id: Session identifier
        """
        ...
    
    async def close_connection(self, session_id: str) -> None:
        """
        Close WebRTC connection for a session
        
        Args:
            session_id: Session identifier
        """
        ...
    
    async def get_connection_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get connection statistics
        
        Args:
            session_id: Session identifier
            
        Returns:
            Connection statistics or None if not connected
        """
        ...
    
    async def cleanup(self) -> None:
        """Cleanup all connections"""
        ...


# ============================================================================
# INFRASTRUCTURE INTERFACES
# ============================================================================


@runtime_checkable
class ILogger(Protocol):
    """
    Logger interface

    Any class implementing these methods can be used as a logger.
    This allows for different logging implementations (loguru, standard logging, etc.)
    """

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message"""
        ...

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message"""
        ...

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message"""
        ...

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message"""
        ...

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message"""
        ...

    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback"""
        ...


@runtime_checkable
class IMetricsCollector(Protocol):
    """
    Metrics collector interface

    Defines the contract for collecting metrics (counters, gauges, histograms).
    Implementations can use Prometheus, StatsD, custom backends, etc.
    """

    def increment(self, metric_name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        ...

    def gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        ...

    def histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        ...

    def timing(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timing value (alias for histogram)"""
        ...

    async def flush(self) -> None:
        """Flush buffered metrics to backend"""
        ...


@runtime_checkable
class ITelemetry(Protocol):
    """
    Telemetry interface for distributed tracing

    Provides OpenTelemetry-compatible tracing and request tracking.
    """

    def get_current_trace_id(self) -> str:
        """Get current request/trace ID (32 hex chars)"""
        ...

    def get_current_span_id(self) -> str:
        """Get current span ID (16 hex chars)"""
        ...

    def trace(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Create a traced span context manager

        Usage:
            with telemetry.trace("database_query", {"table": "users"}):
                # Code here - timing measured automatically
                pass
        """
        ...

    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug with trace context"""
        ...

    def log_info(self, message: str, **kwargs) -> None:
        """Log info with trace context"""
        ...

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning with trace context"""
        ...

    def log_error(self, message: str, **kwargs) -> None:
        """Log error with trace context"""
        ...

    def counter(self, name: str, description: str = "", unit: str = ""):
        """Get or create a counter metric"""
        ...

    def histogram(self, name: str, description: str = "", unit: str = ""):
        """Get or create a histogram metric"""
        ...


class GPUStatus(Enum):
    """GPU status enum"""
    AVAILABLE = "available"
    IN_USE = "in_use"
    ERROR = "error"
    NOT_AVAILABLE = "not_available"


@runtime_checkable
class IGPUManager(Protocol):
    """
    GPU Manager interface

    Manages GPU resources, allocation, and monitoring.
    """

    async def allocate(self, service_name: str, memory_mb: int) -> bool:
        """
        Allocate GPU memory for a service

        Args:
            service_name: Service requesting GPU
            memory_mb: Memory to allocate in MB

        Returns:
            True if allocation successful
        """
        ...

    async def release(self, service_name: str) -> None:
        """Release GPU allocation for a service"""
        ...

    async def get_status(self) -> Dict[str, Any]:
        """
        Get GPU status

        Returns:
            Dict with GPU info (status, memory, temperature, etc.)
        """
        ...

    async def is_available(self) -> bool:
        """Check if GPU is available"""
        ...

    def get_device_id(self) -> Optional[int]:
        """Get CUDA device ID (None if no GPU)"""
        ...


@runtime_checkable
class ISettingsService(Protocol):
    """
    Settings service interface

    Provides centralized configuration management.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key (supports dot notation: "database.host")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        ...

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        ...

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value"""
        ...

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        ...

    def get_str(self, key: str, default: str = "") -> str:
        """Get string configuration value"""
        ...

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get list configuration value"""
        ...

    def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        """Get dictionary configuration value"""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        ...

    def reload(self) -> None:
        """Reload configuration from source"""
        ...


@runtime_checkable
class ICommunicationManager(Protocol):
    """
    Communication Manager interface

    Handles inter-service communication with automatic protocol selection.
    """

    async def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0
    ) -> Any:
        """
        Call another service

        Args:
            service_name: Target service name
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            data: Request payload
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        ...

    async def register_service(self, service_name: str, url: str) -> None:
        """Register a service URL"""
        ...

    async def unregister_service(self, service_name: str) -> None:
        """Unregister a service"""
        ...

    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get registered service URL"""
        ...

    async def health_check(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        ...


@runtime_checkable
class IDatabaseClient(Protocol):
    """
    Database client interface

    Generic interface for database operations.
    """

    async def connect(self) -> None:
        """Establish database connection"""
        ...

    async def disconnect(self) -> None:
        """Close database connection"""
        ...

    async def is_connected(self) -> bool:
        """Check if database is connected"""
        ...

    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute a database query"""
        ...

    async def fetch_one(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch one result"""
        ...

    async def fetch_all(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Fetch all results"""
        ...

    async def transaction(self):
        """
        Begin a transaction context

        Usage:
            async with db.transaction():
                await db.execute("INSERT ...")
                await db.execute("UPDATE ...")
        """
        ...


@runtime_checkable
class ICacheService(Protocol):
    """
    Cache service interface

    Generic interface for caching (Redis, in-memory, etc.)
    """

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete key from cache"""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        ...

    async def clear(self) -> None:
        """Clear all cache entries"""
        ...

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        ...

    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set multiple values in cache"""
        ...


@runtime_checkable
class IServiceRegistry(Protocol):
    """
    Service registry interface

    Manages service discovery and registration.
    """

    async def register(
        self,
        service_name: str,
        host: str,
        port: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register a service

        Returns:
            Service ID
        """
        ...

    async def unregister(self, service_name: str) -> None:
        """Unregister a service"""
        ...

    async def discover(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Discover a service by name

        Returns:
            Service info (host, port, metadata) or None
        """
        ...

    async def discover_all(self) -> List[Dict[str, Any]]:
        """Discover all registered services"""
        ...

    async def health_check(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        ...


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_logger(obj: Any) -> bool:
    """Check if object implements ILogger interface"""
    return isinstance(obj, ILogger)


def is_metrics_collector(obj: Any) -> bool:
    """Check if object implements IMetricsCollector interface"""
    return isinstance(obj, IMetricsCollector)


def is_telemetry(obj: Any) -> bool:
    """Check if object implements ITelemetry interface"""
    return isinstance(obj, ITelemetry)


def is_gpu_manager(obj: Any) -> bool:
    """Check if object implements IGPUManager interface"""
    return isinstance(obj, IGPUManager)


def is_settings_service(obj: Any) -> bool:
    """Check if object implements ISettingsService interface"""
    return isinstance(obj, ISettingsService)


def is_communication_manager(obj: Any) -> bool:
    """Check if object implements ICommunicationManager interface"""
    return isinstance(obj, ICommunicationManager)


def is_database_client(obj: Any) -> bool:
    """Check if object implements IDatabaseClient interface"""
    return isinstance(obj, IDatabaseClient)


def is_cache_service(obj: Any) -> bool:
    """Check if object implements ICacheService interface"""
    return isinstance(obj, ICacheService)


def is_service_registry(obj: Any) -> bool:
    """Check if object implements IServiceRegistry interface"""
    return isinstance(obj, IServiceRegistry)