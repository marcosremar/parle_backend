#!/usr/bin/env python3
"""
Service Communication Manager
Intelligent service communication with automatic protocol selection and failover
Handles routing to internal (service manager) vs external (standalone) services
"""

import aiohttp
import asyncio
import logging
import time
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from enum import Enum

from src.core.protocols import BinaryProtocolAdapter, JsonProtocolAdapter
from src.core.settings import get_service_endpoints_settings
from src.core.exceptions import UltravoxError, wrap_exception

# OpenTelemetry - Trace propagation
try:
    from src.core.observability import inject_trace_context
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Fallback no-op function
    def inject_trace_context(headers):
        return headers

# Resilience patterns
from src.core.resilience import (
    get_circuit_breaker_registry,
    get_retry_policy_registry,
    CircuitBreakerConfig,
    RetryPolicyConfig,
    RetryStrategy,
    CircuitBreakerError,
    RetryExhaustedError
)


logger = logging.getLogger(__name__)


class Priority(str, Enum):
    """Request priority levels"""
    REALTIME = "realtime"      # WebRTC, low-latency required
    NORMAL = "normal"          # Standard API calls
    DEBUG = "debug"            # Testing/debugging, prefer JSON


class ServiceCommunicationManager:
    """
    Manages intelligent service-to-service communication

    Features:
    - Automatic protocol selection (Binary vs JSON)
    - Performance-based optimization
    - Automatic failover on errors
    - Metrics tracking per protocol
    """

    def __init__(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        self.session = session
        self.own_session = session is None

        # Protocol adapters
        self.protocols = {
            'http_binary': BinaryProtocolAdapter(),
            'json': JsonProtocolAdapter()
        }

        # Direct call adapter for internal services (zero overhead!)
        try:
            from src.core.protocols.direct_call_adapter import DirectCallProtocolAdapter
            self.protocols['direct'] = DirectCallProtocolAdapter()
            self.direct_call_enabled = True
            logger.info("🚀 Direct call protocol enabled (zero overhead)")
        except Exception as e:
            logger.warning(f"⚠️ Direct call protocol unavailable: {e}")
            self.direct_call_enabled = False

        # ZeroMQ adapter (PRIMARY - ultra-fast inproc transport, 0.01ms, 410k msg/s)
        self.zeromq_enabled = os.getenv("ENABLE_ZEROMQ", "true").lower() == "true"
        if self.zeromq_enabled:
            try:
                from src.core.protocols.zeromq_adapter import ZeroMQProtocolAdapter
                self.protocols['zeromq'] = ZeroMQProtocolAdapter()
                logger.info("🚀 ZeroMQ protocol enabled (inproc, ultra-fast!)")
            except Exception as e:
                logger.warning(f"⚠️ ZeroMQ protocol unavailable: {e}")
                self.zeromq_enabled = False

        # gRPC adapter (ENABLED by default - gRPC is now FALLBACK protocol)
        self.grpc_enabled = os.getenv("ENABLE_GRPC", "true").lower() == "true"
        if self.grpc_enabled:
            try:
                from src.core.protocols.grpc_adapter import GrpcProtocolAdapter
                self.protocols['grpc'] = GrpcProtocolAdapter()
                logger.info("🚀 gRPC protocol enabled")
            except Exception as e:
                logger.warning(f"⚠️ gRPC protocol unavailable: {e}")
                self.grpc_enabled = False

        # Performance metrics per service/protocol
        self.metrics = {}

        # In-process service registry (for composite services)
        self.inprocess_services = {}  # {service_name: service_instance}

        # Service-specific preferences (ZeroMQ → gRPC → HTTP Binary → JSON)
        self.preferences = {
            # Default preferences (can be overridden per service)
            # Protocol priority: ZeroMQ > gRPC > HTTP Binary > JSON
            'default': {
                'primary': 'zeromq',         # 🚀 ZeroMQ is PRIMARY (0.01ms, 410k msg/s!)
                'secondary': 'grpc',         # ⚡ gRPC is FALLBACK 1 (~7ms)
                'tertiary': 'http_binary',   # 🔄 HTTP Binary is FALLBACK 2
                'fallback': 'json',          # 🐌 JSON is LAST RESORT
                'auto_optimize': True
            },
            # For services: ZeroMQ → gRPC → HTTP Binary → JSON
            'stt': {
                'primary': 'zeromq',
                'secondary': 'grpc',
                'tertiary': 'http_binary',
                'fallback': 'json'
            },
            'llm': {
                'primary': 'zeromq',
                'secondary': 'grpc',
                'tertiary': 'http_binary',
                'fallback': 'json'
            },
            'tts': {
                'primary': 'zeromq',
                'secondary': 'grpc',
                'tertiary': 'http_binary',
                'fallback': 'json'
            }
        }

        # Thresholds for protocol selection
        self.thresholds = {
            'large_payload_bytes': 100_000,  # Use binary for payloads >100KB
            'min_success_rate': 0.8,         # Switch if success rate < 80%
            'max_latency_ms': 100            # Prefer binary if latency critical
        }

        # Service execution configuration (internal vs external routing)
        self.service_config = {}
        self.service_manager_url = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")

        # Load service ports from configuration (eliminates hardcoded values)
        # Fallback to discovery if config loading fails
        try:
            from src.core.config_loader import get_service_ports
            self.default_service_ports = get_service_ports()
            logger.info(
                f"✅ Loaded {len(self.default_service_ports)} service ports from services_config.yaml",
                ports=list(self.default_service_ports.keys())
            )
        except Exception as e:
            logger.warning(
                f"⚠️  Failed to load ports from config: {e}. "
                "Using empty dict (will rely on service discovery)"
            )
            self.default_service_ports = {}

        # Load service execution configuration
        self._load_service_execution_config()

        # Activity tracker for auto-scaling remote services
        self.activity_tracker = None  # Will be initialized when needed
        self._activity_tracking_enabled = False

        # Resilience patterns
        self.circuit_breaker_registry = get_circuit_breaker_registry()
        self.retry_policy_registry = get_retry_policy_registry()
        self._resilience_enabled = os.getenv("ENABLE_RESILIENCE", "true").lower() == "true"

        if self._resilience_enabled:
            logger.info("🛡️  Resilience patterns enabled (Circuit Breaker + Retry Policy)")
        else:
            logger.warning("⚠️  Resilience patterns disabled")

        logger.info("🔗 ServiceCommunicationManager initialized")

    def register_internal_service(self, service_name: str, service_instance: Any) -> None:
        """
        Register an internal (in-process) service for direct calling

        This enables zero-overhead communication for internal services.

        Args:
            service_name: Service identifier (e.g., 'session', 'external_stt')
            service_instance: BaseService instance
        """
        if 'direct' in self.protocols:
            self.protocols['direct'].register_service(service_name, service_instance)
            logger.info(f"✅ Registered internal service: {service_name}")
        else:
            logger.warning(f"⚠️  Direct protocol not available, cannot register {service_name}")

    def unregister_internal_service(self, service_name: str) -> None:
        """Unregister an internal service"""
        if 'direct' in self.protocols:
            self.protocols['direct'].unregister_service(service_name)

    def is_service_internal(self, service_name: str) -> bool:
        """
        Check if a service is running internally (in-process)

        Args:
            service_name: Service identifier

        Returns:
            bool: True if service is internal
        """
        service_cfg = self.service_config.get(service_name, {})
        return service_cfg.get('execution_mode', 'external') == 'internal'

    async def initialize(self) -> None:
        """Initialize HTTP session and protocol adapters if not provided"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            logger.info("✅ Created new aiohttp session")

        # Initialize ZeroMQ adapter if enabled
        if self.zeromq_enabled and 'zeromq' in self.protocols:
            try:
                await self.protocols['zeromq'].initialize()
                logger.info("✅ ZeroMQ adapter initialized")
            except Exception as e:
                logger.warning(f"⚠️  ZeroMQ initialization failed: {e}")

    async def register_inprocess_service(
        self,
        service_name: str,
        service_instance: Any,
        parent_service: Optional[str] = None
    ) -> None:
        """
        Register a service as in-process (for composite services)

        Args:
            service_name: Service name
            service_instance: Service instance object
            parent_service: Parent composite service (optional)
        """
        self.inprocess_services[service_name] = {
            'instance': service_instance,
            'parent': parent_service,
            'registered_at': time.time()
        }

        # Override preference to use direct protocol
        self.preferences[service_name] = {
            'primary': 'direct',
            'secondary': 'http_binary',
            'fallback': 'json'
        }

        logger.info(f"📝 Registered in-process service: {service_name}" +
                   (f" (parent: {parent_service})" if parent_service else ""))

    def is_inprocess_service(self, service_name: str) -> bool:
        """Check if a service is registered as in-process"""
        return service_name in self.inprocess_services

    def get_inprocess_service(self, service_name: str) -> Optional[Any]:
        """Get in-process service instance"""
        info = self.inprocess_services.get(service_name)
        return info['instance'] if info else None

    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Discover service endpoint via Service Discovery

        Args:
            service_name: Service name to discover

        Returns:
            Service discovery info or None
        """
        try:
            if not self.session:
                await self.initialize()

            async with self.session.get(
                f"{self.service_manager_url}/discovery/lookup/{service_name}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"⚠️ Service {service_name} not found in discovery: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.debug(f"Discovery lookup failed for {service_name}: {e}")
            return None

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.own_session and self.session:
            await self.session.close()
            logger.info("🧹 Closed aiohttp session")

    def enable_activity_tracking(self) -> None:
        """Enable activity tracking for auto-scaling remote services"""
        if not self._activity_tracking_enabled:
            from src.core.service_manager.activity_tracker import get_activity_tracker
            self.activity_tracker = get_activity_tracker()
            self._activity_tracking_enabled = True
            logger.info("📊 Activity tracking enabled for Communication Manager")

    def _record_service_activity(self, service_name: str) -> None:
        """Record activity for a service (for auto-scaling)"""
        if self._activity_tracking_enabled and self.activity_tracker:
            self.activity_tracker.record_activity(service_name)

    def _load_service_execution_config(self) -> Optional[Dict[str, Any]]:
        """Load service execution configuration to determine internal vs external routing"""
        try:
            # Find config file
            config_path = Path(__file__).parent.parent.parent / "config" / "service_execution.yaml"
            if not config_path.exists():
                logger.warning(f"⚠️  Service execution config not found: {config_path}")
                return None

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Extract service configurations
            services = config.get('services', {})
            for service_name, service_config in services.items():
                self.service_config[service_name] = {
                    'execution_mode': service_config.get('execution_mode', 'external'),
                    'module_path': service_config.get('module_path'),
                    'port': self.default_service_ports.get(service_name)
                }

            logger.info(f"📋 Loaded service execution config: {len(self.service_config)} services")

        except Exception as e:
            logger.warning(f"⚠️  Failed to load service execution config: {e}")

    def _resolve_service_url(self, service_name: str, endpoint_path: str = "") -> str:
        """
        Resolve service URL based on execution mode (internal vs external)

        Args:
            service_name: Name of the service (e.g., 'conversation_store', 'session')
            endpoint_path: Endpoint path (e.g., '/api/conversations', '/health')

        Returns:
            Full URL to the service endpoint

        Examples:
            Internal service: http://localhost:8888/api/conversation_store/api/conversations
            External service: http://localhost:8010/api/conversations
        """
        # Get service configuration
        service_cfg = self.service_config.get(service_name, {})
        execution_mode = service_cfg.get('execution_mode', 'external')

        # Check for env var override (e.g., CONVERSATION_STORE_URL)
        env_var = f"{service_name.upper()}_URL"
        if env_var in os.environ:
            base_url = os.getenv(env_var).rstrip('/')
            url = f"{base_url}{endpoint_path}"
            logger.debug(f"🔗 Resolved {service_name} from env: {url}")
            return url

        if execution_mode == 'internal':
            # Internal service: route through service manager at /api/<service_name>
            # Strip the /api/<service_name> prefix from endpoint_path since Service Manager adds it
            service_prefix = f"/api/{service_name}"
            if endpoint_path.startswith(service_prefix):
                # Remove the service prefix - Service Manager already routes to /api/<service_name>
                relative_path = endpoint_path[len(service_prefix):]
            else:
                relative_path = endpoint_path

            url = f"{self.service_manager_url}/api/{service_name}{relative_path}"
            logger.debug(f"🔗 Resolved {service_name} (internal): {url}")
            return url
        else:
            # External service: direct connection to service port
            port = service_cfg.get('port', self.default_service_ports.get(service_name, 8000))

            # Check for port override env var (e.g., CONVERSATION_STORE_PORT)
            port_env_var = f"{service_name.upper()}_PORT"
            if port_env_var in os.environ:
                port = int(os.getenv(port_env_var))

            url = f"http://localhost:{port}{endpoint_path}"
            logger.debug(f"🔗 Resolved {service_name} (external): {url}")
            return url

    def _resolve_endpoint_url(self, service_name: str, endpoint: Optional[str] = None,
                              endpoint_name: str = 'text') -> str:
        """
        Resolve endpoint URL - handles both partial paths and full URLs

        Args:
            service_name: Service identifier
            endpoint: Endpoint path (e.g., '/health') or full URL
            endpoint_name: Endpoint name for settings lookup

        Returns:
            Full endpoint URL
        """
        # If endpoint is a full URL (starts with http), use it as-is
        if endpoint and (endpoint.startswith('http://') or endpoint.startswith('https://')):
            return endpoint

        # If endpoint is a path, combine with service URL
        if endpoint:
            return self._resolve_service_url(service_name, endpoint)

        # Otherwise, try to resolve from settings (old behavior)
        return self._resolve_endpoint(service_name, endpoint, endpoint_name)

    def _resolve_endpoint(self, service_name: str, endpoint: Optional[str] = None,
                         action: str = 'audio') -> str:
        """
        Resolve endpoint URL from settings or use provided endpoint

        Args:
            service_name: Service identifier
            endpoint: Optional full URL endpoint
            action: Action/endpoint name to lookup in settings (default: 'audio')

        Returns:
            Full endpoint URL

        Raises:
            ValueError: If endpoint cannot be resolved
        """
        # If full endpoint provided, use it
        if endpoint:
            return endpoint

        # Try to resolve from settings
        try:
            settings = get_service_endpoints_settings()
            resolved = settings.get_endpoint(service_name, action)
            if resolved:
                logger.debug(f"📍 Resolved {service_name}.{action} → {resolved}")
                return resolved
        except Exception as e:
            logger.warning(f"⚠️  Failed to load settings: {e}")

        # No endpoint found
        raise ValueError(
            f"Endpoint not provided and cannot resolve from settings. "
            f"Either provide 'endpoint' parameter or configure service_endpoints.{service_name} in settings.yaml"
        )

    async def call_audio_service(
        self,
        service_name: str,
        audio_data: bytes,
        sample_rate: int = 16000,
        metadata: Dict[str, Any] = None,
        priority: Priority = Priority.NORMAL,
        force_protocol: Optional[str] = None,
        endpoint: Optional[str] = None,
        endpoint_name: str = 'audio'
    ) -> Dict[str, Any]:
        """
        Call service with audio data using optimal protocol

        Args:
            service_name: Service identifier (e.g., "ultravox", "stt")
            audio_data: Raw audio bytes (PCM int16)
            sample_rate: Sample rate in Hz
            metadata: Optional metadata dict
            priority: Request priority level
            force_protocol: Force specific protocol (for testing)
            endpoint: Optional full URL endpoint (if not provided, resolves from settings)
            endpoint_name: Endpoint name to lookup in settings (default: 'audio')

        Returns:
            Response dict with 'text', 'metadata', 'protocol_used', etc.
        """
        if not self.session:
            await self.initialize()

        # Record activity for auto-scaling
        self._record_service_activity(service_name)

        # Resolve endpoint URL (handles internal/external routing)
        # IMPORTANT: Check if service is internal FIRST before looking at settings.yaml
        if endpoint:
            # If endpoint is provided, use _resolve_endpoint_url for proper routing
            endpoint_url = self._resolve_endpoint_url(service_name, endpoint, endpoint_name)
        elif self.is_service_internal(service_name):
            # Internal service: route through Service Manager
            # Use /audio as default path for audio services (or map endpoint_name to path)
            endpoint_path = f"/{endpoint_name}" if endpoint_name else '/audio'
            endpoint_url = self._resolve_service_url(service_name, endpoint_path)
            logger.debug(f"🔗 Internal service {service_name}, routing through Service Manager: {endpoint_url}")
        else:
            # External service: try settings.yaml
            try:
                endpoint_url = self._resolve_endpoint(service_name, endpoint, endpoint_name)
            except ValueError:
                # Fallback to default external service URL
                endpoint_path = f"/{endpoint_name}" if endpoint_name else '/audio'
                endpoint_url = self._resolve_service_url(service_name, endpoint_path)

        # 1. Select optimal protocol
        if force_protocol:
            protocol_name = force_protocol
            logger.info(f"🔒 Forcing protocol: {protocol_name}")
        else:
            protocol_name = self._select_optimal_protocol(
                service_name=service_name,
                data_size=len(audio_data),
                priority=priority,
                data_type='audio'
            )

        protocol = self.protocols[protocol_name]
        logger.info(f"🎯 Selected protocol: {protocol_name} for {service_name}")

        # 2. Handle protocol-specific logic
        start_time = time.time()
        try:
            if protocol_name == 'direct':
                # Direct call (zero overhead!)
                from urllib.parse import urlparse
                parsed = urlparse(endpoint_url)
                endpoint_path = parsed.path or '/audio'

                # Prepare payload
                payload = {
                    'audio_data': audio_data,
                    'sample_rate': sample_rate
                }
                if metadata:
                    payload.update(metadata)

                # Call service directly
                result = await protocol.call_service(
                    service_name=service_name,
                    endpoint=endpoint_path,
                    payload=payload,
                    metadata=metadata
                )

                # Record success
                latency_ms = (time.time() - start_time) * 1000
                self._record_success(service_name, protocol_name, latency_ms)

                result['protocol_used'] = protocol_name
                result['latency_ms'] = latency_ms

                logger.info(f"✅ {service_name} succeeded with {protocol_name} ({latency_ms:.0f}ms)")
                return result

            elif protocol_name == 'grpc':
                # Extract host and port from endpoint_url
                from urllib.parse import urlparse
                parsed = urlparse(endpoint_url)
                http_port = parsed.port or 8000

                # Calculate gRPC port: grpc_port = 50000 + (http_port % 1000)
                grpc_port = 50000 + (http_port % 1000)
                host = parsed.hostname or 'localhost'

                logger.debug(f"🔌 Using gRPC: {host}:{grpc_port} (from HTTP port {http_port})")

                # Call via gRPC generic service
                # Prepare payload as JSON for generic call
                import json
                payload_dict = {
                    'audio_data': audio_data.hex(),  # Convert bytes to hex string for JSON payload
                    'sample_rate': sample_rate
                }
                if metadata:
                    payload_dict['metadata'] = metadata

                payload_bytes = json.dumps(payload_dict).encode('utf-8')

                # Extract endpoint path from URL
                endpoint_path = parsed.path or '/audio'

                result = await protocol.call_generic_service(
                    service_name=service_name,
                    endpoint=endpoint_path,
                    payload=payload_bytes,
                    service_host=host,
                    service_port=grpc_port,
                    metadata=metadata
                )

                # Parse response payload
                response_payload = result.get('payload', b'{}')
                if isinstance(response_payload, bytes):
                    response_data = json.loads(response_payload.decode('utf-8'))
                else:
                    response_data = response_payload

                # Record success
                latency_ms = (time.time() - start_time) * 1000
                self._record_success(service_name, protocol_name, latency_ms)

                response_data['protocol_used'] = protocol_name
                response_data['latency_ms'] = latency_ms

                logger.info(f"✅ {service_name} succeeded with {protocol_name} ({latency_ms:.0f}ms)")
                return response_data

            else:
                # HTTP-based protocols (binary, json)
                # 2. Encode data
                encoded_data, headers = protocol.encode_audio(audio_data, sample_rate, metadata)

                # 3. Try primary protocol via HTTP (30s timeout for audio processing)
                async with self.session.post(endpoint_url, data=encoded_data, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")

                    response_data = await response.read()
                    response_ct = response.headers.get('Content-Type', protocol.content_type)

                    # Decode response
                    result = protocol.decode_audio_response(response_data, response_ct)

                    # Record success
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_success(service_name, protocol_name, latency_ms)

                    result['protocol_used'] = protocol_name
                    result['latency_ms'] = latency_ms

                    logger.info(f"✅ {service_name} succeeded with {protocol_name} ({latency_ms:.0f}ms)")
                    return result

        except Exception as e:
            logger.warning(f"⚠️  {service_name} failed with {protocol_name}: {e}")

            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self._record_failure(service_name, protocol_name, latency_ms, str(e))

            # 4. Try fallback protocol (if not forced and different from primary)
            if not force_protocol:
                fallback_name = self._get_fallback_protocol(protocol_name, service_name)
                if fallback_name != protocol_name:
                    logger.info(f"🔄 Trying fallback protocol: {fallback_name}")

                    return await self.call_audio_service(
                        service_name, audio_data, sample_rate,
                        metadata, priority, force_protocol=fallback_name,
                        endpoint=endpoint, endpoint_name=endpoint_name
                    )

            # No fallback available or fallback also failed
            raise Exception(f"All protocols failed for {service_name}: {e}")

    async def call_text_service(
        self,
        service_name: str,
        text: str,
        metadata: Dict[str, Any] = None,
        priority: Priority = Priority.NORMAL,
        force_protocol: Optional[str] = None,
        endpoint: Optional[str] = None,
        endpoint_name: str = 'text',
        extra_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Call service with text data using optimal protocol

        Args:
            service_name: Name of the service to call
            text: Text data to send
            metadata: Additional metadata dict
            priority: Request priority level
            force_protocol: Force specific protocol ('http_binary' or 'json')
            endpoint: Endpoint path (e.g., '/health') or full URL
            endpoint_name: Endpoint name for settings lookup
            extra_params: Extra parameters to merge with metadata
        """
        if not self.session:
            await self.initialize()

        # Record activity for auto-scaling
        self._record_service_activity(service_name)

        # Merge extra_params with metadata
        if extra_params:
            if metadata is None:
                metadata = {}
            metadata = {**metadata, **extra_params}

        # Resolve endpoint URL (handles both paths and full URLs)
        endpoint_url = self._resolve_endpoint_url(service_name, endpoint, endpoint_name)

        # Similar implementation to call_audio_service but for text
        protocol_name = force_protocol or self._select_optimal_protocol(
            service_name, len(text.encode('utf-8')), priority, 'text'
        )

        protocol = self.protocols[protocol_name]
        encoded_data, headers = protocol.encode_text(text, metadata)

        start_time = time.time()
        try:
            async with self.session.post(endpoint_url, data=encoded_data, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")

                response_data = await response.read()
                result = protocol.decode_text_response(response_data, response.headers.get('Content-Type'))

                latency_ms = (time.time() - start_time) * 1000
                self._record_success(service_name, protocol_name, latency_ms)

                result['protocol_used'] = protocol_name
                result['latency_ms'] = latency_ms
                return result

        except Exception as e:
            if not force_protocol:
                fallback_name = self._get_fallback_protocol(protocol_name, service_name)
                if fallback_name != protocol_name:
                    return await self.call_text_service(
                        service_name, text, metadata, priority, fallback_name,
                        endpoint, endpoint_name, extra_params
                    )
            raise

    def _select_optimal_protocol(
        self,
        service_name: str,
        data_size: int,
        priority: Priority,
        data_type: Literal['audio', 'text']
    ) -> str:
        """
        Select best protocol based on multiple factors

        Decision logic:
        0. In-process service (composite) → Direct call (ZERO overhead!)
        1. Internal service → Direct call (ZERO overhead!)
        2. ZeroMQ (local service) → Ultra-fast inproc transport (0.01ms, 410k msg/s!)
        3. REALTIME priority → gRPC (low latency ~7ms)
        4. DEBUG priority → JSON (human-readable)
        5. Large payload (>100KB) → ZeroMQ or gRPC or Binary (efficient)
        6. Historical performance → Protocol with better success rate
        7. Default to service preference
        """
        # 0. In-process services (composite services) use direct calls
        if self.is_inprocess_service(service_name) and 'direct' in self.protocols:
            logger.debug(f"In-process service {service_name}, using direct call (zero overhead)")
            return 'direct'

        # 1. Internal services use direct calls (zero overhead!)
        if self.is_service_internal(service_name) and 'direct' in self.protocols:
            # Check if service is actually registered in DirectCallAdapter
            direct_adapter = self.protocols['direct']
            if service_name in direct_adapter.service_registry:
                logger.debug(f"Internal service {service_name}, using direct call (zero overhead)")
                return 'direct'
            else:
                logger.debug(f"Internal service {service_name} not yet registered, falling back to ZeroMQ")

        # 2. ZeroMQ for local services (PRIMARY - inproc transport, 0.01ms, 410k msg/s!)
        if self.zeromq_enabled and 'zeromq' in self.protocols:
            try:
                zeromq_adapter = self.protocols['zeromq']
                if zeromq_adapter.is_available(service_name):
                    logger.debug(f"Using ZeroMQ (inproc) for {service_name}")
                    return 'zeromq'
            except Exception as e:
                logger.debug(f"ZeroMQ unavailable for {service_name}: {e}, falling back to gRPC")

        # 4. Priority-based selection
        if priority == Priority.REALTIME:
            # For external/remote services, prefer gRPC if available
            if self.grpc_enabled and 'grpc' in self.protocols:
                return 'grpc'
            return 'http_binary'

        if priority == Priority.DEBUG:
            return 'json'

        # 5. Size-based selection
        if data_size > self.thresholds['large_payload_bytes']:
            logger.debug(f"Large payload ({data_size} bytes), preferring zeromq or grpc or http_binary")
            # Prefer ZeroMQ for large payloads if available
            if self.zeromq_enabled and 'zeromq' in self.protocols:
                try:
                    if self.protocols['zeromq'].is_available(service_name):
                        return 'zeromq'
                except Exception as e:
                    pass
            # Prefer gRPC for large payloads (remote services)
            if self.grpc_enabled and 'grpc' in self.protocols:
                return 'grpc'
            return 'http_binary'

        # 6. Performance-based selection
        service_metrics = self.metrics.get(service_name, {})
        if service_metrics:
            binary_success = service_metrics.get('http_binary', {}).get('success_rate', 0)
            json_success = service_metrics.get('json', {}).get('success_rate', 0)

            # If http_binary has low success rate, use JSON
            if binary_success < self.thresholds['min_success_rate'] and json_success > binary_success:
                logger.debug(f"HTTP Binary success rate low ({binary_success:.1%}), using JSON")
                return 'json'

        # 7. Service preference
        pref = self.preferences.get(service_name, self.preferences['default'])
        return pref.get('primary', 'zeromq')

    def _get_fallback_protocol(self, failed_protocol: str, service_name: str) -> str:
        """Get fallback protocol when primary fails"""
        pref = self.preferences.get(service_name, self.preferences['default'])

        if failed_protocol == pref.get('primary'):
            return pref.get('fallback', 'json' if failed_protocol == 'http_binary' else 'http_binary')

        # If fallback already failed, return the other one
        return 'json' if failed_protocol == 'http_binary' else 'http_binary'

    def _record_success(self, service_name: str, protocol: str, latency_ms: float) -> None:
        """Record successful call metrics"""
        if service_name not in self.metrics:
            self.metrics[service_name] = {}
        if protocol not in self.metrics[service_name]:
            self.metrics[service_name][protocol] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_latency_ms': 0,
                'success_rate': 0,
                'avg_latency_ms': 0
            }

        m = self.metrics[service_name][protocol]
        m['total_calls'] += 1
        m['successful_calls'] += 1
        m['total_latency_ms'] += latency_ms
        m['success_rate'] = m['successful_calls'] / m['total_calls']
        m['avg_latency_ms'] = m['total_latency_ms'] / m['total_calls']

    def _record_failure(self, service_name: str, protocol: str, latency_ms: float, error: str) -> None:
        """Record failed call metrics"""
        if service_name not in self.metrics:
            self.metrics[service_name] = {}
        if protocol not in self.metrics[service_name]:
            self.metrics[service_name][protocol] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_latency_ms': 0,
                'success_rate': 0,
                'avg_latency_ms': 0,
                'last_error': None
            }

        m = self.metrics[service_name][protocol]
        m['total_calls'] += 1
        m['failed_calls'] += 1
        m['total_latency_ms'] += latency_ms
        m['success_rate'] = m['successful_calls'] / m['total_calls'] if m['total_calls'] > 0 else 0
        m['avg_latency_ms'] = m['total_latency_ms'] / m['total_calls']
        m['last_error'] = error

    def get_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        if service_name:
            return self.metrics.get(service_name, {})
        return self.metrics

    def set_preference(self, service_name: str, primary: str, fallback: str) -> None:
        """Set protocol preference for a service"""
        self.preferences[service_name] = {
            'primary': primary,
            'fallback': fallback,
            'auto_optimize': True
        }
        logger.info(f"📌 Set {service_name} preference: {primary} (fallback: {fallback})")

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
        """
        Generic service call with automatic URL resolution + resilience patterns

        Args:
            service_name: Service name (e.g., 'conversation_store', 'session')
            endpoint_path: Endpoint path (e.g., '/api/conversations', '/health')
            method: HTTP method (GET, POST, PUT, DELETE)
            json_data: JSON payload
            data: Raw bytes data
            headers: Optional headers
            timeout: Request timeout in seconds
            enable_resilience: Enable circuit breaker + retry (default: True)

        Returns:
            Response dict

        Example:
            result = await comm_manager.call_service(
                'conversation_store',
                '/api/conversations',
                method='POST',
                json_data={'user_id': 'test'}
            )
        """
        # If resilience is disabled or not enabled for this call, use direct call
        if not self._resilience_enabled or not enable_resilience:
            return await self._call_service_direct(
                service_name, endpoint_path, method, json_data, data, headers, timeout
            )

        # Use resilience patterns
        try:
            # Get or create circuit breaker for this service
            circuit = self.circuit_breaker_registry.get(
                service_name,
                CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=60.0,
                    success_threshold=2,
                    timeout=timeout
                )
            )

            # Get or create retry policy for this service
            retry = self.retry_policy_registry.get(
                service_name,
                RetryPolicyConfig(
                    max_attempts=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    strategy=RetryStrategy.EXPONENTIAL,
                    jitter=True
                )
            )

            # Wrap call with retry + circuit breaker
            async def protected_call() -> Any:
                return await circuit.call(
                    self._call_service_direct,
                    service_name, endpoint_path, method, json_data, data, headers, timeout
                )

            result = await retry.execute(protected_call)
            return result

        except CircuitBreakerError as e:
            logger.error(f"🔴 Circuit breaker OPEN for {service_name}: {e}")
            raise Exception(f"Service {service_name} unavailable (circuit breaker open)")

        except RetryExhaustedError as e:
            logger.error(f"❌ Retry exhausted for {service_name}: {e}")
            raise Exception(f"Service {service_name} failed after all retries")

    async def _call_service_direct(
        self,
        service_name: str,
        endpoint_path: str,
        method: str = 'POST',
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Direct service call without resilience patterns (internal use)
        """
        if not self.session:
            await self.initialize()

        # Resolve URL based on execution mode
        url = self._resolve_service_url(service_name, endpoint_path)

        # Prepare headers
        req_headers = headers or {}

        # Inject trace context for distributed tracing
        req_headers = inject_trace_context(req_headers)

        start_time = time.time()
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=json_data,
                data=data,
                headers=req_headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                latency_ms = (time.time() - start_time) * 1000

                if response.status not in (200, 201):
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                # Try to parse as JSON, fallback to text
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    result = await response.json()
                else:
                    text = await response.text()
                    result = {'response': text}

                result['latency_ms'] = latency_ms
                logger.debug(f"✅ {service_name} {method} {endpoint_path} succeeded ({latency_ms:.0f}ms)")
                return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ {service_name} {method} {endpoint_path} failed: {e}")
            raise Exception(f"{service_name} {method} {endpoint_path} error: {e}")

    async def stream_request(
        self,
        service_name: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.REALTIME,
        force_protocol: Optional[str] = None,
        endpoint: Optional[str] = None,
        endpoint_name: str = 'synthesize',
        chunk_size: int = 8192
    ):
        """
        Stream response from service with progressive chunk delivery

        Args:
            service_name: Service identifier (e.g., "tts")
            payload: Request payload (will be sent as JSON)
            priority: Request priority level
            force_protocol: Force specific protocol (for testing)
            endpoint: Optional full URL endpoint
            endpoint_name: Endpoint name to lookup in settings
            chunk_size: Size of chunks to yield (default: 8KB browser-like)

        Yields:
            bytes: Audio/data chunks as they arrive

        Example:
            async for chunk in comm_manager.stream_request('tts', {'text': '...', 'stream': True}):
                await send_to_webrtc(chunk)
        """
        if not self.session:
            await self.initialize()

        # Resolve endpoint URL
        endpoint_url = self._resolve_endpoint(service_name, endpoint, endpoint_name)

        # Select protocol (for streaming, JSON is simpler and well-supported)
        if force_protocol:
            protocol_name = force_protocol
        else:
            # For streaming, prefer JSON as it's simpler for HTTP streaming
            # Binary protocol could be added later for even better performance
            protocol_name = 'json'

        protocol = self.protocols[protocol_name]
        logger.info(f"🎯 Streaming with {protocol_name} protocol to {service_name}")

        # Encode payload
        encoded_data, headers = protocol.encode_text(str(payload), {})

        # For JSON payload, use proper JSON encoding
        import json
        headers['Content-Type'] = 'application/json'
        encoded_data = json.dumps(payload).encode('utf-8')

        start_time = time.time()
        first_chunk_time = None
        total_bytes = 0
        chunk_count = 0

        try:
            async with self.session.post(endpoint_url, data=encoded_data, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                logger.info(f"✅ Streaming started from {service_name}")

                # Stream chunks
                async for chunk in response.content.iter_chunked(chunk_size):
                    if chunk:
                        chunk_count += 1
                        total_bytes += len(chunk)

                        if first_chunk_time is None:
                            first_chunk_time = time.time() - start_time
                            logger.info(f"⚡ First chunk arrived at {first_chunk_time*1000:.1f}ms")

                        yield chunk

                # Record success
                total_time = time.time() - start_time
                self._record_success(service_name, f"{protocol_name}_stream", total_time * 1000)

                logger.info(
                    f"✅ Streaming completed: {chunk_count} chunks, "
                    f"{total_bytes:,} bytes, {total_time*1000:.0f}ms total, "
                    f"{first_chunk_time*1000:.1f}ms to first chunk"
                )

        except Exception as e:
            total_time = time.time() - start_time
            self._record_failure(service_name, f"{protocol_name}_stream", total_time * 1000, str(e))
            logger.error(f"❌ Streaming failed from {service_name}: {e}")
            raise

    async def stream_json_events(
        self,
        service_name: str,
        action: str,
        payload: Dict[str, Any],
        endpoint: Optional[str] = None,
        priority: Priority = Priority.REALTIME,
        force_protocol: Optional[str] = None
    ):
        """
        Stream JSON events from service with transparent protocol selection.

        Each line of the response should be a valid JSON object.
        The Communication Manager transparently selects the best protocol:
        ZeroMQ (primary, 0.01ms) → gRPC → HTTP Binary → JSON (fallback)

        Args:
            service_name: Service identifier (e.g., "orchestrator")
            action: Action/endpoint name on the service to call (e.g., "conversation-stream")
            payload: Request payload (dict or any JSON-serializable object)
            endpoint: Optional full URL endpoint (if not provided, CM resolves automatically)
            priority: Request priority level
            force_protocol: Force specific protocol (for testing)

        Yields:
            dict: Parsed JSON objects from the stream

        Example:
            async for event in comm_manager.stream_json_events(
                service_name='orchestrator',
                action='conversation-stream',
                payload={'session_id': '123', 'audio': audio_b64}
            ):
                print(f"Event: {event.get('event')}, Data: {event.get('data')}")
        """
        if not self.session:
            await self.initialize()

        # Resolve endpoint URL (Communication Manager handles port, protocol, etc.)
        endpoint_url = self._resolve_endpoint(service_name, endpoint, action)

        # For JSON event streaming, use JSON protocol (HTTP with JSON lines)
        # Binary/ZeroMQ would require special handling for JSON event reconstruction
        protocol_name = force_protocol or 'json'
        logger.info(f"📡 Streaming JSON events with {protocol_name} protocol from {service_name}")

        # Encode payload
        import json
        headers = {'Content-Type': 'application/json'}
        encoded_data = json.dumps(payload).encode('utf-8')

        start_time = time.time()
        first_event_time = None
        event_count = 0
        total_bytes = 0

        try:
            async with self.session.post(endpoint_url, data=encoded_data, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                logger.info(f"✅ JSON event stream started from {service_name}")

                # Stream JSON events line by line
                async for line in response.content:
                    if not line.strip():
                        continue  # Skip empty lines

                    event_count += 1
                    total_bytes += len(line)

                    if first_event_time is None:
                        first_event_time = time.time() - start_time
                        logger.info(f"⚡ First JSON event arrived at {first_event_time*1000:.1f}ms")

                    try:
                        # Parse JSON event
                        event_data = json.loads(line.decode('utf-8') if isinstance(line, bytes) else line)
                        yield event_data

                        logger.debug(
                            f"📨 Received event: {event_data.get('event', 'unknown')} "
                            f"(seq={event_data.get('sequence', '?')})"
                        )

                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Invalid JSON in event stream: {e}")
                        continue

                # Record success
                total_time = time.time() - start_time
                self._record_success(service_name, f"{protocol_name}_json_stream", total_time * 1000)

                logger.info(
                    f"✅ JSON event stream completed: {event_count} events, "
                    f"{total_bytes:,} bytes, {total_time*1000:.0f}ms total, "
                    f"{first_event_time*1000:.1f}ms to first event"
                )

        except Exception as e:
            total_time = time.time() - start_time
            self._record_failure(
                service_name,
                f"{protocol_name}_json_stream",
                total_time * 1000,
                str(e)
            )
            logger.error(f"❌ JSON event stream failed from {service_name}: {e}")
            raise

    # Service-specific methods
    async def external_ultravox(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        metadata: Dict[str, Any] = None,
        priority: Priority = Priority.NORMAL
    ) -> Dict[str, Any]:
        """
        Call External Ultravox service (Groq STT + LLM pipeline)

        Args:
            audio_data: Raw audio bytes (PCM int16)
            sample_rate: Sample rate in Hz
            metadata: Optional metadata (language, max_tokens, temperature, etc.)
            priority: Request priority level

        Returns:
            Response dict with 'text', 'transcript', 'protocol_used', etc.
        """
        return await self.call_audio_service(
            service_name="external_ultravox",
            audio_data=audio_data,
            sample_rate=sample_rate,
            metadata=metadata,
            priority=priority,
            endpoint_name='generate'  # Use /generate endpoint instead of default /audio
        )

    def get_metrics_report(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics report for all services

        Returns:
            Dict with metrics for each service including:
            - Total calls
            - Average latency
            - Protocol breakdown
            - Success rate
        """
        report = {}

        for service_name in self.metrics.keys():
            service_metrics = self.metrics[service_name]

            if not service_metrics:
                continue

            # Calculate aggregates
            total_calls = 0
            total_latency = 0.0
            protocol_stats = {}

            for protocol, protocol_metrics in service_metrics.items():
                calls = protocol_metrics.get('calls', 0)
                total_calls += calls

                if calls > 0:
                    avg_latency = protocol_metrics.get('total_latency', 0) / calls
                    total_latency += protocol_metrics.get('total_latency', 0)

                    protocol_stats[protocol] = {
                        'calls': calls,
                        'avg_latency_ms': avg_latency,
                        'success_rate': protocol_metrics.get('successes', 0) / calls if calls > 0 else 0.0
                    }

            report[service_name] = {
                'total_calls': total_calls,
                'avg_latency_ms': total_latency / total_calls if total_calls > 0 else 0.0,
                'protocol_breakdown': protocol_stats
            }

        return report

    def clear_metrics(self) -> None:
        """Clear all collected metrics"""
        self.metrics = {}

    def get_resilience_stats(self) -> Dict[str, Any]:
        """
        Get resilience statistics (circuit breakers + retry policies)

        Returns:
            Dict with circuit breaker and retry policy stats for all services
        """
        return {
            "circuit_breakers": self.circuit_breaker_registry.get_all_stats(),
            "retry_policies": self.retry_policy_registry.get_all_stats(),
            "resilience_enabled": self._resilience_enabled
        }

    # ============================================================================
    # Phase 4: Port Cache Invalidation (Dynamic Port Pool Integration)
    # ============================================================================

    def invalidate_service_port(self, service_name: str, new_port: int) -> Optional[int]:
        """
        Invalidate cached port for a service and update to new port

        This method is called when a service's port changes (e.g., during
        auto-recovery). It updates both the service_config and default_service_ports
        caches so that subsequent calls to _resolve_service_url() use the new port.

        Args:
            service_name: Name of the service (e.g., 'orchestrator', 'session')
            new_port: New port number for the service

        Returns:
            Previous port number or None if service was not cached

        Example:
            # When ServiceLauncher auto-recovers orchestrator on port 9051:
            old_port = comm_manager.invalidate_service_port('orchestrator', 9051)
            # old_port = 9050, now all calls route to port 9051
        """
        old_port = None

        # Update service_config (used by _resolve_service_url for external services)
        if service_name in self.service_config:
            old_port = self.service_config[service_name].get('port')
            self.service_config[service_name]['port'] = new_port
            logger.info(
                f"📝 Updated service_config cache: {service_name} port "
                f"{old_port} → {new_port}"
            )
        else:
            # Service not in config yet, add it
            self.service_config[service_name] = {
                'execution_mode': 'external',
                'port': new_port
            }
            logger.info(f"📝 Added {service_name} to service_config cache (port {new_port})")

        # Update default_service_ports (fallback for _resolve_service_url)
        if service_name in self.default_service_ports:
            if old_port is None:
                old_port = self.default_service_ports[service_name]
            self.default_service_ports[service_name] = new_port
            logger.info(
                f"📝 Updated default_service_ports cache: {service_name} port "
                f"{old_port} → {new_port}"
            )
        else:
            self.default_service_ports[service_name] = new_port
            logger.info(
                f"📝 Added {service_name} to default_service_ports cache (port {new_port})"
            )

        # Clear metrics for the service (fresh start on new port)
        if service_name in self.metrics:
            logger.debug(f"🧹 Cleared metrics for {service_name} (port changed)")
            del self.metrics[service_name]

        return old_port

    def get_cached_service_port(self, service_name: str) -> Optional[int]:
        """
        Get cached port for a service

        Args:
            service_name: Service name

        Returns:
            Cached port number or None if not cached
        """
        # Try service_config first (most accurate for external services)
        if service_name in self.service_config:
            port = self.service_config[service_name].get('port')
            if port:
                return port

        # Fallback to default_service_ports
        return self.default_service_ports.get(service_name)

    def get_all_cached_ports(self) -> Dict[str, int]:
        """
        Get all cached service ports

        Returns:
            Dictionary of service_name → port for all cached services
        """
        cached_ports = {}

        # Collect from service_config
        for service_name, config in self.service_config.items():
            port = config.get('port')
            if port:
                cached_ports[service_name] = port

        # Add from default_service_ports (if not already in service_config)
        for service_name, port in self.default_service_ports.items():
            if service_name not in cached_ports:
                cached_ports[service_name] = port

        return cached_ports


# Singleton instance
_comm_manager_instance: Optional[ServiceCommunicationManager] = None


def get_communication_manager() -> ServiceCommunicationManager:
    """
    Get global Communication Manager instance (singleton)

    Returns:
        ServiceCommunicationManager instance
    """
    global _comm_manager_instance

    if _comm_manager_instance is None:
        _comm_manager_instance = ServiceCommunicationManager()

    return _comm_manager_instance
