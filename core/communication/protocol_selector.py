"""
Protocol Selector - Chooses optimal protocol for service communication.

Part of Communication Manager refactoring (Phase 2).
Extracted from monolithic communication_manager.py.

Complete implementation with all selection logic from ServiceCommunicationManager.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
import os
from loguru import logger

from .interfaces import CommunicationProtocol, Priority, IProtocolSelector

# Avoid circular imports
if TYPE_CHECKING:
    from .service_registry import ServiceRegistry
    from .metrics_collector import MetricsCollector


class ProtocolSelector:
    """
    Selects the optimal protocol for inter-service communication.

    Selection logic (from ServiceCommunicationManager):
    0. In-process service (composite) → Direct call (ZERO overhead!)
    1. Internal service → Direct call (ZERO overhead!)
    2. ZeroMQ (local service) → Ultra-fast inproc transport (0.01ms, 410k msg/s!)
    3. REALTIME priority → gRPC (low latency ~7ms)
    4. DEBUG priority → JSON (human-readable)
    5. Large payload (>100KB) → ZeroMQ or gRPC or Binary (efficient)
    6. Historical performance → Protocol with better success rate
    7. Default to service preference

    SOLID Principles:
    - Single Responsibility: Only handles protocol selection
    - Open/Closed: Easy to add new protocols
    - Dependency Inversion: Returns Protocol enum, not implementation
    """

    def __init__(
        self,
        service_registry: Optional['ServiceRegistry'] = None,
        metrics_collector: Optional['MetricsCollector'] = None,
        zeromq_enabled: bool = True,
        grpc_enabled: bool = True,
        direct_enabled: bool = True
    ):
        """
        Initialize protocol selector.

        Args:
            service_registry: Registry for in-process services (optional)
            metrics_collector: Metrics collector for performance-based selection (optional)
            zeromq_enabled: Enable ZeroMQ protocol (default: True)
            grpc_enabled: Enable gRPC protocol (default: True)
            direct_enabled: Enable direct calls (default: True)
        """
        self.service_registry = service_registry
        self.metrics_collector = metrics_collector

        # Protocol availability (from env vars)
        self.zeromq_enabled = zeromq_enabled and (
            os.getenv("ENABLE_ZEROMQ", "true").lower() == "true"
        )
        self.grpc_enabled = grpc_enabled and (
            os.getenv("ENABLE_GRPC", "true").lower() == "true"
        )
        self.direct_enabled = direct_enabled

        # Thresholds for protocol selection (from ServiceCommunicationManager)
        self.thresholds = {
            'large_payload_bytes': 100_000,  # Use binary for payloads >100KB
            'min_success_rate': 0.8,         # Switch if success rate < 80%
            'max_latency_ms': 100            # Prefer binary if latency critical
        }

        # Service-specific preferences (ZeroMQ → gRPC → HTTP Binary → JSON)
        # Based on ServiceCommunicationManager lines 104-133
        self.preferences: Dict[str, Dict[str, str]] = {
            'default': {
                'primary': 'zeromq',         # 🚀 ZeroMQ is PRIMARY (0.01ms, 410k msg/s!)
                'secondary': 'grpc',         # ⚡ gRPC is FALLBACK 1 (~7ms)
                'tertiary': 'http_binary',   # 🔄 HTTP Binary is FALLBACK 2
                'fallback': 'json',          # 🐌 JSON is LAST RESORT
                'auto_optimize': True
            },
            # Audio services benefit from binary protocols
            'stt': {
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
            },
            # LLM can use any protocol
            'llm': {
                'primary': 'zeromq',
                'secondary': 'grpc',
                'tertiary': 'http_binary',
                'fallback': 'json'
            }
        }

        # Protocol usage tracking
        self._protocol_usage: Dict[str, int] = {}

        logger.info(
            f"📡 ProtocolSelector initialized "
            f"(zeromq={self.zeromq_enabled}, grpc={self.grpc_enabled}, "
            f"direct={self.direct_enabled})"
        )

    def select_protocol(
        self,
        service_name: str,
        data_size: int,
        priority: Priority,
        data_type: str,
        is_internal: bool = False
    ) -> CommunicationProtocol:
        """
        Select optimal protocol for service communication.

        Complete logic from ServiceCommunicationManager._select_optimal_protocol (lines 730-813).

        Args:
            service_name: Target service identifier
            data_size: Payload size in bytes
            priority: Request priority level
            data_type: Type of data ('audio', 'text', 'json')
            is_internal: Whether service is in-process

        Returns:
            Selected protocol enum

        Example:
            selector = ProtocolSelector()
            protocol = selector.select_protocol(
                service_name="llm",
                data_size=1024,
                priority=Priority.NORMAL,
                data_type="text",
                is_internal=False
            )
        """
        # 0. In-process services (composite services) use direct calls
        if self.service_registry and self.service_registry.is_registered(service_name):
            if self.direct_enabled:
                self._record_usage('direct')
                logger.debug(f"In-process service {service_name}, using direct call (zero overhead)")
                return CommunicationProtocol.DIRECT

        # 1. Internal services use direct calls (zero overhead!)
        if is_internal and self.direct_enabled:
            # Check if service is actually registered in registry
            if self.service_registry and self.service_registry.is_registered(service_name):
                self._record_usage('direct')
                logger.debug(f"Internal service {service_name}, using direct call (zero overhead)")
                return CommunicationProtocol.DIRECT
            else:
                logger.debug(f"Internal service {service_name} not yet registered, falling back to ZeroMQ")

        # 2. ZeroMQ for local services (PRIMARY - inproc transport, 0.01ms, 410k msg/s!)
        if self.zeromq_enabled:
            # ZeroMQ is available for local services
            # TODO: Add is_available() check for ZeroMQ adapter
            self._record_usage('zeromq')
            logger.debug(f"Using ZeroMQ (inproc) for {service_name}")
            return CommunicationProtocol.ZEROMQ

        # 4. Priority-based selection
        if priority == Priority.REALTIME:
            # For external/remote services, prefer gRPC if available
            if self.grpc_enabled:
                self._record_usage('grpc')
                return CommunicationProtocol.GRPC
            self._record_usage('http_binary')
            return CommunicationProtocol.HTTP_BINARY

        if priority == Priority.DEBUG:
            self._record_usage('json')
            return CommunicationProtocol.HTTP_JSON

        # 5. Size-based selection
        if data_size > self.thresholds['large_payload_bytes']:
            logger.debug(f"Large payload ({data_size} bytes), preferring zeromq or grpc or http_binary")
            # Prefer ZeroMQ for large payloads if available
            if self.zeromq_enabled:
                self._record_usage('zeromq')
                return CommunicationProtocol.ZEROMQ
            # Prefer gRPC for large payloads (remote services)
            if self.grpc_enabled:
                self._record_usage('grpc')
                return CommunicationProtocol.GRPC
            self._record_usage('http_binary')
            return CommunicationProtocol.HTTP_BINARY

        # 6. Performance-based selection (using metrics)
        if self.metrics_collector:
            service_metrics = self.metrics_collector.get_metrics(service_name)
            if service_metrics and 'protocols' in service_metrics:
                protocols_data = service_metrics['protocols']

                # Check success rates for protocols
                binary_success = protocols_data.get('http_binary', {}).get('success_rate', 0)
                json_success = protocols_data.get('json', {}).get('success_rate', 0)

                # If http_binary has low success rate, use JSON
                if binary_success < self.thresholds['min_success_rate'] and json_success > binary_success:
                    logger.debug(f"HTTP Binary success rate low ({binary_success:.1%}), using JSON")
                    self._record_usage('json')
                    return CommunicationProtocol.HTTP_JSON

        # 7. Service preference (default or custom)
        pref = self.preferences.get(service_name, self.preferences['default'])
        primary_protocol = pref.get('primary', 'zeromq')

        # Convert string to enum
        protocol_map = {
            'zeromq': CommunicationProtocol.ZEROMQ,
            'grpc': CommunicationProtocol.GRPC,
            'http_binary': CommunicationProtocol.HTTP_BINARY,
            'json': CommunicationProtocol.HTTP_JSON,
            'direct': CommunicationProtocol.DIRECT
        }

        selected = protocol_map.get(primary_protocol, CommunicationProtocol.HTTP_JSON)
        self._record_usage(primary_protocol)

        logger.debug(f"Selected {selected.value} for {service_name} (service preference)")
        return selected

    def get_fallback_protocol(
        self,
        failed_protocol: CommunicationProtocol,
        service_name: str
    ) -> CommunicationProtocol:
        """
        Get fallback protocol when primary fails.

        Based on ServiceCommunicationManager._get_fallback_protocol (lines 815-823).

        Args:
            failed_protocol: Protocol that failed
            service_name: Target service

        Returns:
            Fallback protocol to try

        Example:
            fallback = selector.get_fallback_protocol(
                CommunicationProtocol.GRPC,
                "llm"
            )
        """
        pref = self.preferences.get(service_name, self.preferences['default'])

        # Get protocol preference chain
        primary = pref.get('primary', 'zeromq')
        secondary = pref.get('secondary', 'grpc')
        tertiary = pref.get('tertiary', 'http_binary')
        fallback = pref.get('fallback', 'json')

        failed_str = failed_protocol.value

        # Determine fallback based on what failed
        if failed_str == primary:
            fallback_str = secondary
        elif failed_str == secondary:
            fallback_str = tertiary
        elif failed_str == tertiary:
            fallback_str = fallback
        else:
            # Default fallback
            fallback_str = 'json' if failed_str == 'http_binary' else 'http_binary'

        # Convert to enum
        protocol_map = {
            'zeromq': CommunicationProtocol.ZEROMQ,
            'grpc': CommunicationProtocol.GRPC,
            'http_binary': CommunicationProtocol.HTTP_BINARY,
            'json': CommunicationProtocol.HTTP_JSON,
            'direct': CommunicationProtocol.DIRECT
        }

        fallback_protocol = protocol_map.get(fallback_str, CommunicationProtocol.HTTP_JSON)

        logger.debug(f"Fallback for {failed_protocol.value} → {fallback_protocol.value}")
        return fallback_protocol

    def set_preference(
        self,
        service_name: str,
        primary: CommunicationProtocol,
        fallback: CommunicationProtocol
    ) -> None:
        """
        Set protocol preference for a specific service.

        Args:
            service_name: Service identifier
            primary: Primary protocol to use
            fallback: Fallback protocol if primary fails

        Example:
            selector.set_preference(
                "llm",
                CommunicationProtocol.GRPC,
                CommunicationProtocol.HTTP_JSON
            )
        """
        self.preferences[service_name] = {
            'primary': primary.value,
            'fallback': fallback.value,
            'auto_optimize': True
        }

        logger.info(
            f"📌 Set {service_name} preference: "
            f"{primary.value} (fallback: {fallback.value})"
        )

    def _record_usage(self, protocol_str: str) -> None:
        """
        Record protocol usage for statistics.

        Args:
            protocol_str: Protocol name as string
        """
        if protocol_str not in self._protocol_usage:
            self._protocol_usage[protocol_str] = 0
        self._protocol_usage[protocol_str] += 1

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get protocol usage statistics.

        Returns:
            Dict mapping protocol name to usage count

        Example:
            stats = selector.get_usage_stats()
            # {"zeromq": 1000, "grpc": 500, "json": 200}
        """
        return self._protocol_usage.copy()

    def reset_stats(self) -> None:
        """
        Reset usage statistics.

        Example:
            selector.reset_stats()
        """
        self._protocol_usage.clear()
        logger.debug("Protocol usage stats reset")

    def configure_thresholds(self, thresholds: Dict[str, Any]) -> None:
        """
        Configure protocol selection thresholds.

        Args:
            thresholds: Dict of threshold values
                {
                    "large_payload_bytes": int,
                    "min_success_rate": float,
                    "max_latency_ms": int
                }

        Example:
            selector.configure_thresholds({
                "large_payload_bytes": 200_000,  # 200KB
                "min_success_rate": 0.9  # 90%
            })
        """
        self.thresholds.update(thresholds)
        logger.info(f"Protocol thresholds updated: {thresholds}")

    def configure_service_preference(
        self,
        service_name: str,
        primary: str,
        secondary: Optional[str] = None,
        tertiary: Optional[str] = None,
        fallback: Optional[str] = None
    ) -> None:
        """
        Configure full protocol preference chain for a service.

        Args:
            service_name: Service identifier
            primary: Primary protocol ("zeromq", "grpc", "http_binary", "json")
            secondary: Secondary protocol (optional)
            tertiary: Tertiary protocol (optional)
            fallback: Fallback protocol (optional, defaults to "json")

        Example:
            selector.configure_service_preference(
                "llm",
                primary="grpc",
                secondary="http_binary",
                fallback="json"
            )
        """
        self.preferences[service_name] = {
            'primary': primary,
            'secondary': secondary or 'grpc',
            'tertiary': tertiary or 'http_binary',
            'fallback': fallback or 'json',
            'auto_optimize': True
        }

        logger.info(
            f"📝 Updated preference for {service_name}: "
            f"{primary} → {secondary} → {tertiary} → {fallback}"
        )

    def get_service_preference(self, service_name: str) -> Dict[str, str]:
        """
        Get protocol preference for a service.

        Args:
            service_name: Service identifier

        Returns:
            Preference dict with primary, secondary, tertiary, fallback

        Example:
            pref = selector.get_service_preference("llm")
            # {"primary": "zeromq", "secondary": "grpc", ...}
        """
        return self.preferences.get(service_name, self.preferences['default']).copy()

    def get_all_preferences(self) -> Dict[str, Dict[str, str]]:
        """
        Get all service preferences.

        Returns:
            Dict mapping service_name -> preference_dict

        Example:
            all_prefs = selector.get_all_preferences()
        """
        return self.preferences.copy()
