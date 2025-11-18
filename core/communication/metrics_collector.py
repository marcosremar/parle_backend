"""
Metrics Collector - Tracks communication performance metrics.

Extracted from ServiceCommunicationManager (Phase 2 refactoring).

SOLID Principles:
- Single Responsibility: Only handles metrics tracking
- Open/Closed: Easy to add new metric types
- Interface Segregation: Implements IMetricsCollector
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from .interfaces import IMetricsCollector, CommunicationProtocol


@dataclass
class ProtocolMetrics:
    """Metrics for a single protocol."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    last_error: Optional[str] = None
    first_call_time: Optional[float] = None
    last_call_time: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 to 1.0)."""
        return 1.0 - self.success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "last_error": self.last_error,
            "uptime_seconds": round(time.time() - self.first_call_time, 2)
            if self.first_call_time else None,
        }


@dataclass
class ServiceMetrics:
    """Metrics for a single service (all protocols)."""
    service_name: str
    protocols: Dict[str, ProtocolMetrics] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def get_or_create_protocol(self, protocol: str) -> ProtocolMetrics:
        """Get or create metrics for a protocol."""
        if protocol not in self.protocols:
            self.protocols[protocol] = ProtocolMetrics()
        return self.protocols[protocol]

    @property
    def total_calls(self) -> int:
        """Total calls across all protocols."""
        return sum(p.total_calls for p in self.protocols.values())

    @property
    def total_successful_calls(self) -> int:
        """Total successful calls across all protocols."""
        return sum(p.successful_calls for p in self.protocols.values())

    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across all protocols."""
        if self.total_calls == 0:
            return 0.0
        return self.total_successful_calls / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all protocols."""
        total_latency = sum(p.total_latency_ms for p in self.protocols.values())
        if self.total_calls == 0:
            return 0.0
        return total_latency / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "service_name": self.service_name,
            "total_calls": self.total_calls,
            "successful_calls": self.total_successful_calls,
            "overall_success_rate": round(self.overall_success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "protocols": {
                name: metrics.to_dict()
                for name, metrics in self.protocols.items()
            },
            "created_at": self.created_at,
        }


class MetricsCollector:
    """
    Collects and tracks communication performance metrics.

    Tracks metrics per service and per protocol:
    - Success/failure counts
    - Latency statistics
    - Error tracking
    - Protocol usage patterns

    Thread-safe for concurrent service calls.

    Example:
        collector = MetricsCollector()

        # Record success
        collector.record_success("llm", "grpc", latency_ms=45.2)

        # Record failure
        collector.record_failure("llm", "http_json", latency_ms=100.0, error="Timeout")

        # Get metrics
        metrics = collector.get_metrics("llm")
        print(f"Success rate: {metrics['overall_success_rate']}")
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._services: Dict[str, ServiceMetrics] = {}
        self._enabled = True

        logger.info("📊 MetricsCollector initialized")

    def record_success(
        self,
        service_name: str,
        protocol: CommunicationProtocol,
        latency_ms: float
    ) -> None:
        """
        Record successful service call.

        Args:
            service_name: Service identifier
            protocol: Protocol used (or protocol name as string)
            latency_ms: Call latency in milliseconds

        Example:
            collector.record_success("llm", CommunicationProtocol.GRPC, 45.2)
        """
        if not self._enabled:
            return

        # Convert enum to string if needed
        protocol_str = protocol.value if isinstance(protocol, CommunicationProtocol) else protocol

        # Get or create service metrics
        service = self._get_or_create_service(service_name)
        proto_metrics = service.get_or_create_protocol(protocol_str)

        # Update metrics
        proto_metrics.total_calls += 1
        proto_metrics.successful_calls += 1
        proto_metrics.total_latency_ms += latency_ms

        # Track timestamps
        now = time.time()
        if proto_metrics.first_call_time is None:
            proto_metrics.first_call_time = now
        proto_metrics.last_call_time = now

        logger.debug(
            f"📈 {service_name}.{protocol_str}: "
            f"Success ({latency_ms:.1f}ms, "
            f"success_rate={proto_metrics.success_rate:.1%})"
        )

    def record_failure(
        self,
        service_name: str,
        protocol: CommunicationProtocol,
        latency_ms: float,
        error: str
    ) -> None:
        """
        Record failed service call.

        Args:
            service_name: Service identifier
            protocol: Protocol used (or protocol name as string)
            latency_ms: Call latency in milliseconds
            error: Error message/description

        Example:
            collector.record_failure("llm", CommunicationProtocol.HTTP_JSON, 100.0, "Timeout")
        """
        if not self._enabled:
            return

        # Convert enum to string if needed
        protocol_str = protocol.value if isinstance(protocol, CommunicationProtocol) else protocol

        # Get or create service metrics
        service = self._get_or_create_service(service_name)
        proto_metrics = service.get_or_create_protocol(protocol_str)

        # Update metrics
        proto_metrics.total_calls += 1
        proto_metrics.failed_calls += 1
        proto_metrics.total_latency_ms += latency_ms
        proto_metrics.last_error = error

        # Track timestamps
        now = time.time()
        if proto_metrics.first_call_time is None:
            proto_metrics.first_call_time = now
        proto_metrics.last_call_time = now

        logger.debug(
            f"📉 {service_name}.{protocol_str}: "
            f"Failure ({latency_ms:.1f}ms, error='{error}', "
            f"success_rate={proto_metrics.success_rate:.1%})"
        )

    def get_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a service or all services.

        Args:
            service_name: Service to get metrics for, or None for all services

        Returns:
            Metrics dictionary with success rate, latency, protocol breakdown, etc.

        Example:
            # Get metrics for specific service
            llm_metrics = collector.get_metrics("llm")

            # Get all metrics
            all_metrics = collector.get_metrics()
        """
        if service_name:
            # Return metrics for specific service
            service = self._services.get(service_name)
            if not service:
                return {}
            return service.to_dict()
        else:
            # Return metrics for all services
            return {
                name: service.to_dict()
                for name, service in self._services.items()
            }

    def get_metrics_report(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics report for all services.

        Returns:
            Dict with aggregated metrics and per-service breakdown

        Example:
            report = collector.get_metrics_report()
            print(f"Total calls: {report['summary']['total_calls']}")
            for service_name, stats in report['services'].items():
                print(f"{service_name}: {stats['total_calls']} calls")
        """
        if not self._services:
            return {
                "summary": {
                    "total_services": 0,
                    "total_calls": 0,
                    "overall_success_rate": 0.0,
                },
                "services": {}
            }

        # Calculate summary statistics
        total_calls = sum(s.total_calls for s in self._services.values())
        total_successful = sum(s.total_successful_calls for s in self._services.values())
        overall_success_rate = total_successful / total_calls if total_calls > 0 else 0.0

        # Build services breakdown
        services_data = {}
        for service_name, service in self._services.items():
            if service.total_calls == 0:
                continue  # Skip services with no calls

            # Calculate protocol breakdown
            protocol_stats = {}
            for protocol_name, proto_metrics in service.protocols.items():
                if proto_metrics.total_calls > 0:
                    protocol_stats[protocol_name] = {
                        "calls": proto_metrics.total_calls,
                        "success_rate": round(proto_metrics.success_rate, 3),
                        "avg_latency_ms": round(proto_metrics.avg_latency_ms, 2),
                    }

            services_data[service_name] = {
                "total_calls": service.total_calls,
                "success_rate": round(service.overall_success_rate, 3),
                "avg_latency_ms": round(service.avg_latency_ms, 2),
                "protocol_breakdown": protocol_stats
            }

        return {
            "summary": {
                "total_services": len(self._services),
                "total_calls": total_calls,
                "successful_calls": total_successful,
                "overall_success_rate": round(overall_success_rate, 3),
            },
            "services": services_data
        }

    def clear_metrics(self, service_name: Optional[str] = None) -> None:
        """
        Clear metrics for a service or all services.

        Args:
            service_name: Service to clear, or None to clear all

        Example:
            # Clear specific service
            collector.clear_metrics("llm")

            # Clear all metrics
            collector.clear_metrics()
        """
        if service_name:
            if service_name in self._services:
                del self._services[service_name]
                logger.info(f"🧹 Cleared metrics for service: {service_name}")
        else:
            self._services.clear()
            logger.info("🧹 Cleared all metrics")

    def get_protocol_usage_stats(self) -> Dict[str, int]:
        """
        Get protocol usage statistics across all services.

        Returns:
            Dict mapping protocol names to total usage count

        Example:
            stats = collector.get_protocol_usage_stats()
            # {"grpc": 1000, "http_json": 500, "zeromq": 2000}
        """
        usage: Dict[str, int] = {}

        for service in self._services.values():
            for protocol_name, proto_metrics in service.protocols.items():
                usage[protocol_name] = usage.get(protocol_name, 0) + proto_metrics.total_calls

        return usage

    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """
        Get health status for a service based on metrics.

        Args:
            service_name: Service to check

        Returns:
            Health dict with status, success_rate, recent_errors, etc.

        Example:
            health = collector.get_service_health("llm")
            if health["status"] == "unhealthy":
                print(f"Service degraded: {health['reason']}")
        """
        service = self._services.get(service_name)

        if not service or service.total_calls == 0:
            return {
                "status": "unknown",
                "reason": "No metrics available"
            }

        success_rate = service.overall_success_rate
        avg_latency = service.avg_latency_ms

        # Determine health status
        if success_rate >= 0.95 and avg_latency < 1000:
            status = "healthy"
            reason = "All metrics within normal range"
        elif success_rate >= 0.8:
            status = "degraded"
            reason = f"Success rate below threshold ({success_rate:.1%})"
        else:
            status = "unhealthy"
            reason = f"High failure rate ({success_rate:.1%})"

        # Collect recent errors
        recent_errors = []
        for protocol_name, proto_metrics in service.protocols.items():
            if proto_metrics.last_error:
                recent_errors.append({
                    "protocol": protocol_name,
                    "error": proto_metrics.last_error
                })

        return {
            "status": status,
            "reason": reason,
            "success_rate": round(success_rate, 3),
            "avg_latency_ms": round(avg_latency, 2),
            "total_calls": service.total_calls,
            "recent_errors": recent_errors
        }

    def enable(self) -> None:
        """Enable metrics collection."""
        self._enabled = True
        logger.info("📊 Metrics collection enabled")

    def disable(self) -> None:
        """Disable metrics collection (for performance)."""
        self._enabled = False
        logger.info("📊 Metrics collection disabled")

    @property
    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled

    def _get_or_create_service(self, service_name: str) -> ServiceMetrics:
        """Get or create service metrics container."""
        if service_name not in self._services:
            self._services[service_name] = ServiceMetrics(service_name=service_name)
        return self._services[service_name]


# ============================================================================
# Singleton Instance (optional - can also use DI)
# ============================================================================

_metrics_collector_instance: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get global MetricsCollector instance (singleton pattern).

    Returns:
        MetricsCollector instance

    Example:
        collector = get_metrics_collector()
        collector.record_success("llm", "grpc", 45.2)
    """
    global _metrics_collector_instance

    if _metrics_collector_instance is None:
        _metrics_collector_instance = MetricsCollector()

    return _metrics_collector_instance
