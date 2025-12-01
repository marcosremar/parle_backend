"""
Prometheus Metrics and Observability
Provides metrics collection and Prometheus format export
"""

from collections import defaultdict
import logging
from threading import Lock
from typing import Any
import uuid

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Prometheus-compatible metrics collector.

    Collects and exports metrics in Prometheus format for monitoring.
    Thread-safe implementation using locks.

    Supports:
    - Counters: Monotonically increasing metrics (e.g., request count)
    - Gauges: Metrics that can go up or down (e.g., active sessions)
    - Histograms: Distribution of values (e.g., request duration)
    - Labels: Multi-dimensional metrics with labels

    Attributes:
        service_name: Name of the service for metric namespacing
        _counters: Dictionary of counter metrics
        _gauges: Dictionary of gauge metrics
        _histograms: Dictionary of histogram data
        _labels: Dictionary of metric labels
        _lock: Thread lock for thread-safe operations

    Example:
        metrics = PrometheusMetrics("api")
        metrics.counter("requests_total", labels={"method": "GET"})
        metrics.gauge("active_connections", 42)
        metrics.histogram("request_duration", 0.123)
        prometheus_format = metrics.to_prometheus()
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._labels: dict[str, dict[str, str]] = defaultdict(dict)
        self._lock = Lock()

        logger.info(f"✅ Prometheus metrics initialized for {service_name}")

    def counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None):
        """
        Increment a counter metric.

        Counters are monotonically increasing metrics (e.g., total requests).

        Args:
            name: Metric name (will be prefixed with service_name)
            value: Value to increment by (default: 1.0)
            labels: Optional dictionary of label key-value pairs
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._counters[metric_key] += value
            if labels:
                self._labels[metric_key] = labels

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None):
        """
        Set a gauge metric.

        Gauges can go up or down (e.g., active connections, memory usage).

        Args:
            name: Metric name (will be prefixed with service_name)
            value: Current value to set
            labels: Optional dictionary of label key-value pairs
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._gauges[metric_key] = value
            if labels:
                self._labels[metric_key] = labels

    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None):
        """
        Record a histogram value.

        Histograms track distribution of values (e.g., request duration).

        Args:
            name: Metric name (will be prefixed with service_name)
            value: Value to record
            labels: Optional dictionary of label key-value pairs
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._histograms[metric_key].append(value)
            if labels:
                self._labels[metric_key] = labels

    def _get_metric_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Get metric key with labels"""
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Converts all collected metrics to Prometheus exposition format.
        Includes counters, gauges, and histograms with proper formatting.

        Returns:
            Prometheus text format string ready for /metrics endpoint

        Example output:
            # TYPE api_requests_total counter
            api_requests_total{method="GET"} 1234.0
            # TYPE api_active_connections gauge
            api_active_connections 42.0
        """
        lines = []

        # Counters
        for metric_key, value in self._counters.items():
            lines.append(f"# TYPE {self._get_metric_name(metric_key)} counter")
            lines.append(f"{self._sanitize_metric_name(metric_key)} {value}")

        # Gauges
        for metric_key, value in self._gauges.items():
            lines.append(f"# TYPE {self._get_metric_name(metric_key)} gauge")
            lines.append(f"{self._sanitize_metric_name(metric_key)} {value}")

        # Histograms
        for metric_key, values in self._histograms.items():
            if values:
                metric_name = self._get_metric_name(metric_key)
                lines.append(f"# TYPE {metric_name} histogram")

                # Calculate buckets
                sorted_values = sorted(values)
                count = len(sorted_values)
                sum_values = sum(sorted_values)

                # Standard buckets
                buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
                bucket_counts = []
                for bucket in buckets:
                    count_in_bucket = sum(1 for v in sorted_values if v <= bucket)
                    bucket_counts.append((bucket, count_in_bucket))

                # Add bucket metrics
                for bucket, count_in_bucket in bucket_counts:
                    lines.append(
                        f"{self._sanitize_metric_name(metric_key)}_bucket"
                        f'{{le="{bucket}"}} {count_in_bucket}'
                    )

                # Add sum and count
                lines.append(f"{self._sanitize_metric_name(metric_key)}_sum {sum_values}")
                lines.append(f"{self._sanitize_metric_name(metric_key)}_count {count}")

        return "\n".join(lines) + "\n"

    def _get_metric_name(self, metric_key: str) -> str:
        """Extract metric name from key"""
        if "{" in metric_key:
            return metric_key.split("{")[0]
        return metric_key

    def _sanitize_metric_name(self, name: str) -> str:
        """Sanitize metric name for Prometheus"""
        # Replace invalid characters
        sanitized = name.replace("-", "_").replace(".", "_")
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = f"metric_{sanitized}"
        return sanitized

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as dictionary"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: {
                        "count": len(v),
                        "sum": sum(v),
                        "min": min(v) if v else 0,
                        "max": max(v) if v else 0,
                        "avg": sum(v) / len(v) if v else 0,
                    }
                    for k, v in self._histograms.items()
                },
            }


# Global metrics registry
_metrics_registry: dict[str, PrometheusMetrics] = {}


def get_metrics(service_name: str) -> PrometheusMetrics:
    """Get or create metrics instance for a service"""
    if service_name not in _metrics_registry:
        _metrics_registry[service_name] = PrometheusMetrics(service_name)
    return _metrics_registry[service_name]


def generate_correlation_id() -> str:
    """Generate a correlation ID for request tracing"""
    return str(uuid.uuid4())


class CorrelationIDMiddleware:
    """Middleware to add correlation IDs to requests"""

    @staticmethod
    async def add_correlation_id(request, call_next):
        """FastAPI middleware to add correlation ID"""

        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = generate_correlation_id()

        # Add to request state
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response


# Helper functions for easy metric collection
def increment_counter(
    service_name: str, metric_name: str, value: float = 1.0, labels: dict[str, str] | None = None
):
    """Increment a counter metric"""
    metrics = get_metrics(service_name)
    metrics.counter(metric_name, value, labels)


def set_gauge(
    service_name: str, metric_name: str, value: float, labels: dict[str, str] | None = None
):
    """Set a gauge metric"""
    metrics = get_metrics(service_name)
    metrics.gauge(metric_name, value, labels)


def record_histogram(
    service_name: str, metric_name: str, value: float, labels: dict[str, str] | None = None
):
    """Record a histogram value"""
    metrics = get_metrics(service_name)
    metrics.histogram(metric_name, value, labels)


def record_latency(
    service_name: str,
    operation: str,
    duration_seconds: float,
    labels: dict[str, str] | None = None,
):
    """Record operation latency"""
    if labels is None:
        labels = {}
    labels["operation"] = operation
    record_histogram(service_name, f"{service_name}_latency_seconds", duration_seconds, labels)
