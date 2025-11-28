"""
Prometheus Metrics and Observability
Provides metrics collection and Prometheus format export
"""
import time
import uuid
from typing import Dict, Any, Optional, List
from collections import defaultdict
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Prometheus-compatible metrics collector
    
    Supports:
    - Counters (monotonically increasing)
    - Gauges (can go up or down)
    - Histograms (distribution of values)
    - Correlation IDs for request tracing
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._labels: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._lock = Lock()
        
        logger.info(f"✅ Prometheus metrics initialized for {service_name}")
    
    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._counters[metric_key] += value
            if labels:
                self._labels[metric_key] = labels
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._gauges[metric_key] = value
            if labels:
                self._labels[metric_key] = labels
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._histograms[metric_key].append(value)
            if labels:
                self._labels[metric_key] = labels
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Get metric key with labels"""
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus format
        
        Returns:
            Prometheus text format string
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
    
    def get_metrics_dict(self) -> Dict[str, Any]:
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
                        "avg": sum(v) / len(v) if v else 0
                    }
                    for k, v in self._histograms.items()
                }
            }


# Global metrics registry
_metrics_registry: Dict[str, PrometheusMetrics] = {}


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
        from fastapi import Request
        
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
def increment_counter(service_name: str, metric_name: str, value: float = 1.0, 
                     labels: Optional[Dict[str, str]] = None):
    """Increment a counter metric"""
    metrics = get_metrics(service_name)
    metrics.counter(metric_name, value, labels)


def set_gauge(service_name: str, metric_name: str, value: float, 
              labels: Optional[Dict[str, str]] = None):
    """Set a gauge metric"""
    metrics = get_metrics(service_name)
    metrics.gauge(metric_name, value, labels)


def record_histogram(service_name: str, metric_name: str, value: float,
                    labels: Optional[Dict[str, str]] = None):
    """Record a histogram value"""
    metrics = get_metrics(service_name)
    metrics.histogram(metric_name, value, labels)


def record_latency(service_name: str, operation: str, duration_seconds: float,
                  labels: Optional[Dict[str, str]] = None):
    """Record operation latency"""
    if labels is None:
        labels = {}
    labels["operation"] = operation
    record_histogram(service_name, f"{service_name}_latency_seconds", duration_seconds, labels)

