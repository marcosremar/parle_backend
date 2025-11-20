"""
Unified Telemetry - OpenTelemetry integration.

Provides a single interface for:
- Distributed tracing
- Metrics collection
- Structured logging with trace correlation

Usage:
    from src.core.observability import get_telemetry

    telemetry = get_telemetry("my_service")

    # Tracing
    with telemetry.trace("operation_name"):
        ...

    # Metrics
    telemetry.counter("requests_total").add(1)

    # Logging with trace
    telemetry.log_info("Processing request")
"""

import logging
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager
import os

# OpenTelemetry imports (core SDK)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import Status, StatusCode, Span
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    metrics = None

# Optional exporters (Jaeger, Prometheus, OTLP)
try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    JAEGER_AVAILABLE = True
except ImportError:
    JAEGER_AVAILABLE = False

try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False


# ============================================================================
# Global Configuration
# ============================================================================

_TELEMETRY_INSTANCES: Dict[str, 'UnifiedTelemetry'] = {}
_TELEMETRY_CONFIGURED = False


def configure_telemetry(
    service_name: str = "ultravox-pipeline",
    service_version: str = "1.0.0",
    environment: str = "development",
    jaeger_endpoint: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    enable_prometheus: bool = True,
    enable_console: bool = False
):
    """
    Configure OpenTelemetry globally.

    Call this ONCE at application startup.

    Args:
        service_name: Service name for telemetry
        service_version: Service version
        environment: Environment (development, staging, production)
        jaeger_endpoint: Jaeger collector endpoint (e.g., "localhost:14268")
        otlp_endpoint: OTLP gRPC endpoint (e.g., "localhost:4317")
        enable_prometheus: Enable Prometheus metrics
        enable_console: Enable console exporter (debug)
    """
    global _TELEMETRY_CONFIGURED

    if not OPENTELEMETRY_AVAILABLE:
        logging.warning("OpenTelemetry not available - install with: pip install opentelemetry-sdk")
        return

    if _TELEMETRY_CONFIGURED:
        logging.warning("Telemetry already configured, skipping")
        return

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "environment": environment,
    })

    # ========================================
    # Configure Tracing
    # ========================================
    tracer_provider = TracerProvider(resource=resource)

    # Jaeger exporter
    if jaeger_endpoint and JAEGER_AVAILABLE:
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_endpoint.split(":")[0],
            agent_port=int(jaeger_endpoint.split(":")[1]) if ":" in jaeger_endpoint else 6831,
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    elif jaeger_endpoint and not JAEGER_AVAILABLE:
        logging.warning("Jaeger exporter requested but not available - install: pip install opentelemetry-exporter-jaeger")

    # OTLP exporter (OpenTelemetry Protocol)
    if otlp_endpoint and OTLP_AVAILABLE:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    elif otlp_endpoint and not OTLP_AVAILABLE:
        logging.warning("OTLP exporter requested but not available - install: pip install opentelemetry-exporter-otlp")

    # Console exporter (debug)
    if enable_console:
        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

    trace.set_tracer_provider(tracer_provider)

    # ========================================
    # Configure Metrics
    # ========================================
    readers = []

    # Prometheus reader
    if enable_prometheus and PROMETHEUS_AVAILABLE:
        prometheus_reader = PrometheusMetricReader()
        readers.append(prometheus_reader)
    elif enable_prometheus and not PROMETHEUS_AVAILABLE:
        logging.warning("Prometheus metrics requested but not available - install: pip install opentelemetry-exporter-prometheus")

    # OTLP metrics reader
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        otlp_metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
        otlp_metric_reader = PeriodicExportingMetricReader(otlp_metric_exporter)
        readers.append(otlp_metric_reader)

    meter_provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(meter_provider)

    _TELEMETRY_CONFIGURED = True
    logging.info(f"✅ Telemetry configured for {service_name}")


def shutdown_telemetry():
    """Shutdown telemetry (call on application exit)."""
    global _TELEMETRY_CONFIGURED

    if not OPENTELEMETRY_AVAILABLE or not _TELEMETRY_CONFIGURED:
        return

    # Shutdown tracer provider
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, 'shutdown'):
        tracer_provider.shutdown()

    # Shutdown meter provider
    meter_provider = metrics.get_meter_provider()
    if hasattr(meter_provider, 'shutdown'):
        meter_provider.shutdown()

    _TELEMETRY_CONFIGURED = False
    logging.info("✅ Telemetry shutdown complete")


# ============================================================================
# UnifiedTelemetry Class
# ============================================================================

class UnifiedTelemetry:
    """
    Unified telemetry interface for a service.

    Provides:
    - Distributed tracing with automatic context propagation
    - Metrics collection (counters, histograms, gauges)
    - Structured logging with trace correlation

    Example:
        telemetry = UnifiedTelemetry("my_service")

        # Tracing
        with telemetry.trace("process_request"):
            telemetry.log_info("Processing...")
            telemetry.counter("requests").add(1)
    """

    def __init__(self, service_name: str):
        """
        Initialize telemetry for a service.

        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)

        if OPENTELEMETRY_AVAILABLE:
            # Get tracer
            self.tracer = trace.get_tracer(service_name)

            # Get meter
            self.meter = metrics.get_meter(service_name)

            # Pre-create common metrics
            self._counters: Dict[str, Any] = {}
            self._histograms: Dict[str, Any] = {}
            self._gauges: Dict[str, Any] = {}
        else:
            self.tracer = None
            self.meter = None
            self._counters = {}
            self._histograms = {}
            self._gauges = {}

    # ========================================
    # Tracing
    # ========================================

    @contextmanager
    def trace(self, span_name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Create a trace span.

        Args:
            span_name: Name of the operation
            attributes: Additional span attributes

        Usage:
            with telemetry.trace("database_query", {"table": "users"}):
                result = await db.query("SELECT * FROM users")
        """
        if not OPENTELEMETRY_AVAILABLE or not self.tracer:
            # No-op if telemetry disabled
            yield None
            return

        with self.tracer.start_as_current_span(span_name) as span:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                # Record exception
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID as hex string."""
        if not OPENTELEMETRY_AVAILABLE:
            return None

        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, '032x')
        return None

    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID as hex string."""
        if not OPENTELEMETRY_AVAILABLE:
            return None

        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, '016x')
        return None

    # ========================================
    # Metrics
    # ========================================

    def counter(self, name: str, description: str = "", unit: str = "1"):
        """
        Get or create a counter metric.

        Counters only go up (e.g., requests_total, errors_total).

        Args:
            name: Metric name
            description: Metric description
            unit: Unit of measurement

        Returns:
            Counter object with .add(value, attributes={}) method
        """
        if not OPENTELEMETRY_AVAILABLE or not self.meter:
            # Return no-op counter
            class NoOpCounter:
                def add(self, value, attributes=None):
                    pass
            return NoOpCounter()

        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._counters[name]

    def histogram(self, name: str, description: str = "", unit: str = "1"):
        """
        Get or create a histogram metric.

        Histograms track distributions (e.g., request_duration_ms).

        Args:
            name: Metric name
            description: Metric description
            unit: Unit of measurement

        Returns:
            Histogram object with .record(value, attributes={}) method
        """
        if not OPENTELEMETRY_AVAILABLE or not self.meter:
            # Return no-op histogram
            class NoOpHistogram:
                def record(self, value, attributes=None):
                    pass
            return NoOpHistogram()

        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(
                name=name,
                description=description,
                unit=unit
            )
        return self._histograms[name]

    def gauge(self, name: str, description: str = "", unit: str = "1"):
        """
        Get or create a gauge metric (Observable Gauge).

        Gauges track current values (e.g., active_connections, memory_usage).

        Note: OpenTelemetry gauges require a callback function.

        Args:
            name: Metric name
            description: Metric description
            unit: Unit of measurement

        Returns:
            Gauge object
        """
        if not OPENTELEMETRY_AVAILABLE or not self.meter:
            return None

        if name not in self._gauges:
            # Observable gauges need callbacks
            # For now, return the meter to create manually
            self._gauges[name] = self.meter
        return self._gauges[name]

    # ========================================
    # Structured Logging with Trace Correlation
    # ========================================

    def _log_with_trace(self, level: int, message: str, **kwargs):
        """Log message with trace context."""
        extra = kwargs.get('extra', {})

        # Add trace context
        trace_id = self.get_current_trace_id()
        span_id = self.get_current_span_id()

        if trace_id:
            extra['trace_id'] = trace_id
        if span_id:
            extra['span_id'] = span_id

        extra['service'] = self.service_name

        kwargs['extra'] = extra
        self.logger.log(level, message, **kwargs)

    def log_debug(self, message: str, **kwargs):
        """Log debug message with trace context."""
        self._log_with_trace(logging.DEBUG, message, **kwargs)

    def log_info(self, message: str, **kwargs):
        """Log info message with trace context."""
        self._log_with_trace(logging.INFO, message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log warning message with trace context."""
        self._log_with_trace(logging.WARNING, message, **kwargs)

    def log_error(self, message: str, **kwargs):
        """Log error message with trace context."""
        self._log_with_trace(logging.ERROR, message, **kwargs)

    def log_critical(self, message: str, **kwargs):
        """Log critical message with trace context."""
        self._log_with_trace(logging.CRITICAL, message, **kwargs)


# ============================================================================
# Singleton Factory
# ============================================================================

def get_telemetry(service_name: str) -> UnifiedTelemetry:
    """
    Get or create telemetry instance for a service.

    Args:
        service_name: Service name

    Returns:
        UnifiedTelemetry instance (singleton per service)
    """
    if service_name not in _TELEMETRY_INSTANCES:
        _TELEMETRY_INSTANCES[service_name] = UnifiedTelemetry(service_name)

    return _TELEMETRY_INSTANCES[service_name]
