"""
OpenTelemetry Observability Module.

Provides unified observability with:
- Distributed tracing (Jaeger)
- Metrics collection (Prometheus)
- Structured logging with trace correlation

All three pillars of observability in one place.
"""

from .context import (
    extract_trace_context,
    get_current_span_id,
    get_current_trace_id,
    get_trace_context,
    inject_trace_context,
)
from .instrumentation import (
    instrument_asyncpg,
    instrument_fastapi,
    instrument_httpx,
    instrument_redis,
)
from .telemetry import UnifiedTelemetry, configure_telemetry, get_telemetry, shutdown_telemetry

__all__ = [
    # Core telemetry
    "UnifiedTelemetry",
    "get_telemetry",
    "configure_telemetry",
    "shutdown_telemetry",
    # Auto-instrumentation
    "instrument_fastapi",
    "instrument_httpx",
    "instrument_asyncpg",
    "instrument_redis",
    # Context propagation
    "get_current_trace_id",
    "get_current_span_id",
    "get_trace_context",
    "inject_trace_context",
    "extract_trace_context",
]
