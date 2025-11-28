"""
Trace context propagation utilities.

Helpers for:
- Getting current trace/span IDs
- Injecting trace context into HTTP headers
- Extracting trace context from HTTP headers
- Manual context propagation between services

Usage:
    # Get current trace ID
    trace_id = get_current_trace_id()

    # Inject into HTTP headers
    headers = inject_trace_context({})

    # Extract from incoming headers
    context = extract_trace_context(request.headers)
"""

from typing import Dict, Optional, Any

try:
    from opentelemetry import trace, context
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    context = None


# ============================================================================
# Current Context
# ============================================================================

def get_current_trace_id() -> Optional[str]:
    """
    Get current trace ID as hex string.

    Returns:
        Trace ID (32 hex chars) or None if no active span

    Example:
        trace_id = get_current_trace_id()
        # "5f8a3b2c1d4e7890abcdef1234567890"
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, '032x')
    return None


def get_current_span_id() -> Optional[str]:
    """
    Get current span ID as hex string.

    Returns:
        Span ID (16 hex chars) or None if no active span

    Example:
        span_id = get_current_span_id()
        # "abcdef1234567890"
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, '016x')
    return None


def get_trace_context() -> Dict[str, str]:
    """
    Get current trace context as dict.

    Returns:
        Dict with trace_id and span_id keys

    Example:
        context = get_trace_context()
        # {"trace_id": "5f8a...", "span_id": "abcd..."}
    """
    return {
        "trace_id": get_current_trace_id() or "",
        "span_id": get_current_span_id() or ""
    }


# ============================================================================
# Context Propagation (HTTP Headers)
# ============================================================================

def inject_trace_context(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context into HTTP headers.

    Use this when making HTTP requests to propagate trace context.

    Args:
        headers: Existing headers dict (will be modified)

    Returns:
        Headers dict with trace context injected

    Example:
        headers = {"Content-Type": "application/json"}
        headers = inject_trace_context(headers)
        # headers now contains: traceparent, tracestate

        async with httpx.AsyncClient() as client:
            await client.post(url, headers=headers)
    """
    if not OPENTELEMETRY_AVAILABLE:
        return headers

    # Use W3C Trace Context propagator
    propagator = TraceContextTextMapPropagator()
    carrier = headers.copy()

    # Inject context into carrier
    propagator.inject(carrier)

    return carrier


def extract_trace_context(headers: Dict[str, str]) -> Optional[Any]:
    """
    Extract trace context from HTTP headers.

    Use this when receiving HTTP requests to continue trace.

    Args:
        headers: HTTP headers dict

    Returns:
        Context object (opaque) or None

    Example:
        # In FastAPI route
        @app.post("/process")
        async def process(request: Request):
            # Extract context from incoming request
            ctx = extract_trace_context(dict(request.headers))

            # Attach context to current span
            if ctx:
                from opentelemetry import context
                token = context.attach(ctx)

            # ... process request ...

            # Detach context
            if ctx:
                context.detach(token)
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    # Use W3C Trace Context propagator
    propagator = TraceContextTextMapPropagator()

    # Extract context from carrier
    ctx = propagator.extract(headers)

    return ctx


# ============================================================================
# Manual Context Management
# ============================================================================

def attach_trace_context(ctx: Any) -> Any:
    """
    Attach trace context to current execution.

    Args:
        ctx: Context object from extract_trace_context()

    Returns:
        Token (use with detach_trace_context())

    Example:
        ctx = extract_trace_context(headers)
        token = attach_trace_context(ctx)
        try:
            # ... work with attached context ...
        finally:
            detach_trace_context(token)
    """
    if not OPENTELEMETRY_AVAILABLE or not ctx:
        return None

    return context.attach(ctx)


def detach_trace_context(token: Any):
    """
    Detach trace context.

    Args:
        token: Token from attach_trace_context()

    Example:
        token = attach_trace_context(ctx)
        try:
            # ...
        finally:
            detach_trace_context(token)
    """
    if not OPENTELEMETRY_AVAILABLE or not token:
        return

    context.detach(token)


# ============================================================================
# Utility Functions
# ============================================================================

def create_span_link(trace_id: str, span_id: str) -> Optional[Any]:
    """
    Create a span link to another trace.

    Useful for linking related traces across services.

    Args:
        trace_id: Trace ID to link to (32 hex chars)
        span_id: Span ID to link to (16 hex chars)

    Returns:
        Link object or None

    Example:
        # Link current span to another trace
        link = create_span_link(
            trace_id="5f8a3b2c1d4e7890abcdef1234567890",
            span_id="abcdef1234567890"
        )

        with tracer.start_as_current_span("operation", links=[link]):
            ...
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    try:
        from opentelemetry.trace import Link, SpanContext, TraceFlags

        # Parse IDs
        trace_id_int = int(trace_id, 16)
        span_id_int = int(span_id, 16)

        # Create span context
        span_context = SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=True,
            trace_flags=TraceFlags(0x01)  # Sampled
        )

        # Create link
        return Link(span_context)

    except Exception:
        return None


def format_trace_for_logging(include_span: bool = True) -> str:
    """
    Format trace context for logging.

    Args:
        include_span: Include span ID (default: True)

    Returns:
        Formatted string for logs

    Example:
        logger.info(f"Processing request {format_trace_for_logging()}")
        # "Processing request [trace:5f8a...] [span:abcd...]"
    """
    trace_id = get_current_trace_id()
    span_id = get_current_span_id()

    parts = []

    if trace_id:
        parts.append(f"[trace:{trace_id[:8]}...]")

    if include_span and span_id:
        parts.append(f"[span:{span_id[:8]}...]")

    return " ".join(parts) if parts else "[no-trace]"
