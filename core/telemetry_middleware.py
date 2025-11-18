"""
Telemetry Middleware - Automatic Request Tracking for All Services

This middleware automatically logs request timing for ALL services without
requiring modifications to individual service files.

Usage:
    Add to any FastAPI service:

    from src.core.telemetry_middleware import add_telemetry_middleware

    app = FastAPI()
    add_telemetry_middleware(app, service_name="my_service")
"""

import time
import logging
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from src.core.telemetry_store import get_telemetry_store

# OpenTelemetry - Trace context extraction
try:
    from src.core.observability import extract_trace_context, attach_trace_context, detach_trace_context
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Fallback no-op functions
    def extract_trace_context(headers):
        return None
    def attach_trace_context(ctx):
        return None
    def detach_trace_context(token):
        pass

logger = logging.getLogger(__name__)


class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    Automatic telemetry middleware for all HTTP requests

    Logs:
    - Request reception
    - Request method and path
    - Request payload size
    - Processing time
    - Response size
    - Status code
    """

    def __init__(self, app: ASGIApp, service_name: str):
        super().__init__(app)
        self.service_name = service_name.upper()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with telemetry"""

        # Skip telemetry for health checks and telemetry endpoints themselves (avoid recursion)
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"] or request.url.path.startswith("/telemetry"):
            return await call_next(request)

        # ==========================================
        # TRACE CONTEXT EXTRACTION (OpenTelemetry)
        # ==========================================
        # Extract trace context from incoming request headers
        trace_ctx = extract_trace_context(dict(request.headers))
        trace_token = None

        if trace_ctx:
            # Attach context to current span (enables distributed tracing)
            trace_token = attach_trace_context(trace_ctx)
            logger.debug(f"🔗 [{self.service_name}] Trace context extracted and attached")

        try:
            # ==========================================
            # TELEMETRY START
            # ==========================================
            start_time = time.time()
            # Use UUID suffix to ensure uniqueness even with concurrent requests
            unique_suffix = str(uuid.uuid4())[:8]  # First 8 chars of UUID
            request_id = f"{self.service_name.lower()}_{int(start_time * 1000)}_{unique_suffix}"

            # Get request details
            method = request.method
            path = request.url.path

            # Get request size (if available)
            content_length = request.headers.get("content-length", "0")
            request_size = int(content_length) if content_length.isdigit() else 0

            # Log request reception
            logger.info(
                f"🔷 [{self.service_name}] Received {method} {path} "
                f"(request_id: {request_id}, size: {request_size} bytes)"
            )

            # Process request
            response = await call_next(request)

            # ==========================================
            # TELEMETRY END (SUCCESS)
            # ==========================================
            processing_time = (time.time() - start_time) * 1000

            # Get response size
            response_size = 0
            if hasattr(response, "body"):
                response_size = len(response.body) if response.body else 0

            # Log completion
            status = response.status_code
            if status < 400:
                logger.info(
                    f"✅ [{self.service_name}] {method} {path} completed: {processing_time:.0f}ms "
                    f"(status: {status}, response: {response_size} bytes)"
                )
            else:
                logger.warning(
                    f"⚠️  [{self.service_name}] {method} {path} returned error: {processing_time:.0f}ms "
                    f"(status: {status})"
                )

            # Save to telemetry store
            store = get_telemetry_store()
            store.add_record(
                request_id=request_id,
                service_name=self.service_name,
                method=method,
                path=path,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                processing_time_ms=processing_time,
                status_code=status,
                error=None,
            )

            # Add telemetry headers to response
            response.headers["X-Service-Name"] = self.service_name
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time-Ms"] = str(int(processing_time))

            return response

        except Exception as e:
            # ==========================================
            # TELEMETRY END (ERROR)
            # ==========================================
            processing_time = (time.time() - start_time) * 1000

            error_msg = str(e)
            logger.error(
                f"❌ [{self.service_name}] {method} {path} failed: {processing_time:.0f}ms "
                f"(error: {error_msg})"
            )

            # Save error to telemetry store
            store = get_telemetry_store()
            store.add_record(
                request_id=request_id,
                service_name=self.service_name,
                method=method,
                path=path,
                request_size_bytes=request_size,
                response_size_bytes=0,
                processing_time_ms=processing_time,
                status_code=500,
                error=error_msg,
            )

            raise

        finally:
            # ==========================================
            # TRACE CONTEXT CLEANUP (OpenTelemetry)
            # ==========================================
            # Detach trace context to prevent context leaks
            if trace_token:
                detach_trace_context(trace_token)


def add_telemetry_middleware(app: FastAPI, service_name: str) -> None:
    """
    Add telemetry middleware to a FastAPI application

    Args:
        app: FastAPI application instance
        service_name: Name of the service (e.g., "user", "session", "llm")

    Example:
        app = FastAPI()
        add_telemetry_middleware(app, "user")
    """
    app.add_middleware(TelemetryMiddleware, service_name=service_name)
    logger.info(f"✅ Telemetry middleware enabled for service: {service_name}")


# Emoji mapping for different services
SERVICE_EMOJIS = {
    "api_gateway": "🌐",
    "websocket": "📡",
    "webrtc": "📹",
    "webrtc_signaling": "🔗",
    "rest_polling": "🔄",
    "orchestrator": "🎯",
    "user": "👤",
    "session": "🔐",
    "database": "💾",
    "scenarios": "📋",
    "conversation_store": "💬",
    "external_llm": "🤖",
    "external_stt": "📝",
    "external_tts": "🔊",
    "llm": "🧠",
    "tts": "🎤",
    "stt": "🎧",
    "file_storage": "📁",
    "metrics_testing": "📊",
    "skypilot": "☁️",
    "runpod_llm": "🚀",
}


def get_service_emoji(service_name: str) -> str:
    """Get emoji for service name"""
    return SERVICE_EMOJIS.get(service_name.lower(), "🔷")
