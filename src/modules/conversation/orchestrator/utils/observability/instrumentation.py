"""
Auto-instrumentation for common libraries.

Automatically instruments:
- FastAPI (HTTP requests)
- httpx (HTTP clients)
- asyncpg (PostgreSQL)
- redis (Redis)

Usage:
    from src.core.observability import instrument_fastapi

    app = FastAPI()
    instrument_fastapi(app)
"""

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Instrumentation
# ============================================================================

def instrument_fastapi(app):
    """
    Instrument FastAPI application.

    Automatically creates spans for all HTTP requests.

    Args:
        app: FastAPI application instance

    Returns:
        Instrumented app
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("✅ FastAPI instrumented for tracing")
        return app
    except ImportError:
        logger.warning("⚠️  opentelemetry-instrumentation-fastapi not installed")
        return app


# ============================================================================
# HTTP Client Instrumentation
# ============================================================================

def instrument_httpx():
    """
    Instrument httpx HTTP client.

    Automatically creates spans for all HTTP requests made with httpx.
    """
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.info("✅ httpx instrumented for tracing")
    except ImportError:
        logger.warning("⚠️  opentelemetry-instrumentation-httpx not installed")


def instrument_requests():
    """
    Instrument requests HTTP client.

    Automatically creates spans for all HTTP requests made with requests.
    """
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
        logger.info("✅ requests instrumented for tracing")
    except ImportError:
        logger.warning("⚠️  opentelemetry-instrumentation-requests not installed")


# ============================================================================
# Database Instrumentation
# ============================================================================

def instrument_asyncpg():
    """
    Instrument asyncpg PostgreSQL client.

    Automatically creates spans for all database queries.
    """
    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
        AsyncPGInstrumentor().instrument()
        logger.info("✅ asyncpg instrumented for tracing")
    except ImportError:
        logger.warning("⚠️  opentelemetry-instrumentation-asyncpg not installed")


def instrument_sqlalchemy():
    """
    Instrument SQLAlchemy ORM.

    Automatically creates spans for all ORM operations.
    """
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        SQLAlchemyInstrumentor().instrument()
        logger.info("✅ SQLAlchemy instrumented for tracing")
    except ImportError:
        logger.warning("⚠️  opentelemetry-instrumentation-sqlalchemy not installed")


# ============================================================================
# Cache Instrumentation
# ============================================================================

def instrument_redis():
    """
    Instrument redis client.

    Automatically creates spans for all Redis operations.
    """
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor().instrument()
        logger.info("✅ redis instrumented for tracing")
    except ImportError:
        logger.warning("⚠️  opentelemetry-instrumentation-redis not installed")


# ============================================================================
# Async Frameworks
# ============================================================================

def instrument_aiohttp():
    """
    Instrument aiohttp (async HTTP server/client).

    Automatically creates spans for aiohttp requests.
    """
    try:
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
        AioHttpClientInstrumentor().instrument()
        logger.info("✅ aiohttp instrumented for tracing")
    except ImportError:
        logger.warning("⚠️  opentelemetry-instrumentation-aiohttp-client not installed")


# ============================================================================
# Bulk Instrumentation
# ============================================================================

def instrument_all():
    """
    Instrument all available libraries.

    Call this at application startup to auto-instrument everything.
    """
    logger.info("🔧 Auto-instrumenting libraries...")

    # HTTP
    instrument_httpx()
    instrument_requests()

    # Databases
    instrument_asyncpg()
    instrument_sqlalchemy()

    # Cache
    instrument_redis()

    # Async
    instrument_aiohttp()

    logger.info("✅ Auto-instrumentation complete")
