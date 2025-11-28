"""
Unified Logger Module (v5.3)

Consolidates logging_config.py and structured_logger.py into a single,
consistent logging interface.

Features:
- Loguru-based (modern, feature-rich)
- OpenTelemetry integration (trace correlation)
- Structured logging (JSON, key-value pairs)
- Multiple output targets (console, file, JSON, errors)
- Automatic rotation and compression
- Thread-safe logging
- Scoped loggers (per-service)

Migration Guide:
    OLD (logging_config.py):
        from src.core.core_logging_config import setup_logging
        logger = setup_logging("my_service", level="INFO")

    NEW (unified_logger.py):
        from src.core.core_logging import setup_logging
        logger = setup_logging("my_service", level="INFO")

    OLD (structured_logger.py):
        from src.core.structured_logger import get_logger
        logger = get_logger("my_service")

    NEW (unified_logger.py):
        from src.core.core_logging import get_logger
        logger = get_logger("my_service")
"""

import sys
from typing import Optional, Any, Dict, Callable
from pathlib import Path
from loguru import logger as base_logger

from .log_config import LogConfig, LogLevel, get_default_config, is_startup_message
from .log_formatter import TraceFormatter

# Global state
_configured_services = set()
_default_config: Optional[LogConfig] = None


def configure_logging(config: Optional[LogConfig] = None) -> None:
    """
    Configure global logging settings

    This should be called once at application startup.

    Args:
        config: LogConfig instance (None for defaults)
    """
    global _default_config

    if config is None:
        config = get_default_config("ultravox-pipeline")

    _default_config = config

    # Remove default handler
    base_logger.remove()

    # Configure base logger with default settings
    if config.console_output:
        base_logger.add(
            sys.stdout,
            level=config.level.value,
            format=config.console_format,
            colorize=config.colorize
        )


def setup_logging(
    service_name: str,
    level: str = "INFO",
    logs_dir: Optional[Path] = None,
    config: Optional[LogConfig] = None
) -> Any:
    """
    Setup logging for a service using Loguru

    This is the main function for setting up service logging.
    It configures multiple output targets:
    - Console (colorized, human-readable)
    - Service log file (all logs)
    - Error log file (errors only)
    - JSON log file (for aggregators)
    - Startup log file (shared across services)

    Args:
        service_name: Name of the service (e.g., "user", "orchestrator")
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logs_dir: Custom logs directory (default: ~/.cache/ultravox-pipeline/logs)
        config: Custom LogConfig (overrides other parameters)

    Returns:
        Configured Loguru logger instance (scoped to service)

    Example:
        from src.core.core_logging import setup_logging

        logger = setup_logging("user", level="DEBUG")
        logger.info("Service started", user_id="123")
        logger.error("Database failed", error="timeout")
    """
    global _configured_services

    # Use provided config or create default
    if config is None:
        config = LogConfig(
            service_name=service_name,
            level=LogLevel(level.upper()),
            logs_dir=logs_dir
        )

    # Create scoped logger for this service
    scoped_logger = base_logger.bind(service=service_name)

    # Only configure handlers if not already configured
    if service_name not in _configured_services:
        # Remove default handler if this is the first service
        if not _configured_services:
            base_logger.remove()

        # 1. Console output (colorized, human-readable)
        if config.console_output:
            base_logger.add(
                sys.stdout,
                level=config.level.value,
                format=config.console_format,
                colorize=config.colorize,
                filter=lambda record: record["extra"].get("service") == service_name
            )

        # 2. Service-specific log file (all logs)
        if config.file_output:
            base_logger.add(
                config.get_service_log_file(),
                level="DEBUG",
                rotation=config.rotation,
                retention=config.retention,
                compression=config.compression,
                format=config.file_format,
                enqueue=config.enqueue,
                filter=lambda record: record["extra"].get("service") == service_name
            )

        # 3. Error log file (errors only)
        if config.error_file:
            base_logger.add(
                config.get_error_log_file(),
                level="ERROR",
                rotation="100 MB",
                retention="30 days",
                compression=config.compression,
                format=config.error_format,
                enqueue=config.enqueue,
                filter=lambda record: record["extra"].get("service") == service_name
            )

        # 4. JSON log file (for log aggregators like ELK, Loki)
        if config.json_output:
            base_logger.add(
                config.get_json_log_file(),
                level="INFO",
                rotation=config.rotation,
                retention="7 days",
                compression=config.compression,
                serialize=True,  # JSON format
                enqueue=config.enqueue,
                filter=lambda record: record["extra"].get("service") == service_name
            )

        # 5. Startup log file (captures startup messages from all services)
        if config.startup_file:
            def startup_filter(record) -> bool:
                """Filter to capture startup-related messages for this service"""
                if record["extra"].get("service") != service_name:
                    return False
                return is_startup_message(record["message"])

            base_logger.add(
                config.get_startup_log_file(),
                level="DEBUG",
                rotation="50 MB",
                retention="30 days",
                compression=config.compression,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | [" + service_name + "] {message}",
                filter=startup_filter,
                enqueue=config.enqueue
            )

        _configured_services.add(service_name)

        # Log that logging is configured
        scoped_logger.info(f"✅ Logging configured for {service_name}")

    return scoped_logger


def get_logger(service_name: str) -> Any:
    """
    Get a scoped logger for a service

    This creates a logger bound to the service name, which will be included
    in all log records for filtering and routing.

    Args:
        service_name: Name of the service

    Returns:
        Scoped Loguru logger instance

    Example:
        from src.core.core_logging import get_logger

        logger = get_logger("my_service")
        logger.info("Processing request", request_id="abc123")
    """
    return base_logger.bind(service=service_name)


def get_scoped_logger(
    service_name: str,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **extra_context
) -> Any:
    """
    Get a logger with OpenTelemetry trace context

    This creates a logger with trace context for distributed tracing.
    All logs will include trace_id and span_id for correlation.

    Args:
        service_name: Name of the service
        trace_id: OpenTelemetry trace ID
        span_id: OpenTelemetry span ID
        **extra_context: Additional context to bind to logger

    Returns:
        Scoped Loguru logger with trace context

    Example:
        from src.core.core_logging import get_scoped_logger

        logger = get_scoped_logger(
            "my_service",
            trace_id="abc123",
            span_id="def456",
            user_id="user123"
        )
        logger.info("Request processed")
        # Output includes: [trace:abc123] [span:def456] user_id=user123
    """
    context = {
        "service": service_name,
        **extra_context
    }

    if trace_id:
        context["trace_id"] = trace_id

    if span_id:
        context["span_id"] = span_id

    return base_logger.bind(**context)


def shutdown_logging() -> None:
    """
    Shutdown logging and flush all buffers

    Call this before application exit to ensure all logs are written.
    """
    global _configured_services

    base_logger.info("🛑 Shutting down logging system")

    # Remove all handlers
    base_logger.remove()

    # Clear configured services
    _configured_services.clear()

    base_logger.info("✅ Logging shutdown complete")


def log_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    service_name: Optional[str] = None
) -> None:
    """
    Helper function to log exceptions with context

    Args:
        exception: The exception to log
        context: Additional context (user_id, request_id, etc.)
        service_name: Service name for scoped logging

    Example:
        from src.core.core_logging import log_exception

        try:
            risky_operation()
        except Exception as e:
            log_exception(e, {
                "user_id": "123",
                "operation": "login"
            }, service_name="user")
    """
    logger_instance = base_logger

    if service_name:
        logger_instance = get_logger(service_name)

    if context:
        logger_instance = logger_instance.bind(**context)

    logger_instance.exception(exception)


# Backward compatibility aliases
def get_structured_logger(service_name: str) -> Any:
    """
    DEPRECATED: Use get_logger() instead

    This is a compatibility alias for the old structured_logger.py
    """
    return get_logger(service_name)
