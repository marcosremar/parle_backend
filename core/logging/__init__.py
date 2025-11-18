"""
Unified Logging Module (v5.3)

Consolidates all logging functionality into a single, consistent interface:
- Loguru-based logging (modern, feature-rich)
- OpenTelemetry integration (trace correlation)
- Audit trail (security and compliance)
- Structured logging (JSON, key-value pairs)
- Consistent formatting across all services

Migration from old system:
- OLD: logging_config.py + structured_logger.py (inconsistent)
- NEW: unified_logger.py (single source of truth)

Usage:
    from src.core.logging import get_logger, setup_logging

    # Setup logging for a service
    logger = setup_logging("my_service", level="INFO")

    # Or get a scoped logger
    logger = get_logger("my_service")

    # Use with trace context
    logger.info("Processing request", user_id="123", trace_id=ctx.trace_id)
"""

from .unified_logger import (
    get_logger,
    setup_logging,
    get_scoped_logger,
    configure_logging,
    shutdown_logging
)

from .audit_logger import (
    AuditLogger,
    get_audit_logger,
    log_security_event,
    log_access_event,
    log_data_change
)

from .log_formatter import (
    LogFormatter,
    JSONFormatter,
    TraceFormatter,
    get_formatter
)

from .log_config import (
    LogConfig,
    LogLevel,
    get_default_config,
    get_logs_dir
)

__all__ = [
    # Core logging
    'get_logger',
    'setup_logging',
    'get_scoped_logger',
    'configure_logging',
    'shutdown_logging',

    # Audit logging
    'AuditLogger',
    'get_audit_logger',
    'log_security_event',
    'log_access_event',
    'log_data_change',

    # Formatters
    'LogFormatter',
    'JSONFormatter',
    'TraceFormatter',
    'get_formatter',

    # Configuration
    'LogConfig',
    'LogLevel',
    'get_default_config',
    'get_logs_dir',
]
