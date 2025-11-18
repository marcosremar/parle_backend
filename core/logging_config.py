"""
⚠️ DEPRECATED (v5.3): This module has been consolidated into src.core.logging

Use instead:
    from src.core.logging import setup_logging  # NEW

Old (DEPRECATED):
    from src.core.logging_config import setup_logging

This file is kept for backward compatibility only and will be removed in v6.0.

Centralized Logging Configuration using Loguru
Provides consistent logging across all services
"""
import warnings
from loguru import logger
import sys
import os
from pathlib import Path
from typing import Optional, Any

# Emit deprecation warning
warnings.warn(
    "logging_config.py is deprecated. Use 'from src.core.logging import setup_logging' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Default logs directory from .env
# Read from LOGS_DIR environment variable (must be set in .env)
# Use lazy initialization to handle cases where .env hasn't been loaded yet
_DEFAULT_LOGS_DIR = None

def _get_default_logs_dir() -> Path:
    """Get default logs directory, loading .env if needed"""
    global _DEFAULT_LOGS_DIR
    if _DEFAULT_LOGS_DIR is None:
        logs_dir_str = os.getenv("LOGS_DIR")
        if logs_dir_str:
            _DEFAULT_LOGS_DIR = Path(logs_dir_str)
        else:
            # Fallback to ~/.cache/ultravox-pipeline/logs (portable, persistent)
            _DEFAULT_LOGS_DIR = Path.home() / ".cache" / "ultravox-pipeline" / "logs"
        _DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_LOGS_DIR

# Startup log file - captures all startup messages across all services (lazy)
def get_startup_log_file() -> Path:
    """Get startup log file path (lazy)"""
    return _get_default_logs_dir() / "startup.log"


def setup_logging(
    service_name: str,
    level: str = "INFO",
    logs_dir: Optional[Path] = None
) -> logger:
    """
    Setup centralized logging for a service using Loguru

    Args:
        service_name: Name of the service (e.g., "user", "orchestrator")
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        logs_dir: Custom logs directory (default: ./logs or service/tmp/logs)

    Returns:
        Configured logger instance

    Features:
        - Colorized console output (development)
        - Rotating file logs (production)
        - Separate error logs
        - JSON logs for aggregators
        - Automatic compression and retention

    Example:
        from src.core.logging_config import setup_logging
        from loguru import logger

        # Use service-specific tmp directory
        setup_logging("user", level="DEBUG", logs_dir=Path("Path.home() / ".cache/ultravox-pipeline/logs""))

        logger.info("Service started", user_id="123")
        logger.error("Database failed", error="timeout")
    """

    # Use provided logs_dir or default
    LOGS_DIR = logs_dir if logs_dir else _get_default_logs_dir()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # 1. Console output (colorized, human-readable)
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True
    )

    # 2. Service-specific log file (all logs)
    logger.add(
        LOGS_DIR / f"{service_name}.log",
        level="DEBUG",
        rotation="500 MB",  # Rotate when file reaches 500MB
        retention="10 days",  # Keep logs for 10 days
        compression="zip",  # Compress old logs
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        enqueue=True  # Thread-safe
    )

    # 3. Error log file (errors only)
    logger.add(
        LOGS_DIR / f"{service_name}_errors.log",
        level="ERROR",
        rotation="100 MB",
        retention="30 days",  # Keep errors longer
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        enqueue=True
    )

    # 4. JSON log file (for log aggregators like ELK, Loki)
    logger.add(
        LOGS_DIR / f"{service_name}_json.log",
        level="INFO",
        rotation="500 MB",
        retention="7 days",
        compression="zip",
        serialize=True,  # JSON format
        enqueue=True
    )

    # 5. Startup log file (captures all startup messages from all services)
    # This file is shared across all services for easy debugging
    def startup_filter(record) -> any:
        """Filter to capture startup-related messages"""
        message = record["message"].lower()
        startup_keywords = [
            "starting", "started", "initializing", "initialized", "loading", "loaded",
            "hot reload", "watching", "startup", "lifespan", "listening", "ready",
            "🔥", "✅", "🏁", "📁", "⚠️", "finished", "🚀", "🔧", "registry"
        ]
        return any(keyword in message for keyword in startup_keywords)

    logger.add(
        get_startup_log_file(),
        level="DEBUG",  # Capture DEBUG level too for startup messages
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | [" + service_name + "] {message}",
        filter=startup_filter,
        enqueue=True
    )

    # Log that logging is configured
    logger.bind(service=service_name).info(f"Logging configured for {service_name}")

    return logger


def log_exception(exception: Exception, context: Optional[dict] = None) -> Any:
    """
    Helper function to log exceptions with context

    Args:
        exception: The exception to log
        context: Additional context (user_id, request_id, etc.)

    Example:
        try:
            risky_operation()
        except Exception as e:
            log_exception(e, {"user_id": "123", "operation": "login"})
    """
    if context:
        logger.bind(**context).exception(exception)
    else:
        logger.exception(exception)
