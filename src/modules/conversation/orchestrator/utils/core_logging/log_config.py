"""
Logging Configuration Module (v5.3)

Centralized configuration for the logging system.
Migrated from src/core/logging_config.py with improvements.

Changes from old version:
- Better type safety with dataclasses
- Enum for log levels
- Configuration validation
- Environment-based defaults
- Support for multiple log formats
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


class LogLevel(str, Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format types"""
    TEXT = "text"  # Human-readable text
    JSON = "json"  # JSON format for log aggregators
    STRUCTURED = "structured"  # Key-value pairs


@dataclass
class LogConfig:
    """
    Logging configuration dataclass

    Attributes:
        service_name: Name of the service
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logs_dir: Directory for log files
        console_output: Enable console logging
        file_output: Enable file logging
        json_output: Enable JSON log files
        error_file: Enable separate error log file
        startup_file: Enable shared startup log file
        rotation: File size for rotation (e.g., "500 MB")
        retention: How long to keep old logs (e.g., "10 days")
        compression: Compression format for old logs (zip, gz, bz2)
        colorize: Enable colored console output
        serialize: Use JSON serialization for file logs
        enqueue: Thread-safe logging (recommended)
        format: Log format type (text, json, structured)
    """
    service_name: str
    level: LogLevel = LogLevel.INFO
    logs_dir: Optional[Path] = None

    # Output targets
    console_output: bool = True
    file_output: bool = True
    json_output: bool = True
    error_file: bool = True
    startup_file: bool = True

    # Rotation and retention
    rotation: str = "500 MB"
    retention: str = "10 days"
    compression: str = "zip"

    # Formatting
    colorize: bool = True
    serialize: bool = True
    enqueue: bool = True
    format: LogFormat = LogFormat.TEXT

    # Custom log format strings
    console_format: str = field(default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>")
    file_format: str = field(default="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}")
    error_format: str = field(default="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}")

    # Additional metadata
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set defaults after initialization"""
        # Set logs directory if not provided
        if self.logs_dir is None:
            self.logs_dir = get_logs_dir()

        # Ensure logs_dir is a Path object
        if not isinstance(self.logs_dir, Path):
            self.logs_dir = Path(self.logs_dir)

        # Create logs directory
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Convert string level to LogLevel enum if needed
        if isinstance(self.level, str):
            self.level = LogLevel(self.level.upper())

    def get_service_log_file(self) -> Path:
        """Get path to service-specific log file"""
        return self.logs_dir / f"{self.service_name}.log"

    def get_error_log_file(self) -> Path:
        """Get path to error log file"""
        return self.logs_dir / f"{self.service_name}_errors.log"

    def get_json_log_file(self) -> Path:
        """Get path to JSON log file"""
        return self.logs_dir / f"{self.service_name}_json.log"

    def get_startup_log_file(self) -> Path:
        """Get path to shared startup log file"""
        return self.logs_dir / "startup.log"

    def get_audit_log_file(self) -> Path:
        """Get path to audit log file"""
        return self.logs_dir / f"{self.service_name}_audit.log"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'service_name': self.service_name,
            'level': self.level.value,
            'logs_dir': str(self.logs_dir),
            'console_output': self.console_output,
            'file_output': self.file_output,
            'json_output': self.json_output,
            'error_file': self.error_file,
            'startup_file': self.startup_file,
            'rotation': self.rotation,
            'retention': self.retention,
            'compression': self.compression,
            'colorize': self.colorize,
            'serialize': self.serialize,
            'enqueue': self.enqueue,
            'format': self.format.value,
            'extra_fields': self.extra_fields
        }


def get_logs_dir() -> Path:
    """
    Get default logs directory

    Priority:
    1. LOGS_DIR environment variable
    2. ~/.cache/ultravox-pipeline/logs (default)

    Returns:
        Path to logs directory
    """
    logs_dir_str = os.getenv("LOGS_DIR")
    if logs_dir_str:
        logs_dir = Path(logs_dir_str)
    else:
        # Fallback to ~/.cache/ultravox-pipeline/logs (portable, persistent)
        logs_dir = Path.home() / ".cache" / "ultravox-pipeline" / "logs"

    # Create directory if it doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_default_config(service_name: str, level: str = "INFO") -> LogConfig:
    """
    Get default logging configuration for a service

    Args:
        service_name: Name of the service
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        LogConfig instance with defaults
    """
    return LogConfig(
        service_name=service_name,
        level=LogLevel(level.upper())
    )


def get_startup_log_file() -> Path:
    """
    Get path to shared startup log file

    Returns:
        Path to startup.log
    """
    return get_logs_dir() / "startup.log"


def is_startup_message(message: str) -> bool:
    """
    Check if a message is a startup-related message

    Args:
        message: Log message to check

    Returns:
        True if message is startup-related
    """
    message_lower = message.lower()
    startup_keywords = [
        "starting", "started", "initializing", "initialized",
        "loading", "loaded", "hot reload", "watching", "startup",
        "lifespan", "listening", "ready", "finished", "registry",
        "🔥", "✅", "🏁", "📁", "⚠️", "🚀", "🔧"
    ]
    return any(keyword in message_lower for keyword in startup_keywords)
