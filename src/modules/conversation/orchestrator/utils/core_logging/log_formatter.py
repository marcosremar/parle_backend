"""
Log Formatters Module (v5.3)

Provides consistent log formatting across all services with support for:
- Text formatting (human-readable)
- JSON formatting (log aggregators)
- Trace context formatting (OpenTelemetry integration)
- Custom field injection

This module ensures all logs have consistent structure and formatting,
making them easier to parse and analyze.
"""

import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime


class LogFormatter:
    """
    Base log formatter class

    Provides common formatting utilities for all formatter types.
    """

    @staticmethod
    def format_timestamp(dt: Optional[datetime] = None) -> str:
        """
        Format timestamp consistently

        Args:
            dt: Datetime object (None for current time)

        Returns:
            Formatted timestamp string (ISO 8601)
        """
        if dt is None:
            dt = datetime.utcnow()
        return dt.isoformat() + 'Z'

    @staticmethod
    def format_level(level: str, width: int = 8) -> str:
        """
        Format log level with consistent width

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            width: Width for padding

        Returns:
            Formatted level string
        """
        return level.ljust(width)

    @staticmethod
    def sanitize_value(value: Any) -> Any:
        """
        Sanitize value for logging (remove sensitive data)

        Args:
            value: Value to sanitize

        Returns:
            Sanitized value
        """
        # List of sensitive keys that should be redacted
        sensitive_keys = {
            'password', 'token', 'secret', 'api_key', 'apikey',
            'auth', 'authorization', 'credential', 'private_key'
        }

        if isinstance(value, dict):
            return {
                k: '***REDACTED***' if any(sk in k.lower() for sk in sensitive_keys)
                else LogFormatter.sanitize_value(v)
                for k, v in value.items()
            }
        elif isinstance(value, (list, tuple)):
            return [LogFormatter.sanitize_value(v) for v in value]
        else:
            return value


class JSONFormatter(LogFormatter):
    """
    JSON log formatter

    Formats logs as JSON objects for easy parsing by log aggregators
    (ELK, Loki, CloudWatch, etc.)
    """

    def __init__(self, service_name: str, extra_fields: Optional[Dict[str, Any]] = None):
        """
        Initialize JSON formatter

        Args:
            service_name: Name of the service
            extra_fields: Additional fields to include in every log
        """
        self.service_name = service_name
        self.extra_fields = extra_fields or {}

    def format(
        self,
        level: str,
        message: str,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> str:
        """
        Format log message as JSON

        Args:
            level: Log level
            message: Log message
            timestamp: Timestamp (None for current time)
            **kwargs: Additional fields

        Returns:
            JSON-formatted log string
        """
        log_entry = {
            'timestamp': self.format_timestamp(timestamp),
            'service': self.service_name,
            'level': level,
            'message': message,
            **self.extra_fields,
            **self.sanitize_value(kwargs)
        }

        return json.dumps(log_entry, default=str)


class TraceFormatter(LogFormatter):
    """
    Trace-aware log formatter

    Integrates OpenTelemetry trace context into log messages for
    distributed tracing correlation.
    """

    def __init__(
        self,
        service_name: str,
        include_trace_id: bool = True,
        include_span_id: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trace formatter

        Args:
            service_name: Name of the service
            include_trace_id: Include trace ID in logs
            include_span_id: Include span ID in logs
            extra_fields: Additional fields to include
        """
        self.service_name = service_name
        self.include_trace_id = include_trace_id
        self.include_span_id = include_span_id
        self.extra_fields = extra_fields or {}

    def format(
        self,
        level: str,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> str:
        """
        Format log message with trace context

        Args:
            level: Log level
            message: Log message
            trace_id: OpenTelemetry trace ID
            span_id: OpenTelemetry span ID
            timestamp: Timestamp (None for current time)
            **kwargs: Additional fields

        Returns:
            Formatted log string with trace context
        """
        parts = [
            self.format_timestamp(timestamp),
            self.format_level(level),
            f"[{self.service_name}]"
        ]

        # Add trace context if available
        if self.include_trace_id and trace_id:
            parts.append(f"[trace:{trace_id[:8]}]")

        if self.include_span_id and span_id:
            parts.append(f"[span:{span_id[:8]}]")

        parts.append(message)

        # Add extra fields
        if kwargs or self.extra_fields:
            extra = {**self.extra_fields, **kwargs}
            extra_str = " | " + " | ".join(
                f"{k}={v}" for k, v in self.sanitize_value(extra).items()
            )
            parts.append(extra_str)

        return " ".join(parts)

    def format_json(
        self,
        level: str,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> str:
        """
        Format log message as JSON with trace context

        Args:
            level: Log level
            message: Log message
            trace_id: OpenTelemetry trace ID
            span_id: OpenTelemetry span ID
            timestamp: Timestamp (None for current time)
            **kwargs: Additional fields

        Returns:
            JSON-formatted log string with trace context
        """
        log_entry = {
            'timestamp': self.format_timestamp(timestamp),
            'service': self.service_name,
            'level': level,
            'message': message,
            **self.extra_fields,
            **self.sanitize_value(kwargs)
        }

        # Add trace context if available
        if self.include_trace_id and trace_id:
            log_entry['trace_id'] = trace_id

        if self.include_span_id and span_id:
            log_entry['span_id'] = span_id

        return json.dumps(log_entry, default=str)


class StructuredFormatter(LogFormatter):
    """
    Structured log formatter

    Formats logs with structured key-value pairs for easy parsing
    while maintaining human readability.
    """

    def __init__(
        self,
        service_name: str,
        field_separator: str = " | ",
        kv_separator: str = "=",
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize structured formatter

        Args:
            service_name: Name of the service
            field_separator: Separator between fields
            kv_separator: Separator between key and value
            extra_fields: Additional fields to include
        """
        self.service_name = service_name
        self.field_separator = field_separator
        self.kv_separator = kv_separator
        self.extra_fields = extra_fields or {}

    def format(
        self,
        level: str,
        message: str,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> str:
        """
        Format log message with structured fields

        Args:
            level: Log level
            message: Log message
            timestamp: Timestamp (None for current time)
            **kwargs: Additional fields

        Returns:
            Structured log string
        """
        fields = [
            f"timestamp{self.kv_separator}{self.format_timestamp(timestamp)}",
            f"level{self.kv_separator}{level}",
            f"service{self.kv_separator}{self.service_name}",
            f"message{self.kv_separator}{message}"
        ]

        # Add extra fields
        all_extra = {**self.extra_fields, **kwargs}
        if all_extra:
            sanitized = self.sanitize_value(all_extra)
            fields.extend([
                f"{k}{self.kv_separator}{v}"
                for k, v in sanitized.items()
            ])

        return self.field_separator.join(fields)


def get_formatter(
    formatter_type: str,
    service_name: str,
    **kwargs
) -> LogFormatter:
    """
    Factory function to get appropriate formatter

    Args:
        formatter_type: Type of formatter (json, trace, structured, text)
        service_name: Name of the service
        **kwargs: Additional arguments for formatter

    Returns:
        LogFormatter instance

    Raises:
        ValueError: If formatter type is unknown
    """
    formatters = {
        'json': JSONFormatter,
        'trace': TraceFormatter,
        'structured': StructuredFormatter,
        'text': LogFormatter
    }

    formatter_class = formatters.get(formatter_type.lower())
    if formatter_class is None:
        raise ValueError(
            f"Unknown formatter type: {formatter_type}. "
            f"Available: {', '.join(formatters.keys())}"
        )

    return formatter_class(service_name, **kwargs)
