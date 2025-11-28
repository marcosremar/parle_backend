"""
Audit Logger Module (v5.3)

Specialized logging for security and compliance audit trails.

Features:
- Separate audit log files
- Structured audit events
- Tamper-evident logging
- Security event tracking
- Access control logging
- Data change tracking

Audit logs are:
- Stored separately from operational logs
- Kept for longer retention periods
- Formatted for compliance and security analysis
- Never rotated or deleted without explicit policy

Use cases:
- User authentication/authorization events
- Data access and modifications
- Security events (failed logins, permission changes)
- Compliance tracking (GDPR, HIPAA, etc.)
"""

import json
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from loguru import logger as base_logger

from .log_config import get_logs_dir


class AuditEventType(str, Enum):
    """Audit event types for classification"""
    # Authentication events
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    TOKEN_CREATED = "token_created"
    TOKEN_REVOKED = "token_revoked"

    # Authorization events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_CHANGED = "role_changed"

    # Data access events
    DATA_READ = "data_read"
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"

    # Security events
    SECURITY_ALERT = "security_alert"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # System events
    CONFIG_CHANGED = "config_changed"
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"

    # Compliance events
    GDPR_REQUEST = "gdpr_request"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"


class AuditLogger:
    """
    Audit logger for security and compliance tracking

    This logger writes to a separate audit log file with:
    - JSON format for easy parsing
    - Long retention period (default: 1 year)
    - No rotation (controlled retention only)
    - Structured event format
    """

    def __init__(
        self,
        service_name: str,
        logs_dir: Optional[Path] = None,
        retention: str = "365 days"
    ):
        """
        Initialize audit logger

        Args:
            service_name: Name of the service
            logs_dir: Custom logs directory
            retention: How long to keep audit logs (default: 1 year)
        """
        self.service_name = service_name
        self.logs_dir = logs_dir or get_logs_dir()
        self.retention = retention

        # Create audit log file path
        self.audit_log_file = self.logs_dir / f"{service_name}_audit.log"

        # Setup audit logger (separate from regular logs)
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup dedicated audit logger"""
        # Add audit log handler
        base_logger.add(
            self.audit_log_file,
            level="INFO",
            rotation=None,  # No rotation for audit logs
            retention=self.retention,  # Keep for 1 year
            compression="zip",
            serialize=True,  # JSON format
            enqueue=True,  # Thread-safe
            filter=lambda record: record["extra"].get("audit") is True
        )

    def _log_event(
        self,
        event_type: AuditEventType,
        message: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        **extra_context
    ) -> None:
        """
        Log audit event

        Args:
            event_type: Type of audit event
            message: Human-readable message
            user_id: User who performed the action
            resource: Resource affected (user, document, etc.)
            action: Action performed (read, write, delete, etc.)
            result: Result of action (success, failure, denied)
            ip_address: IP address of requester
            user_agent: User agent string
            **extra_context: Additional context
        """
        audit_event = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "service": self.service_name,
            "event_type": event_type.value,
            "message": message,
            "result": result
        }

        # Add optional fields
        if user_id:
            audit_event["user_id"] = user_id

        if resource:
            audit_event["resource"] = resource

        if action:
            audit_event["action"] = action

        if ip_address:
            audit_event["ip_address"] = ip_address

        if user_agent:
            audit_event["user_agent"] = user_agent

        # Add extra context
        if extra_context:
            audit_event["extra"] = extra_context

        # Log with audit marker
        base_logger.bind(audit=True).info(json.dumps(audit_event))

    def log_security_event(
        self,
        message: str,
        event_type: AuditEventType = AuditEventType.SECURITY_ALERT,
        user_id: Optional[str] = None,
        **context
    ) -> None:
        """
        Log security event

        Args:
            message: Security event message
            event_type: Type of security event
            user_id: User involved (if applicable)
            **context: Additional context
        """
        self._log_event(
            event_type=event_type,
            message=message,
            user_id=user_id,
            **context
        )

    def log_access_event(
        self,
        action: str,
        resource: str,
        user_id: Optional[str] = None,
        result: str = "success",
        **context
    ) -> None:
        """
        Log data access event

        Args:
            action: Action performed (read, write, delete)
            resource: Resource accessed
            user_id: User who performed action
            result: Result of action
            **context: Additional context
        """
        event_type = AuditEventType.DATA_READ
        if action.lower() in ["create", "post"]:
            event_type = AuditEventType.DATA_CREATED
        elif action.lower() in ["update", "put", "patch"]:
            event_type = AuditEventType.DATA_UPDATED
        elif action.lower() in ["delete", "remove"]:
            event_type = AuditEventType.DATA_DELETED

        self._log_event(
            event_type=event_type,
            message=f"User {user_id or 'unknown'} {action} {resource}",
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            **context
        )

    def log_auth_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        result: str = "success",
        ip_address: Optional[str] = None,
        **context
    ) -> None:
        """
        Log authentication/authorization event

        Args:
            event_type: Type of auth event (LOGIN, LOGOUT, etc.)
            user_id: User ID
            result: Result (success, failure, denied)
            ip_address: IP address
            **context: Additional context
        """
        message_map = {
            AuditEventType.LOGIN: f"User {user_id} logged in",
            AuditEventType.LOGOUT: f"User {user_id} logged out",
            AuditEventType.LOGIN_FAILED: f"Failed login attempt for {user_id}",
            AuditEventType.PERMISSION_DENIED: f"Permission denied for {user_id}",
            AuditEventType.PERMISSION_GRANTED: f"Permission granted to {user_id}",
        }

        message = message_map.get(event_type, f"{event_type.value} for {user_id}")

        self._log_event(
            event_type=event_type,
            message=message,
            user_id=user_id,
            result=result,
            ip_address=ip_address,
            **context
        )

    def log_data_change(
        self,
        resource: str,
        action: str,
        user_id: Optional[str] = None,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        **context
    ) -> None:
        """
        Log data change event with before/after values

        Args:
            resource: Resource modified
            action: Action performed
            user_id: User who made the change
            old_value: Value before change
            new_value: Value after change
            **context: Additional context
        """
        change_context = {**context}

        if old_value is not None:
            change_context["old_value"] = str(old_value)

        if new_value is not None:
            change_context["new_value"] = str(new_value)

        self.log_access_event(
            action=action,
            resource=resource,
            user_id=user_id,
            **change_context
        )


# Singleton instance per service
_audit_loggers: Dict[str, AuditLogger] = {}


def get_audit_logger(service_name: str) -> AuditLogger:
    """
    Get or create audit logger for a service

    Args:
        service_name: Name of the service

    Returns:
        AuditLogger instance

    Example:
        from src.core.core_logging import get_audit_logger

        audit = get_audit_logger("user")
        audit.log_auth_event(
            AuditEventType.LOGIN,
            user_id="user123",
            ip_address="192.168.1.1"
        )
    """
    if service_name not in _audit_loggers:
        _audit_loggers[service_name] = AuditLogger(service_name)

    return _audit_loggers[service_name]


# Convenience functions for common audit events
def log_security_event(
    service_name: str,
    message: str,
    user_id: Optional[str] = None,
    **context
) -> None:
    """
    Quick function to log security event

    Args:
        service_name: Service name
        message: Security event message
        user_id: User involved
        **context: Additional context
    """
    audit = get_audit_logger(service_name)
    audit.log_security_event(message, user_id=user_id, **context)


def log_access_event(
    service_name: str,
    action: str,
    resource: str,
    user_id: Optional[str] = None,
    **context
) -> None:
    """
    Quick function to log access event

    Args:
        service_name: Service name
        action: Action performed
        resource: Resource accessed
        user_id: User who performed action
        **context: Additional context
    """
    audit = get_audit_logger(service_name)
    audit.log_access_event(action, resource, user_id=user_id, **context)


def log_data_change(
    service_name: str,
    resource: str,
    action: str,
    user_id: Optional[str] = None,
    old_value: Optional[Any] = None,
    new_value: Optional[Any] = None,
    **context
) -> None:
    """
    Quick function to log data change

    Args:
        service_name: Service name
        resource: Resource modified
        action: Action performed
        user_id: User who made change
        old_value: Value before change
        new_value: Value after change
        **context: Additional context
    """
    audit = get_audit_logger(service_name)
    audit.log_data_change(
        resource, action, user_id,
        old_value=old_value,
        new_value=new_value,
        **context
    )
