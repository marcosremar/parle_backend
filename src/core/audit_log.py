"""
Audit logging for sensitive actions
Tracks user actions for security and compliance
"""

from datetime import datetime, timezone
from typing import Any

from loguru import logger


class AuditLogger:
    """
    Audit logger for tracking sensitive user actions

    Logs actions such as:
    - Authentication events (login, logout, registration)
    - Data modifications (create, update, delete)
    - Permission changes
    - Access to sensitive data
    """

    def __init__(self):
        self.logger = logger.bind(component="audit")

    def log_auth_event(
        self,
        event_type: str,
        user_id: str | None = None,
        email: str | None = None,
        ip_address: str | None = None,
        success: bool = True,
        details: dict[str, Any] | None = None,
    ):
        """
        Log authentication event

        Args:
            event_type: Type of event (login, logout, register, password_reset, etc.)
            user_id: User ID (if available)
            email: User email
            ip_address: Client IP address
            success: Whether the action was successful
            details: Additional details
        """
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "auth",
            "action": event_type,
            "user_id": user_id,
            "email": email,
            "ip_address": ip_address,
            "success": success,
            "details": details or {},
        }

        self.logger.info(f"Audit: {event_type}", **audit_data)

    def log_data_change(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        changes: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ):
        """
        Log data modification event

        Args:
            action: Action performed (create, update, delete)
            resource_type: Type of resource (user, conversation, message, etc.)
            resource_id: ID of the resource
            user_id: User who performed the action
            changes: Dictionary of changes (for update actions)
            ip_address: Client IP address
        """
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "data_change",
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "user_id": user_id,
            "ip_address": ip_address,
            "changes": changes or {},
        }

        self.logger.info(f"Audit: {action} {resource_type} {resource_id}", **audit_data)

    def log_access_event(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        action: str = "access",
        ip_address: str | None = None,
    ):
        """
        Log access to sensitive resources

        Args:
            resource_type: Type of resource accessed
            resource_id: ID of the resource
            user_id: User who accessed the resource
            action: Type of access (read, download, export, etc.)
            ip_address: Client IP address
        """
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "access",
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "user_id": user_id,
            "ip_address": ip_address,
        }

        self.logger.info(f"Audit: {action} {resource_type} {resource_id}", **audit_data)

    def log_security_event(
        self,
        event_type: str,
        severity: str = "medium",
        user_id: str | None = None,
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Log security-related event

        Args:
            event_type: Type of security event (suspicious_activity, rate_limit_exceeded, etc.)
            severity: Severity level (low, medium, high, critical)
            user_id: User ID (if applicable)
            ip_address: Client IP address
            details: Additional details
        """
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "security",
            "security_event": event_type,
            "severity": severity,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {},
        }

        log_level = "warning" if severity in ["high", "critical"] else "info"
        getattr(self.logger, log_level)(f"Security Audit: {event_type}", **audit_data)


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Convenience functions
def log_auth_event(event_type: str, user_id: str | None = None, **kwargs):
    """Log authentication event"""
    get_audit_logger().log_auth_event(event_type, user_id=user_id, **kwargs)


def log_data_change(action: str, resource_type: str, resource_id: str, user_id: str, **kwargs):
    """Log data modification"""
    get_audit_logger().log_data_change(action, resource_type, resource_id, user_id, **kwargs)


def log_access_event(resource_type: str, resource_id: str, user_id: str, **kwargs):
    """Log resource access"""
    get_audit_logger().log_access_event(resource_type, resource_id, user_id, **kwargs)


def log_security_event(event_type: str, **kwargs):
    """Log security event"""
    get_audit_logger().log_security_event(event_type, **kwargs)
