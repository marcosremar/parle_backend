"""
Service Manager Core Module

Core functionality for service orchestration, health monitoring,
and service lifecycle management.
"""

# Health monitoring (✅ extracted)
from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    HealthCheckResult,
    HEALTH_CHECK_TIMEOUT,
    # background_health_checker removed - use HealthMonitor class instead
)

# Service orchestration (⏭️ to be extracted)
# from .orchestrator import ServiceManager

# Service control (⏭️ to be extracted)
# from .service_controller import ServiceController

__all__ = [
    # Health monitoring
    "HealthMonitor",
    "HealthStatus",
    "HealthCheckResult",
    "HEALTH_CHECK_TIMEOUT",
    # background_health_checker removed - deprecated
    # Service orchestration (commented out until extracted)
    # "ServiceManager",
    # Service control (commented out until extracted)
    # "ServiceController",
]
