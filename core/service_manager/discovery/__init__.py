"""
Service Discovery Module

Automatic service discovery, installation, and lifecycle testing.
"""

from .service_discovery import (
    ServiceDiscovery,
    ServiceType,
    ServiceStatus,
    ServiceInfo,
    discover_services
)
from .bulk_installer import (
    BulkInstaller,
    InstallationStatus,
    InstallationResult
)
from .lifecycle_tester import (
    ServiceLifecycleTester,
    CycleResult,
    ServiceLifecycleResult
)

__all__ = [
    # Service Discovery
    "ServiceDiscovery",
    "ServiceType",
    "ServiceStatus",
    "ServiceInfo",
    "discover_services",
    # Bulk Installation
    "BulkInstaller",
    "InstallationStatus",
    "InstallationResult",
    # Lifecycle Testing
    "ServiceLifecycleTester",
    "CycleResult",
    "ServiceLifecycleResult",
]
