"""
Service Manager - Modular Managers Package.

Part of Phase 3 refactoring - split monolithic ServiceManager
into focused, single-responsibility managers.

Managers:
- ProcessManager: Process lifecycle and monitoring
- ServiceConfigurationManager: Configuration loading and validation
- ServiceLifecycleManager: Start/stop/restart operations
- ServiceHealthMonitor: Health monitoring coordination
- HTTPServerManager: FastAPI app setup and router registration (NEW)
- ServiceManagerFacade: Simple unified interface (RECOMMENDED)

Usage (Recommended - Facade + HTTP):
    from src.core.service_manager.managers import ServiceManagerFacade, HTTPServerManager

    facade = ServiceManagerFacade()
    http_manager = HTTPServerManager(facade)
    app = http_manager.create_app()

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)

Usage (Advanced - Individual Managers):
    from src.core.service_manager.managers import (
        ProcessManager,
        ServiceConfigurationManager,
        ServiceLifecycleManager,
        ServiceHealthMonitor,
        HTTPServerManager
    )

    process_mgr = ProcessManager()
    config_mgr = ServiceConfigurationManager()
    lifecycle_mgr = ServiceLifecycleManager(config_mgr, process_mgr)
    health_monitor = ServiceHealthMonitor(config_mgr)
    http_manager = HTTPServerManager(facade)
"""

from .process_manager import ProcessManager
from .service_configuration_manager import (
    ServiceConfigurationManager,
    ServiceConfig,
    ProcessStatus
)
from .service_lifecycle_manager import ServiceLifecycleManager
from .service_health_monitor import ServiceHealthMonitor
from .http_server_manager import HTTPServerManager
from .service_manager_facade import ServiceManagerFacade

__all__ = [
    "ProcessManager",
    "ServiceConfigurationManager",
    "ServiceLifecycleManager",
    "ServiceHealthMonitor",
    "HTTPServerManager",
    "ServiceManagerFacade",
    "ServiceConfig",
    "ProcessStatus",
]
