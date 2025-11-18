"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for all Service Manager tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.service_manager.discovery import (
    ServiceDiscovery,
    ServiceType,
    ServiceStatus,
    ServiceInfo,
    BulkInstaller,
    InstallationStatus,
    InstallationResult
)


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_services_dir():
    """Create a temporary services directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_services_dir(temp_services_dir):
    """Create a mock services directory structure."""
    # Create AI model services
    (temp_services_dir / "llm").mkdir()
    (temp_services_dir / "llm" / "service.py").write_text("# LLM service")
    (temp_services_dir / "llm" / "install.py").write_text("# LLM install")

    (temp_services_dir / "stt").mkdir()
    (temp_services_dir / "stt" / "service.py").write_text("# STT service")
    (temp_services_dir / "stt" / "install.py").write_text("# STT install")

    # Create infrastructure services
    (temp_services_dir / "api_gateway").mkdir()
    (temp_services_dir / "api_gateway" / "service.py").write_text("# API Gateway")

    (temp_services_dir / "websocket").mkdir()
    (temp_services_dir / "websocket" / "service.py").write_text("# WebSocket")

    # Create utility services
    (temp_services_dir / "file_storage").mkdir()
    (temp_services_dir / "file_storage" / "service.py").write_text("# File Storage")

    return temp_services_dir


# ============================================================================
# Service Discovery Fixtures
# ============================================================================

@pytest.fixture
def service_discovery(mock_services_dir):
    """Create a ServiceDiscovery instance with mock services."""
    discovery = ServiceDiscovery(services_dir=mock_services_dir)
    discovery.discover_all_services()
    return discovery


@pytest.fixture
def mock_service_info():
    """Create a mock ServiceInfo object."""
    return ServiceInfo(
        service_id="test_service",
        service_type=ServiceType.INFRASTRUCTURE,
        status=ServiceStatus.AVAILABLE,
        path=Path(str(Path.home() / ".cache" / "ultravox-pipeline" / "test_service"),
        has_install_script=False,
        has_service_file=True,
        has_config=False,
        port=8000,
        dependencies=[],
        health_url="http://localhost:8000/health",
        description="Test Service",
        install_time_estimate="< 1 min",
        disk_space_required="< 100MB"
    )


@pytest.fixture
def mock_ai_service_info():
    """Create a mock AI model ServiceInfo object."""
    return ServiceInfo(
        service_id="llm",
        service_type=ServiceType.AI_MODEL,
        status=ServiceStatus.NOT_INSTALLED,
        path=Path(str(Path.home() / ".cache" / "ultravox-pipeline" / "llm"),
        has_install_script=True,
        has_service_file=True,
        has_config=True,
        port=8100,
        dependencies=["torch", "transformers", "vllm"],
        health_url="http://localhost:8100/health",
        description="Speech-to-Speech LLM (Ultravox)",
        install_time_estimate="10-15 min",
        disk_space_required="25GB"
    )


# ============================================================================
# Bulk Installer Fixtures
# ============================================================================

@pytest.fixture
def bulk_installer(mock_services_dir):
    """Create a BulkInstaller instance with mock services."""
    installer = BulkInstaller()
    installer.discovery = ServiceDiscovery(services_dir=mock_services_dir)
    installer.discovery.discover_all_services()
    return installer


@pytest.fixture
def mock_installation_result_success():
    """Create a mock successful InstallationResult."""
    return InstallationResult(
        service_id="test_service",
        status=InstallationStatus.SUCCESS,
        message="Installation completed successfully",
        duration_seconds=1.5,
        error=None
    )


@pytest.fixture
def mock_installation_result_failed():
    """Create a mock failed InstallationResult."""
    return InstallationResult(
        service_id="test_service",
        status=InstallationStatus.FAILED,
        message="Installation failed",
        duration_seconds=0.5,
        error="ImportError: Module not found"
    )


# ============================================================================
# Mock Service Manager Fixtures
# ============================================================================

@pytest.fixture
def mock_service_manager():
    """Create a mock ServiceManager for testing."""
    manager = MagicMock()
    manager.services = {}
    manager.execution_config = MagicMock()
    manager.discovery = MagicMock()
    return manager


# ============================================================================
# Mock HTTP Client Fixtures
# ============================================================================

@pytest.fixture
def mock_requests_get_healthy(monkeypatch):
    """Mock requests.get to return healthy response."""
    def mock_get(*args, **kwargs):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"status": "healthy"}
        return response

    import requests
    monkeypatch.setattr(requests, "get", mock_get)


@pytest.fixture
def mock_requests_get_unhealthy(monkeypatch):
    """Mock requests.get to return unhealthy response."""
    def mock_get(*args, **kwargs):
        response = Mock()
        response.status_code = 500
        response.json.return_value = {"status": "error"}
        return response

    import requests
    monkeypatch.setattr(requests, "get", mock_get)


@pytest.fixture
def mock_requests_get_timeout(monkeypatch):
    """Mock requests.get to timeout."""
    import requests

    def mock_get(*args, **kwargs):
        raise requests.exceptions.Timeout("Connection timeout")

    monkeypatch.setattr(requests, "get", mock_get)


# ============================================================================
# Mock Port Checking Fixtures
# ============================================================================

@pytest.fixture
def mock_port_open(monkeypatch):
    """Mock port checking to return True (port is open)."""
    def mock_check_port(self, port):
        return True

    from src.core.service_manager.discovery import ServiceDiscovery
    monkeypatch.setattr(ServiceDiscovery, "_check_port_listening", mock_check_port)


@pytest.fixture
def mock_port_closed(monkeypatch):
    """Mock port checking to return False (port is closed)."""
    def mock_check_port(self, port):
        return False

    from src.core.service_manager.discovery import ServiceDiscovery
    monkeypatch.setattr(ServiceDiscovery, "_check_port_listening", mock_check_port)


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def service_list_all_types() -> List[ServiceInfo]:
    """Generate a list of services with all types."""
    return [
        ServiceInfo(
            service_id="llm",
            service_type=ServiceType.AI_MODEL,
            status=ServiceStatus.NOT_INSTALLED,
            path=Path("/services/llm"),
            has_install_script=True,
            has_service_file=True,
            port=8100
        ),
        ServiceInfo(
            service_id="api_gateway",
            service_type=ServiceType.INFRASTRUCTURE,
            status=ServiceStatus.INSTALLED,
            path=Path("/services/api_gateway"),
            has_install_script=False,
            has_service_file=True,
            port=8000
        ),
        ServiceInfo(
            service_id="file_storage",
            service_type=ServiceType.UTILITY,
            status=ServiceStatus.HEALTHY,
            path=Path("/services/file_storage"),
            has_install_script=False,
            has_service_file=True,
            port=8003
        ),
        ServiceInfo(
            service_id="external_llm",
            service_type=ServiceType.EXTERNAL,
            status=ServiceStatus.AVAILABLE,
            path=Path("/services/external_llm"),
            has_install_script=False,
            has_service_file=True,
            port=None
        ),
    ]


@pytest.fixture
def services_by_status() -> Dict[ServiceStatus, List[str]]:
    """Generate services grouped by status."""
    return {
        ServiceStatus.HEALTHY: ["api_gateway", "websocket"],
        ServiceStatus.INSTALLED: ["file_storage", "session"],
        ServiceStatus.NOT_INSTALLED: ["llm", "stt", "tts"],
        ServiceStatus.STOPPED: ["orchestrator"],
        ServiceStatus.AVAILABLE: ["external_llm"],
    }


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "health_monitor: marks tests for HealthMonitor class"
    )
    config.addinivalue_line(
        "markers", "auto_restart: marks tests for auto-restart functionality"
    )
    config.addinivalue_line(
        "markers", "service_restart: marks tests for ServiceManager.restart_service()"
    )


# ============================================================================
# HealthMonitor and Auto-Restart Fixtures
# ============================================================================

@pytest.fixture
def mock_health_monitor():
    """Create a HealthMonitor instance with test configuration."""
    from src.core.service_manager.core.health_monitor import HealthMonitor

    monitor = HealthMonitor(
        check_interval=2,  # Fast for testing
        timeout=1,
        max_history=10,
        parallel_checks=False,  # Sequential for predictable tests
        circuit_breaker_threshold=3,
        auto_restart_enabled=True,
        auto_restart_threshold=3,
        max_restart_attempts=3,
        restart_cooldown_minutes=5
    )
    return monitor


@pytest.fixture
def mock_failing_service():
    """Create a mock service that fails health checks."""
    from datetime import datetime
    from src.core.service_manager.core.health_monitor import HealthStatus

    service = Mock()
    service.service_id = "failing_service"
    service.port = 9999
    service.health_status = HealthStatus.UNHEALTHY
    service.health_details = {"reason": "Service not responding"}
    service.last_check = datetime.now()
    service.process_status = "RUNNING"
    service.pid = 12345
    return service


@pytest.fixture
def mock_healthy_service():
    """Create a mock service that passes health checks."""
    from datetime import datetime
    from src.core.service_manager.core.health_monitor import HealthStatus

    service = Mock()
    service.service_id = "healthy_service"
    service.port = 8888
    service.health_status = HealthStatus.HEALTHY
    service.health_details = {"status": "responding"}
    service.last_check = datetime.now()
    service.process_status = "RUNNING"
    service.pid = 12346
    return service


@pytest.fixture
def mock_module_service():
    """Create a mock MODULE service (in-process, no HTTP endpoint)."""
    from datetime import datetime
    from src.core.service_manager.core.health_monitor import HealthStatus

    service = Mock()
    service.service_id = "module_service"
    service.name = "Module Service"
    service.port = None  # MODULE services don't have ports
    service.health_status = HealthStatus.HEALTHY
    service.health_details = {"type": "module_service", "in_process": True}
    service.last_check = datetime.now()
    service.process_status = None  # No separate process
    service.pid = None
    return service


@pytest.fixture
def mock_service_with_port():
    """Create a mock HTTP service with port."""
    from datetime import datetime
    from src.core.service_manager.core.health_monitor import HealthStatus

    service = Mock()
    service.service_id = "http_service"
    service.name = "HTTP Service"
    service.port = 8000
    service.health_status = HealthStatus.UNKNOWN
    service.health_details = {}
    service.last_check = datetime.now()
    service.process_status = "RUNNING"
    service.pid = 12347
    return service


@pytest.fixture
def mock_service_manager_with_services(mock_healthy_service, mock_failing_service):
    """Create a mock ServiceManager with multiple services."""
    manager = MagicMock()
    manager.services = {
        "healthy_service": mock_healthy_service,
        "failing_service": mock_failing_service
    }
    manager.execution_config = MagicMock()
    manager.execution_config.is_module.return_value = False

    # Mock restart_service method
    async def mock_restart(service_id: str) -> bool:
        return True

    manager.restart_service = Mock(side_effect=mock_restart)
    manager.check_port = Mock(return_value=True)
    manager._is_internal_service = Mock(return_value=False)

    return manager


@pytest.fixture
async def running_service_manager():
    """
    Create a ServiceManager instance for integration tests.
    Note: This requires proper setup and may be slow.
    """
    # This is a placeholder - actual implementation would need:
    # 1. Start Service Manager
    # 2. Wait for initialization
    # 3. Yield for tests
    # 4. Cleanup
    pytest.skip("running_service_manager requires full Service Manager setup")


@pytest.fixture
def mock_restart_history():
    """Create mock restart history data."""
    from datetime import datetime, timedelta
    from collections import deque

    now = datetime.now()
    history = deque(maxlen=5)

    # Add 3 recent restarts
    history.append(now - timedelta(minutes=10))
    history.append(now - timedelta(minutes=5))
    history.append(now - timedelta(minutes=2))

    return history
