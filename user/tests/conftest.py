"""
Pytest fixtures for User Service tests
"""
import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path

from src.core.context import ServiceContext


@pytest.fixture
def mock_service_context():
    """Mock ServiceContext for testing"""
    context = Mock(spec=ServiceContext)
    context.config = {
        "service_name": "user",
        "port": 8200,
        "log_level": "INFO"
    }

    # Mock logger
    context.logger = Mock()
    context.logger.info = Mock()
    context.logger.error = Mock()
    context.logger.warning = Mock()
    context.logger.debug = Mock()
    context.logger.success = Mock()

    # Mock communication manager
    context.comm = AsyncMock()
    context.comm.call_service = AsyncMock(return_value={"status": "success"})

    # Mock GPU manager
    context.gpu = Mock()
    context.gpu.is_available = Mock(return_value=False)

    # Mock metrics
    context.metrics = Mock()

    return context


@pytest.fixture
def user_service(mock_service_context):
    """Create UserService instance with mocked context"""
    from src.services.user.service import UserService
    service = UserService(config=None, context=mock_service_context)
    return service


@pytest.fixture
def service_config():
    """Basic service configuration as dict (legacy support)"""
    return {
        "service_name": "user",
        "port": 8200,
        "log_level": "INFO"
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Configure logging for tests - runs automatically before all tests"""
    from src.core.logging_config import setup_logging
    from pathlib import Path

    # Configure logging to write to service-specific directory
    logs_dir = Path("Path("tmp/logs")  # Service-specific temp logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (this creates the log files)
    setup_logging("user", level="DEBUG", logs_dir=logs_dir)

    yield

    # Cleanup is handled by loguru automatically


@pytest.fixture
def fresh_logs():
    """Truncate log files before test to ensure only new logs are read"""
    from pathlib import Path
    import time

    logs_dir = Path("Path("tmp/logs")  # Service-specific temp logs")

    # Truncate all log files
    for log_file in logs_dir.glob("*.log"):
        log_file.write_text("")  # Clear file

    # Small delay to ensure file system sync
    time.sleep(0.01)

    yield

    # No cleanup needed


@pytest.fixture
def mock_service_context_with_real_logger():
    """Mock ServiceContext but with REAL logger (for logging integration tests)"""
    from loguru import logger

    context = Mock(spec=ServiceContext)
    context.config = {
        "service_name": "user",
        "port": 8200,
        "log_level": "INFO"
    }

    # Use REAL logger instead of mock
    context.logger = logger

    # Mock communication manager
    context.comm = AsyncMock()
    context.comm.call_service = AsyncMock(return_value={"status": "success"})

    # Mock GPU manager
    context.gpu = Mock()
    context.gpu.is_available = Mock(return_value=False)

    # Mock metrics
    context.metrics = Mock()

    return context


# NOTE: Event loop fixture removed - pytest-asyncio now provides this automatically
# Custom event_loop fixtures can cause conflicts and deprecation warnings
