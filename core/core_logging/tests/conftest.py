"""
Pytest fixtures for logging tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_logs_dir():
    """Create temporary logs directory for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup_loggers():
    """Cleanup loggers after each test"""
    from loguru import logger

    # Save original handlers
    original_handlers = logger._core.handlers.copy()

    yield

    # Remove all test handlers
    logger.remove()

    # Restore original handlers
    for handler_id in original_handlers:
        if handler_id in logger._core.handlers:
            continue
        # Handler was removed, skip restoration
