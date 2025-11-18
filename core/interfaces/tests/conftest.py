"""
Pytest configuration for interface tests

Configures pytest fixtures and settings for testing interfaces.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_session_data():
    """Sample session data for testing"""
    return {
        "user_id": "test-user-123",
        "metadata": {
            "device": "web",
            "browser": "chrome",
            "location": "Brazil"
        }
    }


@pytest.fixture
def sample_message_data():
    """Sample message data for testing"""
    return {
        "role": "user",
        "content": "Hello, how are you?",
        "metadata": {
            "audio_duration_ms": 1500,
            "language": "pt-BR"
        }
    }


@pytest.fixture
def sample_profile_config():
    """Sample profile configuration for testing"""
    return {
        "name": "test-profile",
        "description": "Test profile for unit tests",
        "enabled_services": [
            "orchestrator",
            "api_gateway",
            "external_llm",
            "external_stt",
            "external_tts"
        ],
        "service_overrides": {
            "external_llm": {
                "model": "test-model",
                "temperature": 0.5
            }
        }
    }


@pytest.fixture
def sample_gpu_config():
    """Sample GPU configuration for testing"""
    return {
        "device_id": 0,
        "max_memory_mb": 24000,
        "compute_capability": (8, 6)
    }
