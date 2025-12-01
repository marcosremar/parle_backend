"""
Global pytest configuration for Parle Backend tests
"""

import pytest
import os
import sys
import base64
import wave
import io
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["MONOLITH_MODE"] = "true"
os.environ["TESTING"] = "true"

# Disable tutoring modules by default in tests (unless explicitly enabled)
if "ENABLE_TUTORING_MODULES" not in os.environ:
    os.environ["ENABLE_TUTORING_MODULES"] = "false"


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path"""
    return project_root


@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Return the test data directory"""
    return project_root_path / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def reset_module_cache():
    """
    Reset module factory cache between tests.
    
    Ensures test isolation by clearing module cache before and after each test.
    This prevents state leakage between tests.
    """
    from src.modules import module_factory
    module_factory.clear_cache()
    yield
    module_factory.clear_cache()


@pytest.fixture(autouse=True)
def isolate_test_environment():
    """
    Isolate test environment from production settings.
    
    Ensures tests don't affect or depend on production configuration.
    """
    # Save original environment
    original_env = os.environ.copy()
    
    # Set test environment
    os.environ["TESTING"] = "true"
    os.environ["ENVIRONMENT"] = "test"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_module(monkeypatch):
    """
    Create a mock module for testing.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        
    Returns:
        Function to create mock modules
    """
    def _create_mock(module_name: str, **methods):
        """Create mock module with specified methods"""
        mock = AsyncMock()
        for method_name, return_value in methods.items():
            if asyncio.iscoroutinefunction(return_value):
                setattr(mock, method_name, return_value)
            else:
                setattr(mock, method_name, AsyncMock(return_value=return_value))
        return mock
    
    return _create_mock


@pytest.fixture
def clean_database():
    """
    Provide a clean database for each test.
    
    Creates isolated database state for each test to ensure
    no data leakage between tests.
    """
    # This would connect to test database and clean it
    # Implementation depends on database setup
    yield
    # Cleanup after test
    pass


@pytest.fixture
def authenticated_client():
    """
    Create authenticated test client.
    
    Returns FastAPI test client with valid JWT token for testing
    authenticated endpoints.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    client = TestClient(app)
    
    # Register and login to get token
    # This is a placeholder - implement with actual auth
    # client.post("/api/v1/auth/register", json={...})
    # response = client.post("/api/v1/auth/login", json={...})
    # token = response.json()["token"]
    # client.headers = {"Authorization": f"Bearer {token}"}
    
    return client


# ============================================================================
# Audio Fixtures
# ============================================================================

@pytest.fixture
def sample_audio_bytes():
    """Create test audio in bytes (PCM int16, 16kHz, 1 second of silence)"""
    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    
    # Generate silence (zeros)
    samples = np.zeros(num_samples, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def sample_audio_base64(sample_audio_bytes):
    """Create test audio in base64"""
    return base64.b64encode(sample_audio_bytes).decode('utf-8')


@pytest.fixture
def sample_wav_base64():
    """Create valid WAV file in base64"""
    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00\x00' * num_samples)
    
    wav_data = wav_buffer.getvalue()
    return base64.b64encode(wav_data).decode('utf-8')


@pytest.fixture
def long_audio_base64():
    """Create longer test audio (5 seconds)"""
    sample_rate = 16000
    duration = 5
    num_samples = sample_rate * duration
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00\x00' * num_samples)
    
    wav_data = wav_buffer.getvalue()
    return base64.b64encode(wav_data).decode('utf-8')


@pytest.fixture
def corrupted_audio_base64():
    """Create corrupted/invalid audio data"""
    return base64.b64encode(b"not a valid audio file").decode('utf-8')


@pytest.fixture
def empty_audio_base64():
    """Create empty audio (just WAV header)"""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b'')
    
    wav_data = wav_buffer.getvalue()
    return base64.b64encode(wav_data).decode('utf-8')


# ============================================================================
# Text Fixtures
# ============================================================================

@pytest.fixture
def sample_text_pt():
    """Sample text in Portuguese"""
    return "Olá, como você está? Este é um teste."


@pytest.fixture
def sample_text_en():
    """Sample text in English"""
    return "Hello, how are you? This is a test."


@pytest.fixture
def complex_text_pt():
    """Complex text in Portuguese"""
    return "A programação de computadores é uma arte que combina lógica, criatividade e conhecimento técnico para resolver problemas complexos através de algoritmos e estruturas de dados."


@pytest.fixture
def long_text_pt():
    """Long text in Portuguese (for testing long inputs)"""
    return "Este é um texto longo. " * 50  # ~1000 characters


@pytest.fixture
def text_with_numbers():
    """Text with numbers"""
    return "Eu tenho 25 anos e moro na rua número 123."


@pytest.fixture
def text_with_special_chars():
    """Text with special characters"""
    return "Olá! Como você está? Eu gosto de programação (Python, JavaScript) e café ☕."
