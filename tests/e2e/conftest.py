"""
Pytest configuration for E2E tests
"""
import pytest
import os
from pathlib import Path


@pytest.fixture(scope="session")
def test_audio_path():
    """Path to test audio file"""
    audio_path = Path(__file__).parent.parent / "fixtures" / "test_audio_real_speech.wav"
    if not audio_path.exists():
        pytest.skip("Test audio file not found in tests/fixtures/")
    return audio_path


@pytest.fixture(scope="session")
def services_config():
    """Configuration for service URLs"""
    return {
        "stt_url": os.getenv("STT_SERVICE_URL", "http://localhost:8099"),
        "llm_url": os.getenv("LLM_SERVICE_URL", "http://localhost:8110"),
        "tts_url": os.getenv("TTS_SERVICE_URL", "http://localhost:8103"),
        "orchestrator_url": os.getenv("ORCHESTRATOR_URL", "http://localhost:8080"),
        "websocket_url": os.getenv("WEBSOCKET_URL", "http://localhost:8022"),
    }

