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
        "orchestrator_url": os.getenv("ORCHESTRATOR_URL", "http://localhost:8500"),
        "websocket_url": os.getenv("WEBSOCKET_URL", "http://localhost:8022"),
        "conversation_store_url": os.getenv("CONVERSATION_STORE_URL", "http://localhost:8800"),
        "conversation_history_url": os.getenv("CONVERSATION_HISTORY_URL", "http://localhost:8501"),
        "session_url": os.getenv("SESSION_SERVICE_URL", "http://localhost:8200"),
        "scenarios_url": os.getenv("SCENARIOS_SERVICE_URL", "http://localhost:8700"),
        "user_url": os.getenv("USER_SERVICE_URL", "http://localhost:8201"),
        "database_url": os.getenv("DATABASE_SERVICE_URL", "http://localhost:8400"),
        "file_storage_url": os.getenv("FILE_STORAGE_SERVICE_URL", "http://localhost:8300"),
        "rest_polling_url": os.getenv("REST_POLLING_SERVICE_URL", "http://localhost:8701"),
        "api_gateway_url": os.getenv("API_GATEWAY_URL", "http://localhost:8000"),
    }

