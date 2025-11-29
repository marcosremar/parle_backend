"""
Shared fixtures for tests - Audio, Text, Mocks
"""

import pytest
import base64
import wave
import io
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any


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


# ============================================================================
# Mock API Clients
# ============================================================================

@pytest.fixture
def mock_stt_client():
    """Mock STT client (Groq)"""
    client = AsyncMock()
    client.transcribe = AsyncMock(return_value={
        "text": "transcribed text",
        "language": "pt",
        "duration": 1.0
    })
    return client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client (Groq)"""
    client = AsyncMock()
    client.generate = AsyncMock(return_value="AI response text")
    return client


@pytest.fixture
def mock_tts_client():
    """Mock TTS client"""
    client = AsyncMock()
    client.synthesize = AsyncMock(return_value=b"fake_audio_data")
    return client


@pytest.fixture
def mock_service_clients(mock_stt_client, mock_llm_client, mock_tts_client):
    """Mock service clients dictionary"""
    return {
        "stt": mock_stt_client,
        "llm": mock_llm_client,
        "tts": mock_tts_client
    }


# ============================================================================
# Module Fixtures
# ============================================================================

@pytest.fixture
async def stt_module():
    """Create and initialize STT module"""
    from src.modules import module_factory
    stt = module_factory.create("stt")
    await stt.initialize()
    yield stt


@pytest.fixture
async def tts_module():
    """Create and initialize TTS module"""
    from src.modules import module_factory
    tts = module_factory.create("tts")
    await tts.initialize()
    yield tts


@pytest.fixture
async def llm_module():
    """Create and initialize LLM module"""
    from src.modules import module_factory
    llm = module_factory.create("llm")
    await llm.initialize()
    yield llm


@pytest.fixture
async def orchestrator_module():
    """Create and initialize Orchestrator module"""
    from src.modules import module_factory
    orchestrator = module_factory.create("orchestrator")
    await orchestrator.initialize()
    yield orchestrator


# ============================================================================
# Session and Scenario Fixtures
# ============================================================================

@pytest.fixture
def sample_session_id():
    """Sample session ID"""
    return "test_session_123"


@pytest.fixture
def sample_user_id():
    """Sample user ID"""
    return "test_user_123"


@pytest.fixture
def sample_scenario_id():
    """Sample scenario ID"""
    return "test_scenario_123"


@pytest.fixture
def mock_session_data(sample_session_id, sample_user_id, sample_scenario_id):
    """Mock session data"""
    return {
        "id": sample_session_id,
        "user_id": sample_user_id,
        "scenario_id": sample_scenario_id,
        "conversation_id": "test_conversation_123",
        "voice_id": "Rachel",
        "language": "pt"
    }


@pytest.fixture
def mock_scenario_data(sample_scenario_id):
    """Mock scenario data"""
    return {
        "id": sample_scenario_id,
        "name": "Test Scenario",
        "system_prompt": "You are a helpful AI assistant.",
        "description": "A test scenario for unit tests"
    }


@pytest.fixture
def mock_conversation_history():
    """Mock conversation history"""
    return [
        {"role": "user", "content": "Olá"},
        {"role": "assistant", "content": "Olá! Como posso ajudar?"},
        {"role": "user", "content": "Qual é o seu nome?"},
        {"role": "assistant", "content": "Meu nome é Parle."}
    ]
