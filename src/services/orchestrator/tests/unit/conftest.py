"""
Pytest fixtures for orchestrator unit tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any
import numpy as np


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    """Create mock service clients dictionary."""
    return {
        "stt": AsyncMock(),
        "llm": AsyncMock(),
        "tts": AsyncMock(),
        "session": AsyncMock(),
        "scenarios": AsyncMock(),
        "conversation_store": AsyncMock(),
        "conversation_history": AsyncMock(),
        "student_model": AsyncMock(),
        "pedagogical_policy": AsyncMock(),
        "speech_grader": AsyncMock(),
        "learning_path": AsyncMock(),
    }


@pytest.fixture
def mock_fallback_manager() -> Any:
    """Create mock fallback manager."""
    manager = AsyncMock()
    manager.call_llm_with_failover = AsyncMock(return_value={
        "text": "Mock LLM response",
        "llm_used": "primary",
        "success": True
    })
    return manager


@pytest.fixture
def mock_context_loader() -> Any:
    """Create mock context loader."""
    loader = AsyncMock()
    loader.load_context = AsyncMock(return_value=(
        {"session_id": "test_session", "user_id": "test_user"},
        {"scenario_id": "test_scenario", "name": "Test Scenario"},
        [{"role": "user", "content": "Hello"}],
        {"user_id": "test_user", "current_estimated_level": "A2"}
    ))
    return loader


@pytest.fixture
def mock_knowledge_analyzer() -> Any:
    """Create mock knowledge analyzer."""
    analyzer = AsyncMock()
    analyzer.analyze_and_update_knowledge = AsyncMock()
    return analyzer


@pytest.fixture
def mock_stats_tracker() -> Any:
    """Create mock stats tracker."""
    from src.services.orchestrator.engines.stats_tracker import StatsTracker
    return StatsTracker()


@pytest.fixture
def sample_audio_data() -> bytes:
    """Create sample audio data (1 second of silence at 16kHz)."""
    # Generate 1 second of silence (16-bit PCM)
    samples = np.zeros(16000, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def sample_session_data() -> Dict[str, Any]:
    """Create sample session data."""
    return {
        "session_id": "test_session",
        "user_id": "test_user",
        "conversation_id": "test_conv",
        "scenario_id": "test_scenario"
    }


@pytest.fixture
def sample_scenario_data() -> Dict[str, Any]:
    """Create sample scenario data."""
    return {
        "scenario_id": "test_scenario",
        "name": "Test Scenario",
        "description": "A test scenario",
        "expected_topics": ["greeting", "introduction"]
    }


@pytest.fixture
def sample_conversation_history() -> list:
    """Create sample conversation history."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"}
    ]


@pytest.fixture
def sample_student_data() -> Dict[str, Any]:
    """Create sample student CEFR progress data."""
    return {
        "user_id": "test_user",
        "current_estimated_level": "A2",
        "cefr_details": {
            "A1": {"mastery": 0.8},
            "A2": {"mastery": 0.6}
        }
    }


@pytest.fixture
def sample_turn_analysis() -> Dict[str, Any]:
    """Create sample turn analysis."""
    return {
        "errors": [],
        "correct_skills": ["skill_1", "skill_2"],
        "linguistic_features": {
            "complexity": "medium",
            "fluency": "good"
        },
        "semantic_skill_mapping": {
            "skill_1": 0.9,
            "skill_2": 0.8
        }
    }
