"""
Type definitions for Orchestrator service

All data structures and type aliases are defined here for type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Type Aliases
ServiceConfig = dict[str, str]
StatsDict = dict[str, Any]
HealthStatus = dict[str, bool]


@dataclass
class TurnResponse:
    """Response from process_turn() method"""

    success: bool
    text: str | None = None
    audio: bytes | None = None
    transcript: str | None = None
    session_id: str | None = None
    llm_used: str | None = None
    voice_id: str | None = None
    circuit_state: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses"""
        result: dict[str, Any] = {
            "success": self.success,
        }
        if self.text is not None:
            result["text"] = self.text
        if self.audio is not None:
            result["audio"] = self.audio
        if self.transcript is not None:
            result["transcript"] = self.transcript
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.llm_used is not None:
            result["llm_used"] = self.llm_used
        if self.voice_id is not None:
            result["voice_id"] = self.voice_id
        if self.circuit_state is not None:
            result["circuit_state"] = self.circuit_state
        if self.metrics is not None:
            result["metrics"] = self.metrics
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class TextConversationResponse:
    """Response from process_text_conversation() method"""

    success: bool
    response: str | None = None
    session_id: str | None = None
    audio: str | None = None  # base64 encoded
    context_size: int | None = None
    messages_count: int | None = None
    metrics: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses"""
        result: dict[str, Any] = {
            "success": self.success,
        }
        if self.response is not None:
            result["response"] = self.response
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.audio is not None:
            result["audio"] = self.audio
        if self.context_size is not None:
            result["context_size"] = self.context_size
        if self.messages_count is not None:
            result["messages_count"] = self.messages_count
        if self.metrics is not None:
            result["metrics"] = self.metrics
        if self.validation is not None:
            result["validation"] = self.validation
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class StructuredTurnResponse:
    """Response from process_turn_structured() method"""

    success: bool
    text: str | None = None
    audio: bytes | None = None
    transcript: str | None = None
    session_id: str | None = None
    voice_id: str | None = None
    validation: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses"""
        result: dict[str, Any] = {
            "success": self.success,
        }
        if self.text is not None:
            result["text"] = self.text
        if self.audio is not None:
            result["audio"] = self.audio
        if self.transcript is not None:
            result["transcript"] = self.transcript
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.voice_id is not None:
            result["voice_id"] = self.voice_id
        if self.validation is not None:
            result["validation"] = self.validation
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.metrics is not None:
            result["metrics"] = self.metrics
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class SessionData:
    """Session data structure"""

    session_id: str
    conversation_id: str | None = None
    scenario_id: str | None = None
    user_id: str | None = None
    voice_id: str | None = None


@dataclass
class ScenarioData:
    """Scenario data structure"""

    scenario_id: str
    name: str | None = None
    system_prompt: str | None = None
    type: str | None = None
    expected_topics: list[str] = field(default_factory=list)
    ai_role: str | None = None
    user_role: str | None = None
    language: str | None = None


@dataclass
class ConversationHistory:
    """Conversation history structure"""

    messages: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None


@dataclass
class StudentCEFRProgress:
    """Student CEFR progress data"""

    user_id: str
    current_level: str | None = None
    current_estimated_level: str | None = None
    cefr_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetSkill:
    """Target skill data"""

    skill_id: str | None = None
    skill_name: str | None = None
    mastery_probability: float = 0.0


@dataclass
class TurnAnalysis:
    """Turn analysis results"""

    errors: list[dict[str, Any]] = field(default_factory=list)
    correct_skills: list[str] = field(default_factory=list)
    linguistic_features: dict[str, Any] = field(default_factory=dict)
    semantic_skill_mapping: dict[str, float] = field(default_factory=dict)
    summary: str | None = None


@dataclass
class SkillsExtraction:
    """Skills extraction results"""

    skills: list[dict[str, Any]] = field(default_factory=list)
    overall_linguistic_features: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None


@dataclass
class ProcessingMetrics:
    """Processing metrics"""

    input_audio_size: int = 0
    output_audio_size: int = 0
    processing_time_ms: int = 0
    llm_used: str | None = None
    has_tts: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "input_audio_size": self.input_audio_size,
            "output_audio_size": self.output_audio_size,
            "processing_time_ms": self.processing_time_ms,
            "llm_used": self.llm_used,
            "has_tts": self.has_tts,
        }
