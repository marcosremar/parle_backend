"""
Type definitions for Orchestrator service

All data structures and type aliases are defined here for type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

# Type Aliases
ServiceConfig = Dict[str, str]
StatsDict = Dict[str, Any]
HealthStatus = Dict[str, bool]


@dataclass
class TurnResponse:
    """Response from process_turn() method"""
    success: bool
    text: Optional[str] = None
    audio: Optional[bytes] = None
    transcript: Optional[str] = None
    session_id: Optional[str] = None
    llm_used: Optional[str] = None
    voice_id: Optional[str] = None
    circuit_state: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result: Dict[str, Any] = {
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
    response: Optional[str] = None
    session_id: Optional[str] = None
    audio: Optional[str] = None  # base64 encoded
    context_size: Optional[int] = None
    messages_count: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result: Dict[str, Any] = {
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
    text: Optional[str] = None
    audio: Optional[bytes] = None
    transcript: Optional[str] = None
    session_id: Optional[str] = None
    voice_id: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result: Dict[str, Any] = {
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
    conversation_id: Optional[str] = None
    scenario_id: Optional[str] = None
    user_id: Optional[str] = None
    voice_id: Optional[str] = None


@dataclass
class ScenarioData:
    """Scenario data structure"""
    scenario_id: str
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    type: Optional[str] = None
    expected_topics: List[str] = field(default_factory=list)
    ai_role: Optional[str] = None
    user_role: Optional[str] = None
    language: Optional[str] = None


@dataclass
class ConversationHistory:
    """Conversation history structure"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    conversation_id: Optional[str] = None


@dataclass
class StudentCEFRProgress:
    """Student CEFR progress data"""
    user_id: str
    current_level: Optional[str] = None
    current_estimated_level: Optional[str] = None
    cefr_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetSkill:
    """Target skill data"""
    skill_id: Optional[str] = None
    skill_name: Optional[str] = None
    mastery_probability: float = 0.0


@dataclass
class TurnAnalysis:
    """Turn analysis results"""
    errors: List[Dict[str, Any]] = field(default_factory=list)
    correct_skills: List[str] = field(default_factory=list)
    linguistic_features: Dict[str, Any] = field(default_factory=dict)
    semantic_skill_mapping: Dict[str, float] = field(default_factory=dict)
    summary: Optional[str] = None


@dataclass
class SkillsExtraction:
    """Skills extraction results"""
    skills: List[Dict[str, Any]] = field(default_factory=list)
    overall_linguistic_features: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Processing metrics"""
    input_audio_size: int = 0
    output_audio_size: int = 0
    processing_time_ms: int = 0
    llm_used: Optional[str] = None
    has_tts: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "input_audio_size": self.input_audio_size,
            "output_audio_size": self.output_audio_size,
            "processing_time_ms": self.processing_time_ms,
            "llm_used": self.llm_used,
            "has_tts": self.has_tts,
        }
