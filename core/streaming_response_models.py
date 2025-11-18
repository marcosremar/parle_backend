"""
Streaming Response Models with Perceived Latency Metrics

Enhanced response models that include:
- Perceived Latency (STT + LLM + TTS)
- Internal Talk (system reasoning)
- Hints for Next Turn (adaptive instructions)
- Error Detection (user errors with suggestions)

Used by Orchestrator and API Gateway for streaming responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LatencyMetrics(BaseModel):
    """
    Perceived Latency breakdown

    Definition: Perceived Latency = STT + LLM + TTS
    What users experience from speaking until hearing full response.
    """
    stt_time_ms: float = Field(..., description="Speech-to-Text processing time")
    llm_time_ms: float = Field(..., description="LLM response generation time")
    tts_time_ms: float = Field(..., description="Text-to-Speech synthesis time")
    total_time_ms: float = Field(..., description="Total perceived latency (STT + LLM + TTS)")
    perceived_latency_ms: float = Field(..., description="Alias for total_time_ms")

    class Config:
        description = "Perceived latency = STT + LLM + TTS (what users actually experience)"

    @classmethod
    def from_components(
        cls,
        stt_ms: float,
        llm_ms: float,
        tts_ms: float
    ) -> "LatencyMetrics":
        """Create from component timings"""
        total = stt_ms + llm_ms + tts_ms
        return cls(
            stt_time_ms=round(stt_ms, 2),
            llm_time_ms=round(llm_ms, 2),
            tts_time_ms=round(tts_ms, 2),
            total_time_ms=round(total, 2),
            perceived_latency_ms=round(total, 2)
        )


class InternalTalkResponse(BaseModel):
    """System's internal reasoning and thought process"""
    content: str = Field(..., description="Internal thought process")
    generation_time_ms: float = Field(..., description="Time to generate internal talk")

    class Config:
        description = "🧠 System's reasoning about conversation state and next actions"


class HintsForNextTurnResponse(BaseModel):
    """Adaptive instructions for next conversation turn"""
    hints: List[str] = Field(..., description="List of adaptive instructions")
    count: int = Field(..., description="Number of hints")
    generation_time_ms: float = Field(..., description="Time to generate hints")

    class Config:
        description = "💡 Adaptive instructions for next conversation turn"

    @classmethod
    def from_hints(cls, hints: List[str], generation_time_ms: float) -> "HintsForNextTurnResponse":
        """Create from hints list"""
        return cls(
            hints=hints,
            count=len(hints),
            generation_time_ms=round(generation_time_ms, 2)
        )


class UserError(BaseModel):
    """User error detected in conversation"""
    error_type: str = Field(..., description="Type of error (e.g., 'address_incomplete')")
    severity: ErrorSeverity = Field(..., description="Severity level")
    description: str = Field(..., description="Human-readable description")
    suggestion: Optional[str] = Field(None, description="Suggestion to fix the error")

    class Config:
        description = "User error with severity and suggestion"


class ErrorDetectionResponse(BaseModel):
    """Collection of errors detected"""
    errors: List[UserError] = Field(..., description="List of detected errors")
    count: int = Field(..., description="Total number of errors")
    detection_time_ms: float = Field(..., description="Time to detect errors")

    class Config:
        description = "⚠️  Errors detected with suggestions for improvement"

    @classmethod
    def from_errors(
        cls,
        errors: List[Dict[str, Any]],
        detection_time_ms: float
    ) -> "ErrorDetectionResponse":
        """Create from error list"""
        user_errors = []
        for error in errors:
            user_errors.append(
                UserError(
                    error_type=error.get("error_type", "unknown"),
                    severity=ErrorSeverity(error.get("severity", "low")),
                    description=error.get("description", ""),
                    suggestion=error.get("suggestion")
                )
            )

        return cls(
            errors=user_errors,
            count=len(user_errors),
            detection_time_ms=round(detection_time_ms, 2)
        )


class ResponseSegmentsMetrics(BaseModel):
    """Metrics for response segments generated in parallel"""
    max_parallel_generation_time_ms: float = Field(
        ...,
        description="Maximum generation time across all components (parallel execution)"
    )

    class Config:
        description = "Performance metrics for parallel component generation"


class ComponentOutputsResponse(BaseModel):
    """All response components generated in parallel"""
    internal_talk: Optional[InternalTalkResponse] = Field(None, description="System's reasoning")
    hints_for_next_turn: Optional[HintsForNextTurnResponse] = Field(None, description="Adaptive hints")
    error_detection: Optional[ErrorDetectionResponse] = Field(None, description="Detected errors")
    metrics: ResponseSegmentsMetrics = Field(..., description="Component generation metrics")

    class Config:
        description = "Response components generated in parallel without adding latency"


class PerceivedLatencyClassification(str, Enum):
    """Performance classification based on perceived latency"""
    EXCELLENT = "EXCELLENT"  # < 500ms
    GOOD = "GOOD"  # < 800ms
    ACCEPTABLE = "ACCEPTABLE"  # < 1200ms
    SLOW = "SLOW"  # > 1200ms


class StreamingConversationResponse(BaseModel):
    """
    Enhanced streaming response with perceived latency metrics

    Format: Used in NDJSON streaming for WebRTC and HTTP streaming.
    Each turn includes complete latency breakdown and component outputs.
    """
    turn_number: int = Field(..., description="Conversation turn number")
    timestamp: str = Field(..., description="ISO format timestamp")

    # Core response
    user_text: str = Field(..., description="User's input text")
    llm_response: str = Field(..., description="AI's response text")

    # Perceived latency metrics (STT + LLM + TTS)
    latencies: LatencyMetrics = Field(..., description="Breakdown of perceived latency")

    # Component outputs (generated in parallel)
    components: Optional[ComponentOutputsResponse] = Field(None, description="Response components")

    # Classification
    classification: PerceivedLatencyClassification = Field(
        ...,
        description="Performance classification (EXCELLENT/GOOD/ACCEPTABLE/SLOW)"
    )

    class Config:
        description = "Complete streaming response with perceived latency metrics"
        json_schema_extra = {
            "example": {
                "turn_number": 1,
                "timestamp": "2025-10-26T10:30:00Z",
                "user_text": "Preciso de um táxi para o aeroporto",
                "llm_response": "Claro! Vou chamar um táxi...",
                "latencies": {
                    "stt_time_ms": 180,
                    "llm_time_ms": 320,
                    "tts_time_ms": 220,
                    "total_time_ms": 720,
                    "perceived_latency_ms": 720
                },
                "components": {
                    "internal_talk": {
                        "content": "User wants taxi...",
                        "generation_time_ms": 45
                    },
                    "hints_for_next_turn": {
                        "hints": ["Ask for location", "..."],
                        "count": 3,
                        "generation_time_ms": 32
                    },
                    "error_detection": {
                        "errors": [],
                        "count": 0,
                        "detection_time_ms": 15
                    },
                    "metrics": {
                        "max_parallel_generation_time_ms": 78
                    }
                },
                "classification": "GOOD"
            }
        }


class StreamingSessionSummary(BaseModel):
    """Summary of perceived latency across a session"""
    session_id: str = Field(..., description="Session identifier")
    total_turns: int = Field(..., description="Total conversation turns")
    average_perceived_latency_ms: float = Field(..., description="Average perceived latency")
    min_perceived_latency_ms: float = Field(..., description="Minimum perceived latency")
    max_perceived_latency_ms: float = Field(..., description="Maximum perceived latency")

    classifications: Dict[str, int] = Field(
        ...,
        description="Count of each classification (EXCELLENT, GOOD, ACCEPTABLE, SLOW)"
    )

    class Config:
        description = "Session-wide summary of perceived latency metrics"


class StreamingEventType(str, Enum):
    """Types of streaming events"""
    CONVERSATION_RESPONSE = "conversation_response"
    COMPONENT_OUTPUT = "component_output"
    SESSION_SUMMARY = "session_summary"
    LATENCY_METRICS = "latency_metrics"


class StreamingEvent(BaseModel):
    """
    Base streaming event for NDJSON format

    Each event is a complete JSON object on a single line.
    Used for WebRTC and HTTP streaming.
    """
    event_type: StreamingEventType = Field(..., description="Type of streaming event")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = Field(..., description="Event-specific data")

    class Config:
        description = "Streaming event in NDJSON format"


class ConversationTurnWithLatency(BaseModel):
    """
    Complete conversation turn with latency metrics

    Used internally by Orchestrator to track and report performance.
    """
    turn_number: int
    user_text: str
    transcribed_text: str
    llm_response: str

    # Latency metrics
    stt_latency_ms: float
    llm_latency_ms: float
    tts_latency_ms: float
    total_latency_ms: float = Field(default=0, description="STT + LLM + TTS")

    # Audio metrics
    user_audio_bytes: int
    response_audio_bytes: int

    # Quality metrics
    transcription_accuracy: float = Field(default=100.0)
    response_coherence: float = Field(default=100.0)
    context_maintained: bool = Field(default=True)

    # Component outputs
    internal_talk: Optional[str] = None
    hints_next_turn: Optional[List[str]] = None
    errors_detected: Optional[List[Dict[str, Any]]] = None

    class Config:
        description = "Complete conversation turn with all metrics"

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate total latency if not provided
        if self.total_latency_ms == 0:
            self.total_latency_ms = (
                self.stt_latency_ms + self.llm_latency_ms + self.tts_latency_ms
            )
