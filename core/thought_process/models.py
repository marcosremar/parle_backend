"""
Pydantic models for Thought Process Framework.

Defines the structure for all 7 thought processes in language teaching AI.
(Optimized for speech-only learning: Error Detection and Pronunciation Evaluation removed)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# INDIVIDUAL PROCESS MODELS (7 Processes - optimized for speech-only learning)
# ============================================================================

class GrammarAnalysis(BaseModel):
    """Process 1: Grammar Analysis - Detailed grammatical analysis."""

    sentence_structure: str = Field(..., description="SVO, VSO, OSV, etc.")
    main_verb: str = Field(..., description="tense, mood, aspect")
    noun_phrases: List[str] = Field(default_factory=list)
    complexity_level: str = Field(..., description="beginner, intermediate, advanced")
    correctness_score: float = Field(..., ge=0.0, le=1.0)
    analysis: str = Field(..., description="detailed grammatical breakdown")
    next_step: Optional[str] = Field(default=None, description="grammar focus area for next turn")


class VocabularyAssessment(BaseModel):
    """Process 2: Vocabulary Assessment - Vocabulary level and appropriateness."""

    cefr_level: str = Field(..., description="A1, A2, B1, B2, C1, C2")
    word_count: int
    unknown_words: List[str] = Field(default_factory=list)
    advanced_words: List[str] = Field(default_factory=list)
    vocabulary_consistency: str = Field(..., description="beginner, intermediate, advanced")
    vocabulary_match_with_response: float = Field(..., ge=0.0, le=1.0)
    analysis: str = Field(..., description="vocabulary assessment details")
    next_step: Optional[str] = Field(default=None, description="vocabulary focus area for next turn")


class PedagogicalStrategy(BaseModel):
    """Process 3: Pedagogical Strategy - Teaching approach selection."""

    strategy: str = Field(..., description="recast, explicit, scaffolding, prompting, metalinguistic")
    strategy_description: str
    implementation: str = Field(..., description="how the response implements this strategy")
    learning_objective: str = Field(..., description="what skill is being developed")
    confidence: float = Field(..., ge=0.0, le=1.0)
    analysis: str = Field(..., description="explanation of pedagogical decision")
    next_step: Optional[str] = Field(default=None, description="suggested follow-up action for next turn")


class ConversationFlow(BaseModel):
    """Process 4: Conversation Flow - Natural conversation dynamics."""

    topic_coherence: float = Field(..., ge=0.0, le=1.0)
    topic_change_detected: bool
    current_topic: str
    response_relevance: str = Field(..., description="direct_answer, indirect, expansion, tangential")
    engagement_level: str = Field(..., description="high, medium, low")
    flow_analysis: str = Field(..., description="assessment of conversation naturalness")
    next_step: Optional[str] = Field(default=None, description="suggested conversation direction for next turn")


class LearningProgress(BaseModel):
    """Process 5: Learning Progress - Student development tracking & dynamic level detection."""

    overall_progress: float = Field(..., ge=0.0, le=1.0)
    sessions_completed: int
    skills_mastered: List[str] = Field(default_factory=list)
    skills_developing: List[str] = Field(default_factory=list)
    needs_practice: List[str] = Field(default_factory=list)
    improvement_trend: str = Field(..., description="steady, accelerating, plateauing")
    estimated_next_level: str = Field(..., description="A1, A2, B1, B2, C1, C2")
    detected_current_level: Optional[str] = Field(default=None, description="Student's CEFR level detected THIS TURN (A1, A2, B1, B2)")
    analysis: str = Field(..., description="assessment of student progress")
    next_step: Optional[str] = Field(default=None, description="focus area based on progress analysis")


class CulturalContext(BaseModel):
    """Process 6: Cultural Context - Regional and cultural appropriateness."""

    class RegionalVariation(BaseModel):
        region: str
        difference: str

    cultural_region: str = Field(..., description="Brazil, Portugal, Spain, Mexico, etc.")
    formality_level: str = Field(..., description="formal, informal, neutral")
    cultural_notes: str
    appropriate_register: str
    regional_variations: List[RegionalVariation] = Field(default_factory=list)
    analysis: str = Field(..., description="cultural appropriateness analysis")
    next_step: Optional[str] = Field(default=None, description="cultural awareness focus for next turn")


class LearningRecommendation(BaseModel):
    """Process 7: Learning Recommendation - Personalized learning guidance."""

    class Exercise(BaseModel):
        type: str = Field(..., description="verb_conjugation, vocabulary, listening, etc.")
        focus: str
        duration_minutes: int
        priority: str = Field(..., description="low, medium, high")

    class ContentRecommendation(BaseModel):
        topic: str
        level: str = Field(..., description="CEFR level")
        reason: str

    recommended_practice: str
    practice_intensity: str = Field(..., description="low, medium, high")
    suggested_exercises: List[Exercise] = Field(default_factory=list)
    content_recommendations: List[ContentRecommendation] = Field(default_factory=list)
    analysis: str = Field(..., description="learning recommendations details")
    next_turn_hints: List[str] = Field(default_factory=list, description="hints to guide next student input")


# ============================================================================
# COMPOSITE MODELS
# ============================================================================

class ThoughtProcess(BaseModel):
    """Container for a single thought process execution."""

    id: int = Field(..., description="1-7 (7 thought processes for speech-only learning)")
    name: str = Field(..., description="Process name")
    content: Dict[str, Any] = Field(..., description="Process-specific data")
    generation_time_ms: float = Field(..., description="time to generate this process")


class ThoughtProcessCollection(BaseModel):
    """All 7 thought processes for a single response (optimized for speech-only learning)."""

    processes: List[ThoughtProcess] = Field(..., min_items=7, max_items=7)
    total_processes: int = 7
    total_generation_time_ms: float = Field(..., description="total time for all processes")


class HintsForNextTurn(BaseModel):
    """Hints to guide the next student input (DEPRECATED - now integrated into processes)."""

    hints: List[str] = Field(default_factory=list, description="actionable suggestions")
    count: int = 0
    generation_time_ms: float = 0.0


class ResponseMetadata(BaseModel):
    """Complete response metadata with thought processes.

    Hints are now integrated into each thought process's 'next_step' field.
    This maintains backward compatibility while consolidating all thinking.
    """

    thought_process: ThoughtProcessCollection
    hints_for_next_turn: Optional[HintsForNextTurn] = Field(default=None, description="(DEPRECATED) use process.next_step fields instead")


class LatencyMetrics(BaseModel):
    """Latency timing for STT, LLM, TTS."""

    stt_time_ms: float
    llm_time_ms: float
    tts_time_ms: float
    total_time_ms: float
    perceived_latency_ms: float


class StreamingConversationResponse(BaseModel):
    """Complete response from conversation processing."""

    turn_number: int
    timestamp: str
    user_text: str
    llm_response: str
    response_metadata: ResponseMetadata
    latencies: LatencyMetrics
    classification: str = Field(..., description="EXCELLENT, GOOD, ACCEPTABLE, SLOW")


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class PromptConfig(BaseModel):
    """Configuration for prompt customization."""

    language: str = Field(default="portuguese", description="Target language")
    level: str = Field(default="A2", description="CEFR level: A1, A2, B1, B2, C1, C2")
    region: Optional[str] = Field(default=None, description="Brazil, Portugal, Spain, etc.")
    learning_context: Optional[str] = Field(default=None, description="Conversational, Business, etc.")
    session_number: Optional[int] = Field(default=None, description="Session count for progress tracking")


class ThoughtProcessConfig(BaseModel):
    """Configuration for thought process generation (7 processes for speech-only learning)."""

    enable_grammar_analysis: bool = True
    enable_vocabulary_assessment: bool = True
    enable_pedagogical_strategy: bool = True
    enable_conversation_flow: bool = True
    enable_learning_progress: bool = True
    enable_cultural_context: bool = True
    enable_learning_recommendation: bool = True
    max_processes: int = 7
