"""
Constants for Orchestrator service

All magic numbers and strings are extracted here for maintainability.
"""

from enum import Enum
from typing import Final

# Cache Configuration
VALID_SKILLS_CACHE_TTL_SECONDS: Final[float] = 300.0  # 5 minutes

# Audio Configuration
DEFAULT_SAMPLE_RATE: Final[int] = 16000  # Hz
AUDIO_INT16_MAX: Final[int] = 32767
AUDIO_INT16_MIN: Final[int] = -32768
AUDIO_NORMALIZATION_DIVISOR: Final[float] = 32768.0
MINIMUM_AUDIO_DURATION_MS: Final[int] = 40  # milliseconds
MINIMUM_AUDIO_SAMPLES: Final[int] = 640  # samples @ 16kHz
MAXIMUM_AUDIO_SIZE_MB: Final[int] = 50  # MB

# Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.7
DEFAULT_MASTERY_PROBABILITY: Final[float] = 0.0

# Default Values
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_MAX_TOKENS: Final[int] = 100
DEFAULT_VOICE_SPEED: Final[float] = 1.0

# Service URLs (defaults) - Only for external services
# Note: Module services (stt, tts, llm, session, scenarios) use direct calls, no URLs needed
DEFAULT_EXTERNAL_ULTRAVOX_URL: Final[str] = "http://localhost:8112"  # External service
DEFAULT_CONVERSATION_STORE_URL: Final[str] = "http://localhost:8800"  # May be external
DEFAULT_CONVERSATION_HISTORY_URL: Final[str] = "http://localhost:8501"  # May be external

# Environment Variable Names
# Note: Module services (llm, tts, stt, session, scenarios) use direct calls, no env vars needed
# Only external services need environment variables
ENV_ORCHESTRATOR_EXTERNAL_ULTRAVOX_URL: Final[str] = "ORCHESTRATOR_EXTERNAL_ULTRAVOX_URL"
ENV_CONVERSATION_STORE_URL: Final[str] = "CONVERSATION_STORE_URL"  # May be external
ENV_CONVERSATION_HISTORY_URL: Final[str] = "CONVERSATION_HISTORY_URL"  # May be external
ENV_ORCHESTRATOR_SKIP_HEALTH_CHECKS: Final[str] = "ORCHESTRATOR_SKIP_HEALTH_CHECKS"

# Context Types
class ContextType(str, Enum):
    """Context types for skill filtering"""
    PRODUCTION = "production"
    COMPREHENSION = "comprehension"
    INTERACTION = "interaction"

# Service Names
class ServiceName(str, Enum):
    """Service identifiers"""
    LLM = "llm"
    TTS = "tts"
    STT = "stt"
    SESSION = "session"
    SCENARIOS = "scenarios"
    CONVERSATION_STORE = "conversation_store"
    CONVERSATION_HISTORY = "conversation_history"
    EXTERNAL_ULTRAVOX = "external_ultravox"
    STUDENT_MODEL = "student_model"
    LEARNING_PATH = "learning_path"
    SPEECH_GRADER = "speech_grader"
    PEDAGOGICAL_POLICY = "pedagogical_policy"

# LLM Providers
class LLMProvider(str, Enum):
    """LLM provider identifiers"""
    IN_PROCESS = "in_process"
    PRIMARY = "primary"
    FALLBACK = "fallback"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

# Default System Prompts
DEFAULT_SYSTEM_PROMPT: Final[str] = """You are a helpful AI assistant. Your task is to:

1. LISTEN CAREFULLY to the audio and identify the specific question being asked
2. ANSWER ONLY that specific question directly and accurately
3. Provide a concise, factual answer (1-2 sentences maximum)
4. DO NOT ask questions back to the user
5. DO NOT change the topic or discuss unrelated things
6. Focus on accuracy and relevance

Example:
Audio: "Qual é a capital da França?"
Correct response: "A capital da França é Paris."
Incorrect: "Qual é o teu nome?" (asking a different question)
Incorrect: "Paris é uma bela cidade. Você gostaria de saber mais?" (asking follow-up questions)

Listen to the audio, identify the question, and answer it directly."""

DEFAULT_TEXT_SYSTEM_PROMPT: Final[str] = """You are a helpful AI assistant. Your task is to:

1. LISTEN CAREFULLY to the user's message and identify the specific question or topic
2. ANSWER directly and accurately
3. Provide a concise, helpful response
4. DO NOT ask unrelated questions
5. Focus on being helpful and accurate

Respond naturally in a conversational tone."""

# Stats Keys
class StatsKey(str, Enum):
    """Statistics counter keys"""
    TOTAL_TURNS = "total_turns"
    SUCCESSFUL_TURNS = "successful_turns"
    FAILED_TURNS = "failed_turns"
    PRIMARY_LLM_COUNT = "primary_llm_count"
    FALLBACK_LLM_COUNT = "fallback_llm_count"
    IN_PROCESS_COUNT = "in_process_count"
    HTTP_FALLBACK_COUNT = "http_fallback_count"
    TOTAL_PROCESSING_TIME = "total_processing_time"
