"""
Constants for Orchestrator service

All magic numbers and strings are extracted here for maintainability.
Values are loaded from config/settings.yaml with fallback to defaults.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache
def _load_orchestrator_config() -> dict[str, Any]:
    """
    Load orchestrator configuration from settings.yaml

    Returns:
        Dict with orchestrator configuration or empty dict if not found
    """
    if not CONFIG_YAML_PATH.exists():
        return {}

    try:
        with open(CONFIG_YAML_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        return config.get("orchestrator", {})
    except Exception:
        return {}


def _get_config_value(key_path: str, default: Any) -> Any:
    """
    Get configuration value from nested dict using dot notation

    Args:
        key_path: Dot-separated path (e.g., "cache.valid_skills_ttl_seconds")
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    config = _load_orchestrator_config()
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value if value is not None else default


# Cache Configuration
VALID_SKILLS_CACHE_TTL_SECONDS: Final[float] = _get_config_value(
    "cache.valid_skills_ttl_seconds", 300.0
)  # 5 minutes

# Audio Configuration
DEFAULT_SAMPLE_RATE: Final[int] = _get_config_value("audio.default_sample_rate", 16000)  # Hz
AUDIO_INT16_MAX: Final[int] = _get_config_value("audio.int16_max", 32767)
AUDIO_INT16_MIN: Final[int] = _get_config_value("audio.int16_min", -32768)
AUDIO_NORMALIZATION_DIVISOR: Final[float] = _get_config_value(
    "audio.normalization_divisor", 32768.0
)
MINIMUM_AUDIO_DURATION_MS: Final[int] = _get_config_value(
    "audio.minimum_duration_ms", 40
)  # milliseconds
MINIMUM_AUDIO_SAMPLES: Final[int] = _get_config_value(
    "audio.minimum_samples", 640
)  # samples @ 16kHz
MAXIMUM_AUDIO_SIZE_MB: Final[int] = _get_config_value("audio.maximum_size_mb", 50)  # MB

# Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD: Final[float] = _get_config_value("confidence.high_threshold", 0.7)
DEFAULT_MASTERY_PROBABILITY: Final[float] = _get_config_value(
    "confidence.default_mastery_probability", 0.0
)

# Default Values
DEFAULT_TEMPERATURE: Final[float] = _get_config_value("defaults.temperature", 0.7)
DEFAULT_MAX_TOKENS: Final[int] = _get_config_value("defaults.max_tokens", 100)
DEFAULT_VOICE_SPEED: Final[float] = _get_config_value("defaults.voice_speed", 1.0)

# Service URLs (defaults) - Only for external services
# Note: Module services (stt, tts, llm, session, scenarios) use direct calls, no URLs needed
DEFAULT_EXTERNAL_ULTRAVOX_URL: Final[str] = _get_config_value(
    "service_urls.external_ultravox", "http://localhost:8112"
)
DEFAULT_CONVERSATION_STORE_URL: Final[str] = _get_config_value(
    "service_urls.conversation_store", "http://localhost:8800"
)
DEFAULT_CONVERSATION_HISTORY_URL: Final[str] = _get_config_value(
    "service_urls.conversation_history", "http://localhost:8501"
)

# Environment Variable Names
# Note: Module services (llm, tts, stt, session, scenarios) use direct calls, no env vars needed
# Only external services need environment variables
ENV_ORCHESTRATOR_EXTERNAL_ULTRAVOX_URL: Final[str] = "ORCHESTRATOR_EXTERNAL_ULTRAVOX_URL"
ENV_CONVERSATION_STORE_URL: Final[str] = "CONVERSATION_STORE_URL"  # May be external
ENV_CONVERSATION_HISTORY_URL: Final[str] = "CONVERSATION_HISTORY_URL"  # May be external
ENV_ORCHESTRATOR_SKIP_HEALTH_CHECKS: Final[str] = "ORCHESTRATOR_SKIP_HEALTH_CHECKS"

# Client Configuration Defaults
DEFAULT_MAX_RETRIES: Final[int] = _get_config_value("client.max_retries", 3)
DEFAULT_BASE_BACKOFF: Final[float] = _get_config_value("client.base_backoff", 1.0)
DEFAULT_TIMEOUT: Final[float] = _get_config_value("client.timeout", 30.0)
DEFAULT_HEALTH_CHECK_TIMEOUT: Final[float] = _get_config_value("client.health_check_timeout", 2.0)
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = _get_config_value(
    "client.circuit_breaker.failure_threshold", 3
)
DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[float] = _get_config_value(
    "client.circuit_breaker.recovery_timeout", 30.0
)


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
_DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant. Your task is to:

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

_DEFAULT_TEXT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant. Your task is to:

1. LISTEN CAREFULLY to the user's message and identify the specific question or topic
2. ANSWER directly and accurately
3. Provide a concise, helpful response
4. DO NOT ask unrelated questions
5. Focus on being helpful and accurate

Respond naturally in a conversational tone."""

DEFAULT_SYSTEM_PROMPT: Final[str] = _get_config_value(
    "prompts.default_system_prompt", _DEFAULT_SYSTEM_PROMPT_TEMPLATE
)
DEFAULT_TEXT_SYSTEM_PROMPT: Final[str] = _get_config_value(
    "prompts.default_text_system_prompt", _DEFAULT_TEXT_SYSTEM_PROMPT_TEMPLATE
)


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


# Heuristic Analysis Constants
HEURISTIC_SHORT_RESPONSE_THRESHOLD: Final[int] = _get_config_value(
    "heuristics.short_response_threshold", 50
)  # Characters
HEURISTIC_LONG_RESPONSE_THRESHOLD: Final[int] = _get_config_value(
    "heuristics.long_response_threshold", 100
)  # Characters
HEURISTIC_SHORT_LLM_OUTPUT_THRESHOLD: Final[int] = _get_config_value(
    "heuristics.short_llm_output_threshold", 20
)  # Characters
HEURISTIC_CONFUSION_DETECTION_THRESHOLD: Final[int] = _get_config_value(
    "heuristics.confusion_detection_threshold", 50
)  # Characters
HEURISTIC_ANALYSIS_CONFIDENCE: Final[float] = _get_config_value(
    "heuristics.analysis_confidence", 0.85
)
HEURISTIC_FALLBACK_CONFIDENCE: Final[float] = _get_config_value(
    "heuristics.fallback_confidence", 0.5
)
HEURISTIC_DEFAULT_ESTIMATED_TURNS: Final[int] = _get_config_value(
    "heuristics.default_estimated_turns", 3
)
HEURISTIC_SHORT_RESPONSE_ESTIMATED_TURNS: Final[int] = _get_config_value(
    "heuristics.short_response_estimated_turns", 2
)
HEURISTIC_COMPLEX_QUESTION_ESTIMATED_TURNS: Final[int] = _get_config_value(
    "heuristics.complex_question_estimated_turns", 4
)
HEURISTIC_PROMPT_PREFIX_TRUNCATE_LENGTH: Final[int] = _get_config_value(
    "heuristics.prompt_prefix_truncate_length", 50
)  # Characters
