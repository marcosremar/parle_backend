"""
Strategy Pattern implementations for LLM and TTS processing (HTTP-based).
"""

from .llm_strategy import (
    HTTPLLMStrategy,
    LLMStrategy,
    LLMStrategyFactory,
)
from .tts_strategy import (
    HTTPTTSStrategy,
    TTSStrategy,
    TTSStrategyFactory,
)

__all__ = [
    "HTTPLLMStrategy",
    "HTTPTTSStrategy",
    "LLMStrategy",
    "LLMStrategyFactory",
    "TTSStrategy",
    "TTSStrategyFactory",
]
