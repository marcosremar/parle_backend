"""
Strategy Pattern implementations for LLM and TTS processing (HTTP-based).
"""

from .llm_strategy import (
    LLMStrategy,
    HTTPLLMStrategy,
    LLMStrategyFactory,
)
from .tts_strategy import (
    TTSStrategy,
    HTTPTTSStrategy,
    TTSStrategyFactory,
)

__all__ = [
    "LLMStrategy",
    "HTTPLLMStrategy",
    "LLMStrategyFactory",
    "TTSStrategy",
    "HTTPTTSStrategy",
    "TTSStrategyFactory",
]
