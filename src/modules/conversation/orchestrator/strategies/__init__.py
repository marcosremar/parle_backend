"""
Strategy Pattern implementations for LLM and TTS processing modes.
"""

from .llm_strategy import (
    LLMStrategy,
    InProcessLLMStrategy,
    HTTPLLMStrategy,
    LLMStrategyFactory,
)
from .tts_strategy import (
    TTSStrategy,
    InProcessTTSStrategy,
    HTTPTTSStrategy,
    TTSStrategyFactory,
)

__all__ = [
    "LLMStrategy",
    "InProcessLLMStrategy",
    "HTTPLLMStrategy",
    "LLMStrategyFactory",
    "TTSStrategy",
    "InProcessTTSStrategy",
    "HTTPTTSStrategy",
    "TTSStrategyFactory",
]
