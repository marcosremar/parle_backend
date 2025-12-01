"""
TTS Providers
"""

from .elevenlabs import ElevenLabsTTSProvider
from .gtts import GTTSProvider
from .huggingface import HuggingFaceTTSProvider
from .manager import TTSProviderManager

__all__ = [
    "ElevenLabsTTSProvider",
    "GTTSProvider",
    "HuggingFaceTTSProvider",
    "TTSProviderManager",
]
