"""
TTS Providers
"""

from .gtts import GTTSProvider
from .elevenlabs import ElevenLabsTTSProvider
from .huggingface import HuggingFaceTTSProvider
from .manager import TTSProviderManager

__all__ = [
    "GTTSProvider",
    "ElevenLabsTTSProvider",
    "HuggingFaceTTSProvider",
    "TTSProviderManager",
]
