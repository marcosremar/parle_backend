"""
TTS Providers
All Text-to-Speech providers with unified interface
"""

from .base import BaseTTSProvider
# NOTE: EdgeTTSProvider disabled - edge_tts dependency not used
# from .edge_tts_provider import EdgeTTSProvider

# TODO: Add more providers as needed
# from .azure_tts_provider import AzureTTSProvider
# from .kokoro_provider import KokoroProvider
# from .elevenlabs_provider import ElevenLabsProvider
# from .openai_tts_provider import OpenAITTSProvider

__all__ = [
    'BaseTTSProvider',
    # 'EdgeTTSProvider',  # Disabled
]