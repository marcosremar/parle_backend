"""
TTS Provider Manager
"""

from typing import Dict, Any, List, Optional
from fastapi import HTTPException

from .gtts import GTTSProvider
from .huggingface import HuggingFaceTTSProvider
from .elevenlabs import ElevenLabsTTSProvider


class TTSProviderManager:
    """Manages multiple TTS providers"""

    def __init__(self):
        self.providers = {}
        self.available_providers = []

        # Initialize gTTS provider (default - free, no API key required)
        try:
            gtts_provider = GTTSProvider()
            if gtts_provider.available:
                self.providers["gtts"] = gtts_provider
                self.available_providers.insert(0, "gtts")
        except Exception as e:
            pass

        # Initialize Hugging Face provider
        try:
            hf_provider = HuggingFaceTTSProvider()
            if hf_provider.available:
                self.providers["huggingface"] = hf_provider
                self.available_providers.append("huggingface")
        except Exception as e:
            pass

        # Initialize Eleven Labs provider
        try:
            elevenlabs_provider = ElevenLabsTTSProvider()
            if elevenlabs_provider.available:
                self.providers["elevenlabs"] = elevenlabs_provider
                self.available_providers.append("elevenlabs")
        except Exception as e:
            pass

    def get_provider(self, provider_name: str):
        """Get a specific provider"""
        return self.providers.get(provider_name)

    def get_available_voices(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available voices, optionally filtered by provider"""
        voices = []

        if provider and provider in self.providers:
            voices.extend(self.providers[provider].get_available_voices())
        else:
            for provider_name, provider_instance in self.providers.items():
                provider_voices = provider_instance.get_available_voices()
                voices.extend(provider_voices)

        return voices

    async def synthesize_speech(self, text: str, provider: str = "gtts", voice: str = None, **kwargs) -> Dict[str, Any]:
        """Synthesize speech using the specified provider"""
        if provider not in self.providers:
            raise HTTPException(status_code=400, detail=f"Provider '{provider}' not available")

        provider_instance = self.providers[provider]

        if voice in (None, "None", ""):
            voice = None

        # Validate and normalize voice for Eleven Labs
        if provider == "elevenlabs" and voice:
            known_invalid_voices = ["pf_dora"]
            if voice in known_invalid_voices:
                voice = None
            else:
                try:
                    if hasattr(provider_instance, 'is_valid_voice'):
                        if not provider_instance.is_valid_voice(voice):
                            voice = None
                except Exception as e:
                    voice = None

        # Set default voice based on provider if not specified
        if not voice:
            if provider == "gtts":
                voice = "pt-br"
            elif provider == "elevenlabs":
                voice = "Rachel"
            elif provider == "huggingface":
                voice = "af_heart"
            else:
                voice = "pt-br"

        return await provider_instance.synthesize_speech(text=text, voice=voice, **kwargs)
