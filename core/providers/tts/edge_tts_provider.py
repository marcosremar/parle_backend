"""
Edge TTS Provider - Free Text-to-Speech using Microsoft Edge
No API key required!
"""

import edge_tts
import asyncio
import logging
from typing import Optional, List, Dict
import io

from .base import BaseTTSProvider

logger = logging.getLogger(__name__)


class EdgeTTSProvider(BaseTTSProvider):
    """
    Free TTS provider using Microsoft Edge's text-to-speech
    High quality voices, no API key required
    """

    # Popular voices for different languages
    DEFAULT_VOICES = {
        "pt-BR": {
            "female": "pt-BR-FranciscaNeural",
            "male": "pt-BR-AntonioNeural"
        },
        "en-US": {
            "female": "en-US-JennyNeural",
            "male": "en-US-GuyNeural"
        },
        "es-ES": {
            "female": "es-ES-ElviraNeural",
            "male": "es-ES-AlvaroNeural"
        }
    }

    def __init__(self, voice: Optional[str] = None, **kwargs):
        """
        Initialize Edge TTS provider

        Args:
            voice: Default voice to use (e.g., "pt-BR-FranciscaNeural")
        """
        self.default_voice = voice or self.DEFAULT_VOICES["pt-BR"]["female"]
        self.rate = kwargs.get("rate", "+0%")  # Speaking rate
        self.pitch = kwargs.get("pitch", "+0Hz")  # Voice pitch
        self.volume = kwargs.get("volume", "+0%")  # Volume

        logger.info(f"Initialized Edge TTS with voice: {self.default_voice}")

    def get_audio_format(self) -> str:
        """Edge TTS produces MP3 audio"""
        return "mp3"

    async def synthesize(self,
                        text: str,
                        voice: Optional[str] = None,
                        language: str = "pt-BR",
                        speed: float = 1.0,
                        **kwargs) -> bytes:
        """
        Convert text to speech using Edge TTS

        Args:
            text: Text to convert
            voice: Voice to use (optional)
            language: Language code

        Returns:
            Audio bytes (MP3 format)
        """
        try:
            # Use provided voice or default
            voice_to_use = voice or self.default_voice

            # If voice is not specified, try to get default for language
            if not voice and language in self.DEFAULT_VOICES:
                voice_to_use = self.DEFAULT_VOICES[language]["female"]

            # Create communicate instance
            communicate = edge_tts.Communicate(
                text,
                voice_to_use,
                rate=kwargs.get("rate", self.rate),
                pitch=kwargs.get("pitch", self.pitch),
                volume=kwargs.get("volume", self.volume)
            )

            # Generate audio
            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])

            audio_bytes = audio_data.getvalue()
            logger.info(f"Generated {len(audio_bytes)} bytes of audio")

            return audio_bytes

        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise

    def get_available_voices(self, language: str = "pt-BR") -> List[Dict[str, str]]:
        """
        Get list of available voices for a language

        Returns:
            List of voice dictionaries with name, language, and gender
        """
        try:
            # Run async function synchronously
            loop = asyncio.new_event_loop()
            voices = loop.run_until_complete(edge_tts.list_voices())
            loop.close()

            # Filter by language if specified
            if language:
                voices = [v for v in voices if v["Locale"].startswith(language)]

            # Format the response
            return [
                {
                    "name": v["ShortName"],
                    "language": v["Locale"],
                    "gender": v["Gender"],
                    "display_name": v["FriendlyName"]
                }
                for v in voices
            ]

        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            # Return default voices as fallback
            if language in self.DEFAULT_VOICES:
                return [
                    {
                        "name": self.DEFAULT_VOICES[language]["female"],
                        "language": language,
                        "gender": "Female",
                        "display_name": f"Default Female ({language})"
                    },
                    {
                        "name": self.DEFAULT_VOICES[language]["male"],
                        "language": language,
                        "gender": "Male",
                        "display_name": f"Default Male ({language})"
                    }
                ]
            return []

    async def save_to_file(self, text: str, filename: str, voice: Optional[str] = None):
        """
        Helper method to save TTS directly to file

        Args:
            text: Text to convert
            filename: Output filename
            voice: Voice to use
        """
        voice_to_use = voice or self.default_voice

        communicate = edge_tts.Communicate(
            text,
            voice_to_use,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume
        )

        await communicate.save(filename)
        logger.info(f"Saved audio to {filename}")

    @staticmethod
    async def list_all_voices() -> List[Dict[str, str]]:
        """Get all available voices from Edge TTS"""
        voices = await edge_tts.list_voices()
        return voices