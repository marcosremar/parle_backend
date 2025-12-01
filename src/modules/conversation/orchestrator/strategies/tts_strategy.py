"""
TTS Strategy Pattern

HTTP-based TTS processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TTSStrategy(ABC):
    """Abstract base class for TTS processing strategies."""

    @abstractmethod
    async def synthesize(self, text: str, voice_id: str | None) -> bytes | None:
        """
        Synthesize audio from text.

        Args:
            text: Text to synthesize
            voice_id: Optional voice ID

        Returns:
            Audio bytes or None if synthesis fails
        """


class HTTPTTSStrategy(TTSStrategy):
    """HTTP TTS strategy with fallback to external TTS."""

    def __init__(self, tts_client: Any) -> None:
        """
        Initialize HTTP TTS strategy.

        Args:
            tts_client: TTS service client
        """
        self.tts_client = tts_client

    async def synthesize(self, text: str, voice_id: str | None) -> bytes | None:
        """Synthesize audio using HTTP TTS with fallback."""
        try:
            # Try local TTS first
            audio_response = await self.tts_client.synthesize(text=text, voice_id=voice_id)
            logger.info(f"🔊 TTS (local) generated: {len(audio_response)} bytes")
            return audio_response
        except Exception as e:
            logger.warning(f"⚠️ Local TTS failed: {e}")
            logger.info("   Trying external TTS (HuggingFace)...")

            # Fallback to external TTS
            try:
                audio_response = await self.tts_client.synthesize(
                    text=text, voice="af_heart", format="wav"
                )
                logger.info(f"🔊 TTS (external) generated: {len(audio_response)} bytes")
                return audio_response
            except Exception as e2:
                logger.error(f"❌ External TTS also failed: {e2}")
                return None


class TTSStrategyFactory:
    """Factory for creating TTS strategy (always HTTP-based)."""

    @staticmethod
    def create_strategy(
        tts_client: Any,
        **kwargs,  # Accept but ignore legacy parameters (in_process_mode, tts_instance)
    ) -> TTSStrategy:
        """
        Create TTS strategy (always HTTP-based).

        Args:
            tts_client: TTS service client
            **kwargs: Ignored (for backward compatibility)

        Returns:
            HTTPTTSStrategy instance
        """
        return HTTPTTSStrategy(tts_client)
