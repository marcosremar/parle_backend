"""
TTS Strategy Pattern

Eliminates if/else chains for in-process vs HTTP TTS processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class TTSStrategy(ABC):
    """Abstract base class for TTS processing strategies."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str]
    ) -> Optional[bytes]:
        """
        Synthesize audio from text.
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID
            
        Returns:
            Audio bytes or None if synthesis fails
        """
        pass


class InProcessTTSStrategy(TTSStrategy):
    """In-process TTS strategy using direct module calls."""

    def __init__(self, tts_instance: Any) -> None:
        """
        Initialize in-process TTS strategy.
        
        Args:
            tts_instance: In-process TTS instance
        """
        self.tts_instance = tts_instance

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str]
    ) -> Optional[bytes]:
        """Synthesize audio using in-process TTS."""
        try:
            logger.info("⚡ Using in-process TTS (ultra-low latency)...")
            audio_response = await self.tts_instance.synthesize(
                text=text,
                voice_id=voice_id
            )
            logger.info(f"✅ In-process TTS generated: {len(audio_response)} bytes")
            return audio_response
        except Exception as e:
            logger.warning(f"⚠️ In-process TTS failed: {e}")
            logger.info("   Falling back to HTTP TTS...")
            return None  # Trigger fallback


class HTTPTTSStrategy(TTSStrategy):
    """HTTP TTS strategy with fallback to external TTS."""

    def __init__(self, tts_client: Any) -> None:
        """
        Initialize HTTP TTS strategy.
        
        Args:
            tts_client: TTS service client
        """
        self.tts_client = tts_client

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str]
    ) -> Optional[bytes]:
        """Synthesize audio using HTTP TTS with fallback."""
        try:
            # Try local TTS first
            audio_response = await self.tts_client.synthesize(
                text=text,
                voice_id=voice_id
            )
            logger.info(f"🔊 TTS (local) generated: {len(audio_response)} bytes")
            return audio_response
        except Exception as e:
            logger.warning(f"⚠️ Local TTS failed: {e}")
            logger.info("   Trying external TTS (HuggingFace)...")
            
            # Fallback to external TTS
            try:
                audio_response = await self.tts_client.synthesize(
                    text=text,
                    voice="af_heart",
                    format="wav"
                )
                logger.info(f"🔊 TTS (external) generated: {len(audio_response)} bytes")
                return audio_response
            except Exception as e2:
                logger.error(f"❌ External TTS also failed: {e2}")
                return None


class TTSStrategyFactory:
    """Factory for creating appropriate TTS strategy."""

    @staticmethod
    def create_strategy(
        in_process_mode: bool,
        tts_instance: Optional[Any],
        tts_client: Any
    ) -> TTSStrategy:
        """
        Create appropriate TTS strategy based on configuration.
        
        Args:
            in_process_mode: Whether in-process mode is enabled
            tts_instance: Optional in-process TTS instance
            tts_client: TTS service client
            
        Returns:
            TTSStrategy instance
        """
        if in_process_mode and tts_instance:
            return InProcessTTSStrategy(tts_instance)
        else:
            return HTTPTTSStrategy(tts_client)
