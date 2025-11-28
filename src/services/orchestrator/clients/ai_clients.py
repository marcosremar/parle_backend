"""
AI Service Clients

Clients for LLM, TTS, and STT services.
"""

from __future__ import annotations

import base64
import logging
from typing import Dict, Any, Optional
import numpy as np

from .base import BaseServiceClient, ServiceClientError, Priority

logger = logging.getLogger(__name__)


class LLMClient(BaseServiceClient):
    """Ultravox LLM service client (Primary LLM)"""

    def __init__(self) -> None:
        super().__init__("llm")

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        max_tokens: int = 512,
        voice_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        priority: Priority = Priority.NORMAL
    ) -> Dict[str, Any]:
        """
        Process audio through Ultravox (integrated STT + LLM).

        Args:
            audio_data: Audio bytes (PCM or WAV)
            sample_rate: Sample rate in Hz
            max_tokens: Maximum tokens to generate
            voice_id: Voice ID for language detection
            system_prompt: Optional system prompt/context for the LLM
            priority: Request priority

        Returns:
            Dict with 'text' (response) and 'transcript' (optional)
        """
        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            request_data = {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "max_tokens": max_tokens,
                "voice_id": voice_id
            }
            if system_prompt:
                request_data["system_prompt"] = system_prompt

            result = await self._post("/process_audio", json_data=request_data, timeout=60.0)
            logger.info(f"🤖 LLM responded: {result.get('text', '')[:100]}...")

            return {
                "text": result.get('text', ''),
                "transcript": result.get('transcript', ''),
                "metadata": result.get('metadata', {}),
                "latency_ms": result.get('latency_ms', 0)
            }
        except Exception as e:
            logger.error(f"❌ LLM error: {e}")
            raise ServiceClientError(f"LLM processing failed: {e}")


class TTSClient(BaseServiceClient):
    """TTS service client"""

    def __init__(self) -> None:
        super().__init__("tts")

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 16000,
        format: str = "wav"
    ) -> bytes:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            voice_id: Voice ID
            speed: Speech speed
            sample_rate: Sample rate (8000, 16000, or 24000 Hz)
            format: Audio format (wav, mp3, opus, or ogg)

        Returns:
            Audio bytes in specified format
        """
        try:
            if voice_id:
                valid_elevenlabs_voices = [
                    "Rachel", "Drew", "Clyde", "Paul", "Domi", "Dave", "Fin",
                    "Bella", "Antoni", "Thomas", "Charlie", "Emily", "Elli",
                    "Josh", "Arnold", "Adam", "Sam"
                ]
                if voice_id not in valid_elevenlabs_voices:
                    logger.warning(f"⚠️  Voice '{voice_id}' is not valid, normalizing to None")
                    voice_id = None
                elif not voice_id.strip():
                    voice_id = None
            
            data = {
                "text": text,
                "voice": voice_id,
                "speed": speed,
                "sample_rate": sample_rate,
                "format": format
            }

            audio_data = await self._post("/synthesize", json_data=data, timeout=20.0)
            logger.info(f"🔊 TTS generated: {len(audio_data)} bytes ({sample_rate}Hz, {format})")
            return audio_data
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            raise ServiceClientError(f"TTS synthesis failed: {e}")


class STTClient(BaseServiceClient):
    """STT (Speech-to-Text) service client"""

    def __init__(self) -> None:
        super().__init__("stt")

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_data: Audio bytes
            sample_rate: Sample rate in Hz
            language: Optional language code

        Returns:
            Dict with 'text' (transcribed text)
        """
        try:
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_list = (audio_array / 32768.0).tolist()

            request_data: Dict[str, Any] = {"audio": audio_list}
            if language:
                request_data["language"] = language

            result = await self._post("/audio/transcribe", json_data=request_data, timeout=15.0)
            text = result.get("text", "")
            logger.info(f"📝 STT transcribed: {text[:100]}...")
            return {"text": text}
        except Exception as e:
            logger.error(f"❌ STT error: {e}")
            raise ServiceClientError(f"STT transcription failed: {e}")


class ExternalUltravoxClient(BaseServiceClient):
    """External Ultravox service client (Groq STT + LLM)"""

    def __init__(self) -> None:
        super().__init__("external_ultravox")

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        max_tokens: int = 512,
        voice_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        priority: Priority = Priority.NORMAL
    ) -> Dict[str, Any]:
        """Process audio through External LLM service via HTTP."""
        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            request_data = {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "max_tokens": max_tokens,
                "voice_id": voice_id
            }
            if system_prompt:
                request_data["system_prompt"] = system_prompt

            result = await self._post("/process_audio", json_data=request_data, timeout=60.0)
            logger.info(f"🤖 External LLM responded: {result.get('text', '')[:100]}...")

            return {
                "text": result.get('text', ''),
                "transcript": result.get('transcript', ''),
                "metadata": result.get('metadata', {}),
                "latency_ms": result.get('latency_ms', 0)
            }
        except Exception as e:
            logger.error(f"❌ External LLM error: {e}")
            raise ServiceClientError(f"External LLM processing failed: {e}")


class ExternalLLMClient(BaseServiceClient):
    """External LLM service client (renamed from external_llm)"""

    def __init__(self) -> None:
        super().__init__("llm", is_module_service=True)

    async def call_conversation(
        self,
        user_input: str,
        history: list,
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call LLM with conversation context."""
        try:
            result = await self._post(
                "/api/conversation",
                json_data={
                    "user_input": user_input,
                    "history": history,
                    "session_data": session_data
                },
                timeout=60.0
            )
            return result
        except Exception as e:
            logger.error(f"❌ External LLM conversation error: {e}")
            raise ServiceClientError(f"External LLM conversation failed: {e}")


class ExternalSTTClient(BaseServiceClient):
    """External STT service client (renamed from external_stt)"""

    def __init__(self) -> None:
        super().__init__("stt", is_module_service=True)

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using external STT service."""
        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            result = await self._post(
                "/api/transcribe",
                json_data={
                    "audio_base64": audio_base64,
                    "sample_rate": sample_rate,
                    "language": language or "auto"
                },
                timeout=15.0
            )
            return {"text": result.get("text", "")}
        except Exception as e:
            logger.error(f"❌ External STT error: {e}")
            raise ServiceClientError(f"External STT transcription failed: {e}")


class ExternalTTSClient(BaseServiceClient):
    """External TTS service client (renamed from external_tts)"""

    def __init__(self) -> None:
        super().__init__("tts", is_module_service=True)

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        format: str = "wav"
    ) -> bytes:
        """Synthesize text using external TTS service."""
        try:
            result = await self._post(
                "/api/synthesize",
                json_data={
                    "text": text,
                    "voice": voice or "af_heart",
                    "format": format
                },
                timeout=20.0
            )
            return result
        except Exception as e:
            logger.error(f"❌ External TTS error: {e}")
            raise ServiceClientError(f"External TTS synthesis failed: {e}")
