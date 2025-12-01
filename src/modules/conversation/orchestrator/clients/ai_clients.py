"""
AI Service Clients

Clients for LLM, TTS, and STT services.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from .base import BaseServiceClient, Priority, ServiceClientError

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
        voice_id: str | None = None,
        system_prompt: str | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> dict[str, Any]:
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
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            request_data = {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "max_tokens": max_tokens,
                "voice_id": voice_id,
            }
            if system_prompt:
                request_data["system_prompt"] = system_prompt

            result = await self._post("/process_audio", json_data=request_data, timeout=60.0)
            logger.info(f"🤖 LLM responded: {result.get('text', '')[:100]}...")

            return {
                "text": result.get("text", ""),
                "transcript": result.get("transcript", ""),
                "metadata": result.get("metadata", {}),
                "latency_ms": result.get("latency_ms", 0),
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
        voice_id: str | None = None,
        speed: float = 1.0,
        sample_rate: int = 16000,
        format: str = "wav",
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
                    "Rachel",
                    "Drew",
                    "Clyde",
                    "Paul",
                    "Domi",
                    "Dave",
                    "Fin",
                    "Bella",
                    "Antoni",
                    "Thomas",
                    "Charlie",
                    "Emily",
                    "Elli",
                    "Josh",
                    "Arnold",
                    "Adam",
                    "Sam",
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
                "format": format,
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

    async def transcribe(
        self, audio_data: bytes, sample_rate: int = 16000, language: str | None = None
    ) -> dict[str, Any]:
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

            request_data: dict[str, Any] = {"audio": audio_list}
            if language:
                request_data["language"] = language

            result = await self._post("/audio/transcribe", json_data=request_data, timeout=15.0)
            text = result.get("text", "")
            logger.info(f"📝 STT transcribed: {text[:100]}...")
            return {"text": text}
        except Exception as e:
            logger.error(f"❌ STT error: {e}")
            raise ServiceClientError(f"STT transcription failed: {e}")


class SecondaryLLMClient(BaseServiceClient):
    """Secondary LLM service client (fallback LLM via external API)"""

    def __init__(self) -> None:
        # external_ultravox is an external service, not a module service
        super().__init__("external_ultravox", is_module_service=False)  # Keep service name for backward compatibility

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        max_tokens: int = 512,
        voice_id: str | None = None,
        system_prompt: str | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> dict[str, Any]:
        """Process audio through External LLM service via HTTP."""
        try:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            request_data = {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "max_tokens": max_tokens,
                "voice_id": voice_id,
            }
            if system_prompt:
                request_data["system_prompt"] = system_prompt

            result = await self._post("/process_audio", json_data=request_data, timeout=60.0)
            logger.info(f"🤖 Secondary LLM responded: {result.get('text', '')[:100]}...")

            return {
                "text": result.get("text", ""),
                "transcript": result.get("transcript", ""),
                "metadata": result.get("metadata", {}),
                "latency_ms": result.get("latency_ms", 0),
            }
        except Exception as e:
            logger.error(f"❌ Secondary LLM error: {e}")
            raise ServiceClientError(f"Secondary LLM processing failed: {e}")


# Backward compatibility alias
ExternalUltravoxClient = SecondaryLLMClient


class ExternalLLMClient(BaseServiceClient):
    """External LLM service client (renamed from external_llm)"""

    def __init__(self) -> None:
        super().__init__("llm", is_module_service=True)

    async def call_conversation(
        self, user_input: str, history: list, session_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Call LLM with conversation context."""
        try:
            result = await self._post(
                "/api/conversation",
                json_data={
                    "user_input": user_input,
                    "history": history,
                    "session_data": session_data,
                },
                timeout=60.0,
            )
            return result
        except Exception as e:
            logger.error(f"❌ External LLM conversation error: {e}")
            raise ServiceClientError(f"External LLM conversation failed: {e}")

    async def generate(
        self,
        text: str,
        system_prompt: str | None = None,
        conversation_history: list | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate text using LLM

        Args:
            text: Input text/message
            system_prompt: Optional system prompt
            conversation_history: Optional conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated text as string
        """
        # Validate input
        if not text or not isinstance(text, str) or not text.strip():
            raise ServiceClientError("text must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ServiceClientError("max_tokens must be a positive integer")
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ServiceClientError("temperature must be between 0 and 2")

        # Use direct module call (module services always use direct calls)
        await self._ensure_module_initialized()

        try:
            result = await self.direct_module.generate(
                prompt=text,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )
            # Normalizar resposta para string
            if isinstance(result, dict):
                return result.get(
                    "text", result.get("response", result.get("content", str(result)))
                )
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        except Exception as e:
            self._handle_module_error("generate", e)

    async def _generate_http(
        self,
        text: str,
        system_prompt: str | None = None,
        conversation_history: list | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """HTTP fallback for generate"""
        try:
            data = {"text": text, "max_tokens": max_tokens, "temperature": temperature}
            if system_prompt:
                data["system_prompt"] = system_prompt
            if conversation_history:
                data["conversation_history"] = conversation_history
            data.update(kwargs)

            result = await self._post("/api/generate", json_data=data)
            # Normalizar resposta para string
            if isinstance(result, dict):
                return result.get(
                    "text", result.get("response", result.get("content", str(result)))
                )
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        except ServiceClientError as e:
            logger.error(f"❌ LLM generation failed: {e}")
            raise


class ExternalSTTClient(BaseServiceClient):
    """External STT service client (renamed from external_stt)"""

    def __init__(self) -> None:
        super().__init__("stt", is_module_service=True)

    async def transcribe(
        self, audio_data: bytes, sample_rate: int = 16000, language: str | None = None
    ) -> dict[str, Any]:
        """Transcribe audio using STT module (direct call)."""
        # Validate input
        if not audio_data or not isinstance(audio_data, bytes) or len(audio_data) == 0:
            raise ServiceClientError("audio_data must be non-empty bytes")
        if not isinstance(sample_rate, int) or sample_rate < 1:
            raise ServiceClientError("sample_rate must be a positive integer")

        # Use direct module call (module services always use direct calls)
        await self._ensure_module_initialized()

        try:
            result = await self.direct_module.transcribe(audio_data, sample_rate, language)
            return result if isinstance(result, dict) else {"text": str(result)}
        except Exception as e:
            self._handle_module_error("transcribe", e)

    async def _transcribe_http(
        self, audio_data: bytes, sample_rate: int, language: str | None
    ) -> dict[str, Any]:
        """
        HTTP fallback for transcribe

        ⚠️ DEPRECATED: Not used for module services (which use direct calls).
        Kept for reference/debugging only.
        """
        try:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            result = await self._post(
                "/api/transcribe",
                json_data={
                    "audio_base64": audio_base64,
                    "sample_rate": sample_rate,
                    "language": language or "auto",
                },
                timeout=15.0,
            )
            return {"text": result.get("text", "")}
        except Exception as e:
            logger.error(f"❌ External STT error: {e}")
            raise ServiceClientError(f"External STT transcription failed: {e}")


class ExternalTTSClient(BaseServiceClient):
    """External TTS service client (renamed from external_tts)"""

    def __init__(self) -> None:
        super().__init__("tts", is_module_service=True)

    async def synthesize(self, text: str, voice: str | None = None, format: str = "wav") -> bytes:
        """Synthesize text using TTS module (direct call)."""
        # Validate input
        if not text or not isinstance(text, str) or not text.strip():
            raise ServiceClientError("text must be a non-empty string")
        if format not in ["wav", "mp3", "pcm"]:
            raise ServiceClientError(f"format must be one of: wav, mp3, pcm (got: {format})")

        # Use direct module call (module services always use direct calls)
        await self._ensure_module_initialized()

        try:
            result = await self.direct_module.synthesize(text, voice_id=voice, format=format)
            return result if isinstance(result, bytes) else bytes(result)
        except Exception as e:
            self._handle_module_error("synthesize", e)

    async def _synthesize_http(self, text: str, voice: str | None, format: str) -> bytes:
        """
        HTTP fallback for synthesize

        ⚠️ DEPRECATED: Not used for module services (which use direct calls).
        Kept for reference/debugging only.
        """
        try:
            result = await self._post(
                "/api/synthesize",
                json_data={"text": text, "voice": voice or "af_heart", "format": format},
                timeout=20.0,
            )
            return result
        except Exception as e:
            logger.error(f"❌ External TTS error: {e}")
            raise ServiceClientError(f"External TTS synthesis failed: {e}")
