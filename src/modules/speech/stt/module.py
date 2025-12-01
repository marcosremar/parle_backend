"""
STT Module - Direct Python calls for Speech-to-Text
"""

import base64
import tempfile
from typing import Any

from src.modules.base_module import BaseModule

from .providers.groq import GroqTranscriptionProvider


class STTModule(BaseModule):
    """STT Module for direct Python calls"""

    def __init__(self):
        super().__init__("stt")
        self.provider = None
        self.initialized = False

    async def _initialize(self) -> bool:
        """Initialize STT provider"""
        try:
            self.provider = GroqTranscriptionProvider()
            self.initialized = True
            self.logger.info("✅ STT Module initialized with Groq provider")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize STT Module: {e}")
            # Try to initialize anyway (provider might be available later)
            self.initialized = True
            return True

    async def transcribe(
        self,
        audio_base64: str | None = None,
        audio_url: str | None = None,
        language: str = "pt",
        model: str = "whisper-large-v3",
    ) -> dict[str, Any]:
        """
        Transcribe audio to text

        Args:
            audio_base64: Base64 encoded audio data
            audio_url: URL to audio file
            language: Language code (pt, en, etc.)
            model: Whisper model to use

        Returns:
            Dict with text, language, duration, model, provider
        """
        await self.ensure_initialized()

        try:
            # Decode base64 audio if provided
            audio_data = None
            if audio_base64:
                audio_data = base64.b64decode(audio_base64)
            elif audio_url:
                # Download from URL
                from src.core.http_client import HTTPClient

                session = await HTTPClient.get_session()
                async with session.get(audio_url) as resp:
                    audio_data = await resp.read()

            if not audio_data:
                raise ValueError("Either audio_base64 or audio_url must be provided")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            try:
                # Read audio data from temp file (async)
                import asyncio

                audio_bytes = await asyncio.to_thread(lambda: open(tmp_path, "rb").read())

                # Transcribe using provider
                result = await self.provider.transcribe_audio(
                    audio_data=audio_bytes, language=language, model=model
                )

                return {
                    "text": result.get("text", ""),
                    "language": result.get("language", language),
                    "duration": result.get("duration"),
                    "model": model,
                    "provider": "groq",
                }
            finally:
                # Cleanup temp file
                import os

                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except OSError as e:
                    self.logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

        except Exception as e:
            self.logger.error(f"❌ Transcription failed: {e}")
            raise

    async def get_models(self) -> dict[str, Any]:
        """Get available models"""
        return {
            "models": ["whisper-large-v3", "whisper-large-v2", "whisper-medium"],
            "provider": "groq",
        }
