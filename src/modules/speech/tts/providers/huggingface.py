"""
Hugging Face TTS Provider
"""

import base64
import io
import os
import time
from typing import Any
import wave

from fastapi import HTTPException
from loguru import logger


class HuggingFaceTTSProvider:
    """Hugging Face Inference API TTS provider"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
        self.available = False

        if not self.api_key:
            return

        # Try to import huggingface_hub
        try:
            from huggingface_hub import InferenceClient

            self.client = InferenceClient(token=self.api_key)
            self.available = True
        except ImportError:
            pass
        except Exception:
            self.available = False

    def _get_voice_config(self, voice: str) -> dict[str, Any]:
        """Get voice configuration - Expanded voice set"""
        voice_configs = {
            # American Female voices
            "af_heart": {
                "voice": "af_heart",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (warm, expressive)",
                "gender": "female",
                "accent": "american",
            },
            "af_nicole": {
                "voice": "af_nicole",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (clear, professional)",
                "gender": "female",
                "accent": "american",
            },
            "af_alloy": {
                "voice": "af_alloy",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (neutral, calm)",
                "gender": "female",
                "accent": "american",
            },
            "af_sarah": {
                "voice": "af_sarah",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (youthful, energetic)",
                "gender": "female",
                "accent": "american",
            },
            "af_kore": {
                "voice": "af_kore",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (soft, gentle)",
                "gender": "female",
                "accent": "american",
            },
            "af_bella": {
                "voice": "af_bella",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Female voice (sweet, melodic)",
                "gender": "female",
                "accent": "american",
            },
            # American Male voices
            "am_adam": {
                "voice": "am_adam",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (deep, authoritative)",
                "gender": "male",
                "accent": "american",
            },
            "am_alex": {
                "voice": "am_alex",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (clear, friendly)",
                "gender": "male",
                "accent": "american",
            },
            "am_michael": {
                "voice": "am_michael",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (confident, professional)",
                "gender": "male",
                "accent": "american",
            },
            "am_fenrir": {
                "voice": "am_fenrir",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (strong, resonant)",
                "gender": "male",
                "accent": "american",
            },
            "am_levi": {
                "voice": "am_levi",
                "lang_code": "a",
                "speed": 1.0,
                "description": "Male voice (warm, approachable)",
                "gender": "male",
                "accent": "american",
            },
            # British voices
            "bf_alice": {
                "voice": "bf_alice",
                "lang_code": "b",
                "speed": 1.0,
                "description": "British female voice (elegant, clear)",
                "gender": "female",
                "accent": "british",
            },
            "bm_george": {
                "voice": "bm_george",
                "lang_code": "b",
                "speed": 1.0,
                "description": "British male voice (refined, articulate)",
                "gender": "male",
                "accent": "british",
            },
        }
        return voice_configs.get(voice, voice_configs["af_heart"])

    async def synthesize_speech(
        self, text: str, voice: str = "af_heart", **kwargs
    ) -> dict[str, Any]:
        """Synthesize speech using Hugging Face"""
        if not self.available:
            raise HTTPException(status_code=503, detail="TTS provider not available")

        self._get_voice_config(voice)

        try:
            start_time = time.time()
            audio_bytes = self.client.text_to_speech(text=text, model="default")
            end_time = time.time()

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Try to get audio info
            try:
                wav_info = wave.open(io.BytesIO(audio_bytes), "rb")
                sample_rate = wav_info.getframerate()
                n_frames = wav_info.getnframes()
                duration = n_frames / sample_rate if sample_rate > 0 else None
                wav_info.close()
            except (OSError, wave.Error, ValueError) as e:
                logger.debug(f"Failed to read audio info: {e}, using defaults")
                sample_rate = 24000
                duration = None

            return {
                "audio_data": audio_b64,
                "duration": duration,
                "sample_rate": sample_rate,
                "format": "wav",
                "voice": voice,
                "model": "default",
                "provider": "huggingface",
                "latency_ms": (end_time - start_time) * 1000,
            }

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e!s}")

    def get_available_voices(self) -> list[dict[str, Any]]:
        """Get available voices"""
        if not self.available:
            return []

        return [
            {
                "id": "af_heart",
                "name": "Heart",
                "language": "en",
                "gender": "female",
                "description": "Female voice (warm, expressive)",
                "provider": "huggingface",
            },
            {
                "id": "af_nicole",
                "name": "Nicole",
                "language": "en",
                "gender": "female",
                "description": "Female voice (clear, professional)",
                "provider": "huggingface",
            },
            {
                "id": "af_alloy",
                "name": "Alloy",
                "language": "en",
                "gender": "female",
                "description": "Female voice (neutral, calm)",
                "provider": "huggingface",
            },
            {
                "id": "am_adam",
                "name": "Adam",
                "language": "en",
                "gender": "male",
                "description": "Male voice (deep, authoritative)",
                "provider": "huggingface",
            },
            {
                "id": "am_alex",
                "name": "Alex",
                "language": "en",
                "gender": "male",
                "description": "Male voice (clear, friendly)",
                "provider": "huggingface",
            },
        ]
