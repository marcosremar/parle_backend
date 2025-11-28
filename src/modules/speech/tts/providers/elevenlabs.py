"""
Eleven Labs TTS Provider
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from loguru import logger


class ElevenLabsTTSProvider:
    """Eleven Labs TTS provider"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY') or os.getenv('ELEVEN_LABS_API_KEY')
        self.available = False
        self.client = None
        self._valid_voices = None

        if not self.api_key:
            return

        # Try to import elevenlabs
        try:
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=self.api_key)
            self.available = True
        except ImportError:
            pass
        except Exception as e:
            self.available = False

    # ElevenLabs voice name → voice_id mapping
    VOICE_MAPPING = {
        "Rachel": "21m00Tcm4TlvDq8ikWAM",
        "Drew": "29vD33N1CtxCmqQRPOHJ",
        "Clyde": "2EiwWnXFnvU5JabPnv8n",
        "Paul": "5Q0t7uMcjvnagumLfvZi",
        "Domi": "AZnzlk1XvdvUeBnXmlld",
        "Dave": "CYw3kZ02Hs0563khs1Fj",
        "Fin": "D38z5RcWu1voky8WS1ja",
        "Bella": "EXAVITQu4vr4xnSDxMaL",
        "Antoni": "ErXwobaYiN019PkySvjV",
        "Thomas": "GBv7mTt0atIp3Br8iCZE",
        "Charlie": "IKne3meq5aSn9XLyUdCD",
        "Emily": "LcfcDJNUP1GQjkzn1xUU",
        "Elli": "MF3mGyEYCl7XYWbV9V6O",
        "Josh": "TxGEqnHWrfWFTfGW9XjX",
        "Arnold": "VR6AewLTigWG4xSOukaG",
        "Adam": "pNInz6obpgDQGcFmaJgB",
        "Sam": "yoZ06aMxZJJ28mfd3POQ",
    }

    def _fetch_valid_voices_from_api(self) -> Dict[str, str]:
        """Fetch valid voices from Eleven Labs API"""
        if not self.available or not self.client:
            return self.VOICE_MAPPING
        
        try:
            try:
                voices_response = self.client.voices.get_all()
                voices_list = voices_response.voices if hasattr(voices_response, 'voices') else voices_response
            except AttributeError:
                try:
                    voices_list = self.client.voices.get_all()
                except:
                    voices_list = []
            
            valid_voices = {}
            if isinstance(voices_list, list):
                for voice in voices_list:
                    if hasattr(voice, 'name') and hasattr(voice, 'voice_id'):
                        valid_voices[voice.name] = voice.voice_id
                    elif isinstance(voice, dict):
                        voice_name = voice.get('name')
                        voice_id = voice.get('voice_id')
                        if voice_name and voice_id:
                            valid_voices[voice_name] = voice_id
            
            if valid_voices:
                self._valid_voices = valid_voices
                return valid_voices
            else:
                return self.VOICE_MAPPING
        except Exception as e:
            return self.VOICE_MAPPING

    def get_valid_voices(self) -> Dict[str, str]:
        """Get valid voices (from API cache or fallback)"""
        if self._valid_voices is None:
            self._valid_voices = self._fetch_valid_voices_from_api()
        return self._valid_voices

    def is_valid_voice(self, voice: str) -> bool:
        """Check if voice is valid for Eleven Labs"""
        if not voice:
            return False
        valid_voices = self.get_valid_voices()
        return voice in valid_voices

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available Eleven Labs voices"""
        if not self.available:
            return []

        valid_voices = self.get_valid_voices()
        female_voices = ["Rachel", "Domi", "Bella", "Emily", "Elli", "Lily", "Molly"]
        
        return [
            {
                "id": voice_name,
                "name": voice_name,
                "language": "en",
                "gender": "female" if voice_name in female_voices else "male",
                "description": f"Eleven Labs {voice_name} voice",
                "provider": "elevenlabs"
            }
            for voice_name in valid_voices.keys()
        ]

    async def synthesize_speech(self, text: str, voice: str = "Rachel", model: str = "eleven_turbo_v2_5", **kwargs) -> Dict[str, Any]:
        """Synthesize speech using Eleven Labs"""
        if not self.available:
            raise HTTPException(status_code=503, detail="Eleven Labs provider not available")

        if voice in (None, "None", ""):
            voice = None

        known_invalid_voices = ["pf_dora"]
        if voice in known_invalid_voices:
            voice = None

        if not voice:
            voice = "Rachel"

        valid_voices = self.get_valid_voices()
        if voice not in valid_voices:
            available_voices = ", ".join(sorted(valid_voices.keys()))
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice}' is not valid for Eleven Labs. Available voices: {available_voices}"
            )
        
        voice_id = valid_voices[voice]

        try:
            import time
            import base64
            start_time = time.time()

            model_id = model or "eleven_turbo_v2_5"
            audio_data = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id
            )

            end_time = time.time()

            audio_bytes = b""
            for chunk in audio_data:
                if isinstance(chunk, bytes):
                    audio_bytes += chunk
                else:
                    audio_bytes += chunk
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "audio_data": audio_b64,
                "duration": None,
                "sample_rate": 44100,
                "format": "mp3",
                "voice": voice,
                "model": model or "eleven_turbo_v2_5",
                "provider": "elevenlabs",
                "latency_ms": (end_time - start_time) * 1000
            }

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Eleven Labs synthesis failed: {str(e)}")
