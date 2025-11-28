"""
Google Text-to-Speech (gTTS) Provider
"""

import os
import time
import io
import base64
import subprocess
import tempfile
from typing import Dict, Any, List
from fastapi import HTTPException


class GTTSProvider:
    """Google Text-to-Speech (gTTS) provider - Free and no API key required"""

    def __init__(self):
        self.available = False
        
        # Try to import gTTS
        try:
            from gtts import gTTS
            self.gTTS = gTTS
            self.available = True
        except ImportError as e:
            self.available = False
        except Exception as e:
            self.available = False

    def _get_language_code(self, voice: str) -> str:
        """Map voice to language code for gTTS"""
        language_map = {
            "pt-br": "pt-br", "pt": "pt", "en": "en", "es": "es", "fr": "fr",
            "de": "de", "it": "it", "ja": "ja", "ko": "ko", "zh": "zh",
            "zh-cn": "zh-cn", "ru": "ru", "ar": "ar", "hi": "hi", "nl": "nl",
            "pl": "pl", "tr": "tr", "cs": "cs", "hu": "hu"
        }
        return language_map.get(voice, "pt-br")

    async def synthesize_speech(self, text: str, voice: str = "pt-br", speed: float = 1.0, **kwargs) -> Dict[str, Any]:
        """Synthesize speech using gTTS"""
        if not self.available:
            raise HTTPException(status_code=503, detail="gTTS provider not available")

        try:
            start_time = time.time()
            language = self._get_language_code(voice)
            slow = speed < 0.9

            # Generate with gTTS
            tts = self.gTTS(text=text, lang=language, slow=slow)

            # Save to temporary MP3 file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                tts.write_to_fp(tmp_mp3)
                tmp_mp3_path = tmp_mp3.name

            try:
                # Try to convert MP3 to WAV using ffmpeg if available
                try:
                    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=2)
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                        tmp_wav_path = tmp_wav.name
                    
                    cmd = ['ffmpeg', '-i', tmp_mp3_path, '-ar', '16000', '-ac', '1', '-f', 'wav', '-y', tmp_wav_path]
                    subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                    
                    with open(tmp_wav_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    os.unlink(tmp_wav_path)
                    format_type = "wav"
                    sample_rate = 16000
                    
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    with open(tmp_mp3_path, 'rb') as f:
                        audio_bytes = f.read()
                    format_type = "mp3"
                    sample_rate = 24000

            finally:
                if os.path.exists(tmp_mp3_path):
                    os.unlink(tmp_mp3_path)

            end_time = time.time()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            estimated_duration = len(text) / 10.0

            return {
                "audio_data": audio_b64,
                "duration": estimated_duration,
                "sample_rate": sample_rate,
                "format": format_type,
                "voice": voice,
                "model": "gtts",
                "provider": "gtts",
                "latency_ms": (end_time - start_time) * 1000
            }

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"gTTS synthesis failed: {str(e)}")

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available languages for gTTS"""
        if not self.available:
            return []

        return [
            {"id": "pt-br", "name": "Portuguese (Brazil)", "language": "pt-br", "gender": "neutral", "description": "Portuguese (Brazil)", "provider": "gtts"},
            {"id": "pt", "name": "Portuguese", "language": "pt", "gender": "neutral", "description": "Portuguese", "provider": "gtts"},
            {"id": "en", "name": "English", "language": "en", "gender": "neutral", "description": "English", "provider": "gtts"},
            {"id": "es", "name": "Spanish", "language": "es", "gender": "neutral", "description": "Spanish", "provider": "gtts"},
            {"id": "fr", "name": "French", "language": "fr", "gender": "neutral", "description": "French", "provider": "gtts"},
            {"id": "de", "name": "German", "language": "de", "gender": "neutral", "description": "German", "provider": "gtts"},
            {"id": "it", "name": "Italian", "language": "it", "gender": "neutral", "description": "Italian", "provider": "gtts"},
            {"id": "ja", "name": "Japanese", "language": "ja", "gender": "neutral", "description": "Japanese", "provider": "gtts"},
            {"id": "ko", "name": "Korean", "language": "ko", "gender": "neutral", "description": "Korean", "provider": "gtts"},
            {"id": "zh", "name": "Chinese", "language": "zh", "gender": "neutral", "description": "Chinese", "provider": "gtts"},
            {"id": "ru", "name": "Russian", "language": "ru", "gender": "neutral", "description": "Russian", "provider": "gtts"},
            {"id": "ar", "name": "Arabic", "language": "ar", "gender": "neutral", "description": "Arabic", "provider": "gtts"},
            {"id": "hi", "name": "Hindi", "language": "hi", "gender": "neutral", "description": "Hindi", "provider": "gtts"},
        ]
