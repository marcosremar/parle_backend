#!/usr/bin/env python3
"""
TTS Module with Real Voice (using gTTS)
Replaces the BIP-generating SimpleTTS with actual voice synthesis
"""

import base64
import io
import time
import logging
import asyncio
import tempfile
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import gTTS for real voice
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("✅ gTTS disponível - voz real habilitada")
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("⚠️ gTTS não disponível - instale com: pip install gtts")

# For audio processing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("⚠️ NumPy não disponível")


class VoiceTTSModule:
    """
    TTS Module that generates real voice using gTTS
    Falls back to simple tones only if gTTS is not available
    """

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.language = config.get("language", "pt-br")  # Portuguese Brazil
        self.slow = config.get("slow", False)
        self.sample_rate = config.get("sample_rate", 16000)
        self.is_initialized = False

        # Stats
        self.stats = {
            "total_generations": 0,
            "total_characters": 0,
            "avg_generation_time": 0,
            "use_gtts": GTTS_AVAILABLE
        }

        logger.info(f"✅ Voice TTS criado (gTTS: {GTTS_AVAILABLE})")

    async def initialize(self):
        """Initialize TTS module"""
        if self.is_initialized:
            return

        logger.info("🔊 Inicializando Voice TTS...")

        # Test generation
        if GTTS_AVAILABLE:
            try:
                test_text = "Teste"
                tts = gTTS(text=test_text, lang=self.language, slow=self.slow)
                logger.info(f"✅ gTTS inicializado com idioma: {self.language}")
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar gTTS: {e}")
                self.stats["use_gtts"] = False

        self.is_initialized = True
        await asyncio.sleep(0.05)
        logger.info("✅ Voice TTS inicializado!")

    async def generate_audio_bytes(self, text: str) -> bytes:
        """
        Generate speech audio as raw bytes (for direct use, better performance)

        Args:
            text: Text to convert to speech

        Returns:
            Raw WAV audio bytes
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            if GTTS_AVAILABLE and self.stats["use_gtts"]:
                # Generate with gTTS and return raw bytes
                import tempfile
                import os

                # Create TTS object
                tts = gTTS(text=text, lang=self.language, slow=self.slow)

                # Generate to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                    tmp_mp3_path = tmp_mp3.name
                    tts.save(tmp_mp3_path)

                # Convert MP3 to WAV bytes
                try:
                    import pydub
                    audio = pydub.AudioSegment.from_mp3(tmp_mp3_path)
                    audio = audio.set_frame_rate(self.sample_rate).set_channels(1)

                    # Export to bytes
                    wav_buffer = io.BytesIO()
                    audio.export(wav_buffer, format="wav")
                    audio_bytes = wav_buffer.getvalue()

                    os.unlink(tmp_mp3_path)
                    return audio_bytes

                except ImportError:
                    # Fallback if pydub not available
                    os.unlink(tmp_mp3_path)
                    return b''  # Empty bytes

            else:
                # Simple tone fallback as WAV bytes
                import wave
                duration = len(text) * 0.1  # 100ms per character
                sample_rate = self.sample_rate
                frames = int(duration * sample_rate)

                # Generate simple tone
                import math
                audio_samples = []
                for i in range(frames):
                    # Simple sine wave
                    sample = 0.1 * math.sin(2 * math.pi * 440 * i / sample_rate)
                    audio_samples.append(sample)

                # Convert to WAV bytes
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)

                    # Convert to int16
                    audio = np.array(audio_samples)
                    audio = np.clip(audio, -1, 1)
                    audio = (audio * 32767).astype(np.int16)
                    wav_file.writeframes(audio.tobytes())

                return wav_buffer.getvalue()

        except Exception as e:
            logger.error(f"Erro ao gerar áudio bytes: {e}")
            return b''

    async def generate_speech(self, text: str, voice: Optional[str] = None,
                             style: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate speech from text using real voice

        Args:
            text: Text to convert to speech
            voice: Voice to use (ignored for gTTS)
            style: Style to apply (ignored for gTTS)

        Returns:
            Dict with audio data and metadata
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()
        self.stats["total_generations"] += 1
        self.stats["total_characters"] += len(text)

        try:
            if GTTS_AVAILABLE and self.stats["use_gtts"]:
                # Generate real voice with gTTS
                audio_data = await self._generate_gtts_voice(text)
            else:
                # Fallback to simple tones (avoid BIP sound)
                audio_data = await self._generate_simple_voice(text)

            generation_time = (time.time() - start_time) * 1000

            # Update average
            if self.stats["avg_generation_time"] == 0:
                self.stats["avg_generation_time"] = generation_time
            else:
                self.stats["avg_generation_time"] = (
                    self.stats["avg_generation_time"] * 0.9 + generation_time * 0.1
                )

            logger.info(f"🔊 TTS gerado em {generation_time:.1f}ms ({len(text)} caracteres)")

            return {
                "audio": audio_data,
                "sample_rate": self.sample_rate,
                "format": "pcm16",
                "duration_ms": generation_time,
                "method": "gtts" if GTTS_AVAILABLE and self.stats["use_gtts"] else "simple"
            }

        except Exception as e:
            logger.error(f"❌ Erro ao gerar TTS: {e}")
            # Return silence on error
            silence = np.zeros(int(self.sample_rate * 0.5), dtype=np.int16)
            return {
                "audio": base64.b64encode(silence.tobytes()).decode('utf-8'),
                "sample_rate": self.sample_rate,
                "format": "pcm16",
                "duration_ms": 500,
                "method": "silence"
            }

    async def _generate_gtts_voice(self, text: str) -> str:
        """Generate real voice using gTTS"""
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=self.language, slow=self.slow)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                tts.save(tmp_mp3.name)
                tmp_mp3_path = tmp_mp3.name

            # Convert MP3 to PCM using ffmpeg
            tmp_wav_path = tmp_mp3_path.replace('.mp3', '.wav')

            # Use ffmpeg to convert (should be available in most systems)
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', tmp_mp3_path,
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                tmp_wav_path,
                '-y'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception("FFmpeg conversion failed")

            # Read the WAV file
            with open(tmp_wav_path, 'rb') as f:
                # Skip WAV header (44 bytes)
                f.seek(44)
                audio_data = f.read()

            # Clean up temporary files
            os.unlink(tmp_mp3_path)
            os.unlink(tmp_wav_path)

            # Convert to base64
            return base64.b64encode(audio_data).decode('utf-8')

        except Exception as e:
            logger.error(f"❌ Erro no gTTS: {e}")
            # Fallback to simple voice
            return await self._generate_simple_voice(text)

    async def _generate_simple_voice(self, text: str) -> str:
        """
        Generate simple voice (better than BIP)
        Uses varying frequencies to simulate speech rhythm
        """
        if not NUMPY_AVAILABLE:
            # Return silence if numpy not available
            return base64.b64encode(b'\x00' * self.sample_rate).decode('utf-8')

        # Create more natural sound than pure tones
        duration_per_char = 0.05  # 50ms per character
        audio_samples = []

        for i, char in enumerate(text.lower()):
            char_duration = duration_per_char
            t = np.linspace(0, char_duration, int(self.sample_rate * char_duration))

            if char == ' ':
                # Silence for spaces
                samples = np.zeros_like(t)
            elif char in 'aeiou':
                # Vowels - lower frequencies with harmonics
                freq = 200 + (ord(char) - ord('a')) * 50
                samples = np.sin(2 * np.pi * freq * t) * 0.3
                samples += np.sin(2 * np.pi * freq * 2 * t) * 0.1  # Harmonic
            else:
                # Consonants - noise-like
                samples = np.random.randn(len(t)) * 0.1
                # Add some frequency modulation
                freq = 300 + (ord(char) % 10) * 30
                samples *= np.sin(2 * np.pi * freq * t)

            # Apply envelope for more natural sound
            envelope = np.exp(-t * 10)  # Fast attack, slow decay
            samples *= envelope

            audio_samples.extend(samples)

        # Convert to int16
        audio = np.array(audio_samples)
        audio = np.clip(audio, -1, 1)
        audio = (audio * 32767).astype(np.int16)

        return base64.b64encode(audio.tobytes()).decode('utf-8')

    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics"""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "type": "VoiceTTS",
            "language": self.language
        }

    def cleanup(self):
        """Clean up resources"""
        self.is_initialized = False
        logger.info("🧹 Voice TTS limpo")


# Convenience function
def create_voice_tts(config=None):
    """Create Voice TTS module"""
    return VoiceTTSModule(config)


if __name__ == "__main__":
    # Test
    import asyncio

    async def test():
        tts = VoiceTTSModule()
        await tts.initialize()

        # Test Portuguese text
        result = await tts.generate_speech("Olá! Como posso ajudar você hoje?")
        print(f"Generated audio: {len(result['audio'])} bytes (base64)")
        print(f"Method used: {result['method']}")
        print(f"Stats: {tts.get_stats()}")

    asyncio.run(test())