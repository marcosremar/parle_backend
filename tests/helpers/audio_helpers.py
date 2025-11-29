"""
Helper functions for audio testing
"""

import base64
import wave
import io
import numpy as np
from typing import Optional


def create_silence_audio(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Create silence audio in PCM int16 format
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio bytes (PCM int16)
    """
    num_samples = int(sample_rate * duration)
    samples = np.zeros(num_samples, dtype=np.int16)
    return samples.tobytes()


def create_wav_file(audio_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """
    Create WAV file from audio bytes
    
    Args:
        audio_bytes: PCM audio data
        sample_rate: Sample rate in Hz
        channels: Number of channels (1 = mono, 2 = stereo)
        
    Returns:
        WAV file bytes
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    
    return wav_buffer.getvalue()


def audio_to_base64(audio_bytes: bytes) -> str:
    """Convert audio bytes to base64 string"""
    return base64.b64encode(audio_bytes).decode('utf-8')


def base64_to_audio(audio_base64: str) -> bytes:
    """Convert base64 string to audio bytes"""
    return base64.b64decode(audio_base64)


def create_test_audio_base64(
    duration: float = 1.0,
    sample_rate: int = 16000,
    format: str = "wav"
) -> str:
    """
    Create test audio in base64 format
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        format: Audio format ("wav" or "pcm")
        
    Returns:
        Base64 encoded audio string
    """
    audio_bytes = create_silence_audio(duration, sample_rate)
    
    if format == "wav":
        audio_bytes = create_wav_file(audio_bytes, sample_rate)
    
    return audio_to_base64(audio_bytes)


def validate_audio_format(audio_bytes: bytes) -> bool:
    """
    Validate if audio bytes are in valid WAV format
    
    Args:
        audio_bytes: Audio data to validate
        
    Returns:
        True if valid WAV format
    """
    try:
        wav_buffer = io.BytesIO(audio_bytes)
        with wave.open(wav_buffer, 'rb') as wav_file:
            # Try to read some frames
            wav_file.readframes(1)
        return True
    except Exception:
        return False


def get_audio_duration(audio_bytes: bytes, sample_rate: int = 16000) -> float:
    """
    Get audio duration in seconds
    
    Args:
        audio_bytes: Audio data (PCM int16)
        sample_rate: Sample rate in Hz
        
    Returns:
        Duration in seconds
    """
    num_samples = len(audio_bytes) // 2  # 16-bit = 2 bytes per sample
    return num_samples / sample_rate
