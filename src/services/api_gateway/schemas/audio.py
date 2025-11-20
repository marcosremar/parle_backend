"""
Pydantic models for audio-related requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class TTSEngine(str, Enum):
    """Available TTS engines"""
    GTTS = "gtts"


class AudioFormat(str, Enum):
    """Audio output formats"""
    BASE64 = "base64"
    BYTES = "bytes"


class TTSRequest(BaseModel):
    """TTS generation request"""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    engine: TTSEngine = Field(TTSEngine.GTTS, description="TTS engine to use")
    voice: str = Field("pt-br", description="Voice ID or language code")
    speed: float = Field(1.0, description="Speech speed multiplier", ge=0.5, le=2.0)
    format: AudioFormat = Field(AudioFormat.BASE64, description="Output format")

    class Config:
        """Configuration settings for """
        json_schema_extra = {
            "example": {
                "text": "Olá, como você está?",
                "engine": "gtts",
                "voice": "pt-br",
                "speed": 1.0,
                "format": "base64"
            }
        }


class TTSResponse(BaseModel):
    """TTS generation response"""
    success: bool
    audio: Optional[str] = Field(None, description="Base64 encoded audio")
    duration_seconds: Optional[float] = None
    sample_rate: int = 16000
    error: Optional[str] = None


class ProcessRequest(BaseModel):
    """Process audio/text request (backward compatible)"""
    type: Literal["audio", "text"]
    audio: Optional[str] = Field(None, description="Base64 encoded audio")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio (alias for audio)")
    text: Optional[str] = Field(None, description="Text input")
    sample_rate: int = Field(16000, description="Audio sample rate")
    language: str = Field("pt-BR", description="Language code")
    voice_id: str = Field("af_bella", description="Voice ID for TTS")
    session_id: Optional[str] = None
    force_external_llm: bool = Field(False, description="Force use of external LLM (skip primary) for benchmarking")

    class Config:
        """Configuration settings for """
        json_schema_extra = {
            "example": {
                "type": "text",
                "text": "Qual é a capital do Brasil?",
                "language": "pt-BR",
                "voice_id": "af_bella"
            }
        }


class ProcessResponse(BaseModel):
    """Process response"""
    success: bool
    transcript: Optional[str] = None
    response: Optional[str] = None
    audio: Optional[str] = Field(None, description="Base64 encoded response audio")
    metrics: Optional[dict] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
    validation_failed: Optional[bool] = None
    audio_info: Optional[dict] = None


class STTRequest(BaseModel):
    """Speech-to-text request"""
    audio: str = Field(..., description="Base64 encoded audio")
    sample_rate: int = Field(16000, description="Audio sample rate")
    language: str = Field("pt", description="Language code")

    class Config:
        """Configuration settings for """
        json_schema_extra = {
            "example": {
                "audio": "base64_encoded_audio_data",
                "sample_rate": 16000,
                "language": "pt"
            }
        }


class STTResponse(BaseModel):
    """Speech-to-text response"""
    success: bool
    transcript: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None