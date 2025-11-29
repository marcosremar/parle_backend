"""
Models for Neural Codec Module
"""
from typing import Literal
from pydantic import BaseModel, Field, field_validator
import base64


class EncodeRequest(BaseModel):
    """Request model for encoding audio"""
    audio_data: str = Field(..., description="Base64 encoded PCM audio data")
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    codec: Literal["encodec"] = Field(default="encodec", description="Codec to use")
    
    @field_validator("audio_data")
    @classmethod
    def validate_audio_data(cls, v: str) -> str:
        try:
            base64.b64decode(v)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}")
        return v


class EncodeResponse(BaseModel):
    """Response model for encoded audio"""
    encoded_data: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    latency_ms: float
    codec: str


class DecodeRequest(BaseModel):
    """Request model for decoding audio"""
    encoded_data: str = Field(..., description="Base64 encoded compressed audio")
    codec: Literal["encodec"] = Field(default="encodec", description="Codec to use")


class DecodeResponse(BaseModel):
    """Response model for decoded audio"""
    audio_data: str
    sample_rate: int
    audio_size: int
    latency_ms: float
    codec: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    codec_available: bool
    device: str


class CodecInfoResponse(BaseModel):
    """Codec information response"""
    codec: str
    sample_rate: int
    bitrate: float
    latency_ms: float
    compression_ratio: float
    device: str
    streaming_enabled: bool
