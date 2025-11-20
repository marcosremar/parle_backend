"""
Orchestrator Service - Pydantic Models
Request and response models for conversation orchestration
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ConversationRequest(BaseModel):
    """Text conversation request"""
    session_id: str = Field(..., description="Session ID for conversation")
    message: str = Field(..., description="User message text")
    scenario_id: Optional[str] = Field(default=None, description="Scenario ID (overrides session scenario)")
    voice_id: Optional[str] = Field(default=None, description="Voice ID for TTS response (also triggers STT validation)")
    stt_mode: str = Field(default="local", description="STT service to use: 'local' (port 8099)")


class ConversationResponse(BaseModel):
    """Text conversation response"""
    success: bool
    response: str
    session_id: str
    audio: Optional[str] = Field(default=None, description="Base64 encoded audio response (if voice_id provided)")
    transcription: Optional[str] = Field(default=None, description="STT transcription of audio response")
    transcription_match: Optional[bool] = Field(default=None, description="True if transcription matches original text")
    context_size: Optional[int] = None
    messages_count: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
