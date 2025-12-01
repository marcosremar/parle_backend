#!/usr/bin/env python3
"""
Pydantic Models for Session Service
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LLMType(str, Enum):
    """Which LLM is currently serving the session"""

    PRIMARY = "primary"  # Local Ultravox
    FALLBACK = "fallback"  # External Groq


class SessionCreate(BaseModel):
    """Request to create a new session"""

    session_id: str | None = Field(
        None, description="Specific session ID to use (generates UUID if not provided)"
    )
    scenario_id: str = Field(..., description="ID of the scenario to use")
    conversation_id: str | None = Field(None, description="Existing conversation ID (optional)")
    user_id: str | None = Field(None, description="User identifier (optional)")
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Additional metadata")
    # Multi-speaker support
    speakers: dict[str, str] | None = Field(
        default_factory=dict,
        description="Speaker ID to voice ID mapping (e.g., {'SPEAKER_00': 'af_heart', 'SPEAKER_01': 'am_adam'})",
    )
    speaker_names: dict[str, str] | None = Field(
        default_factory=dict,
        description="Speaker ID to human-readable name mapping (optional, e.g., {'SPEAKER_00': 'Alice'})",
    )


class SessionUpdate(BaseModel):
    """Request to update session metadata"""

    metadata: dict[str, Any] | None = None
    active_llm: LLMType | None = None
    # Multi-speaker support
    speakers: dict[str, str] | None = None
    speaker_names: dict[str, str] | None = None


class SessionResponse(BaseModel):
    """Session information response"""

    id: str = Field(..., description="Session ID")
    scenario_id: str = Field(..., description="Scenario ID")
    conversation_id: str = Field(..., description="Conversation ID")
    user_id: str | None = Field(None, description="User ID")
    active_llm: LLMType = Field(LLMType.PRIMARY, description="Currently active LLM")
    failover_count: int = Field(0, description="Number of times failover occurred")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    last_activity: str = Field(..., description="Last activity timestamp (ISO format)")
    ttl_seconds: int = Field(..., description="Seconds until expiration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    # Multi-speaker support
    speakers: dict[str, str] = Field(
        default_factory=dict,
        description="Speaker ID to voice ID mapping (e.g., {'SPEAKER_00': 'af_heart', 'SPEAKER_01': 'am_adam'})",
    )
    speaker_names: dict[str, str] = Field(
        default_factory=dict,
        description="Speaker ID to human-readable name mapping (optional, e.g., {'SPEAKER_00': 'Alice'})",
    )


class SessionListResponse(BaseModel):
    """List of active sessions"""

    sessions: list[SessionResponse]
    total: int
    active_count: int


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    redis_connected: bool
    active_sessions: int
    timestamp: str
