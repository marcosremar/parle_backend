"""
Pydantic models for Viber Gateway
Supports: Text, Video, Images, Files, Location, Stickers
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# ============================================================================
# USER
# ============================================================================

class ViberUser(BaseModel):
    """Viber user info"""
    id: str
    name: str
    avatar: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None


# ============================================================================
# MESSAGE TYPES
# ============================================================================

class TextMessage(BaseModel):
    """Text message"""
    text: str


class VideoMessage(BaseModel):
    """Video message"""
    media: str  # URL to MP4 file
    thumbnail: Optional[str] = None
    size: int  # File size in bytes
    duration: Optional[int] = None  # Duration in seconds


class ImageMessage(BaseModel):
    """Image/Picture message"""
    media: str  # URL to image file
    thumbnail: Optional[str] = None
    text: Optional[str] = None  # Caption


class FileMessage(BaseModel):
    """File message (receive only)"""
    media: str  # URL to file
    file_name: str
    size: int


class LocationMessage(BaseModel):
    """Location message"""
    lat: float
    lon: float


# ============================================================================
# INCOMING MESSAGES
# ============================================================================

class IncomingMessage(BaseModel):
    """Incoming message from Viber"""
    message_token: str
    timestamp: int
    sender: ViberUser
    type: Literal["text", "picture", "video", "file", "location", "contact", "sticker", "url"]

    # Message content (based on type)
    text: Optional[str] = None
    media: Optional[str] = None  # URL for picture/video/file
    file_name: Optional[str] = None  # For files
    size: Optional[int] = None
    duration: Optional[int] = None
    thumbnail: Optional[str] = None

    # Location
    location: Optional[LocationMessage] = None


# ============================================================================
# OUTGOING MESSAGES
# ============================================================================

class SendTextRequest(BaseModel):
    """Send text message"""
    receiver: str  # Viber user ID
    text: str


class SendVideoRequest(BaseModel):
    """Send video message"""
    receiver: str
    video_url: str  # Must be HTTPS, .mp4
    size: int
    thumbnail: Optional[str] = None
    duration: Optional[int] = None


class SendImageRequest(BaseModel):
    """Send image/picture message"""
    receiver: str
    image_url: str  # Must be HTTPS
    text: Optional[str] = None  # Caption


# ============================================================================
# RESPONSES
# ============================================================================

class SendMessageResponse(BaseModel):
    """Response after sending message"""
    success: bool
    message_token: Optional[str] = None
    status: Optional[int] = None
    status_message: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# ORCHESTRATOR INTEGRATION
# ============================================================================

class ProcessMessageRequest(BaseModel):
    """Request to Orchestrator"""
    user_id: str
    session_id: str
    source: str = "viber"

    message_type: Literal["text", "video"]
    text: Optional[str] = None
    video_data: Optional[bytes] = None


class ProcessMessageResponse(BaseModel):
    """Response from Orchestrator"""
    success: bool
    response_type: Literal["text", "video"]
    text: Optional[str] = None
    video_url: Optional[str] = None
    processing_time_ms: Optional[float] = None
