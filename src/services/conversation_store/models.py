"""
Pydantic Models for Conversation Store

Business logic layer - defines data structures and validation.
Uses generic Database Service SDK for storage (no DB logic here).
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4


class Message(BaseModel):
    """
    Message model with validation

    Stored in Database Service as:
        key: "message:{message_id}"
        metadata: {"type": "message", "conversation_id": "...", "role": "..."}
    """
    message_id: str = Field(
        default_factory=lambda: f"msg_{uuid4().hex[:16]}",
        description="Unique message identifier"
    )
    conversation_id: str = Field(..., description="Parent conversation ID")
    role: str = Field(..., description="Message role: 'user' | 'assistant' | 'system'")
    content: str = Field(..., min_length=1, description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    # Multi-speaker support
    speaker_id: Optional[str] = Field(
        None,
        description="Speaker identifier for multi-speaker conversations (e.g., 'SPEAKER_00', 'SPEAKER_01')"
    )

    class Config:
        """Configuration settings for """
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def dict(self, **kwargs):
        """Override dict() to properly serialize datetime"""
        d = super().dict(**kwargs)
        # Convert datetime to ISO string
        if isinstance(d.get('created_at'), datetime):
            d['created_at'] = d['created_at'].isoformat()
        return d

    def to_storage_key(self) -> str:
        """Generate storage key for Database Service"""
        return f"message:{self.message_id}"

    def to_vector_metadata(self) -> Dict[str, Any]:
        """Generate FAISS metadata for semantic search filtering"""
        metadata = {
            "type": "message",
            "conversation_id": self.conversation_id,
            "role": self.role,
            "created_at": self.created_at.isoformat()
        }
        # Add speaker_id if present (multi-speaker support)
        if self.speaker_id:
            metadata["speaker_id"] = self.speaker_id
        return metadata


class Conversation(BaseModel):
    """
    Conversation model with validation

    Stored in Database Service as:
        key: "conversation:{conversation_id}"
        metadata: {"type": "conversation", "title": "...", "user_id": "..."}
    """
    conversation_id: str = Field(
        default_factory=lambda: f"conv_{uuid4().hex[:16]}",
        description="Unique conversation identifier"
    )
    user_id: str = Field(..., description="Owner user ID")
    title: str = Field(default="New Conversation", description="Conversation title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    message_count: int = Field(default=0, description="Number of messages in conversation")

    class Config:
        """Configuration settings for """
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def dict(self, **kwargs):
        """Override dict() to properly serialize datetime"""
        d = super().dict(**kwargs)
        # Convert datetime to ISO string
        if isinstance(d.get('created_at'), datetime):
            d['created_at'] = d['created_at'].isoformat()
        if isinstance(d.get('updated_at'), datetime):
            d['updated_at'] = d['updated_at'].isoformat()
        return d

    def to_storage_key(self) -> str:
        """Generate storage key for Database Service"""
        return f"conversation:{self.conversation_id}"

    def to_vector_metadata(self) -> Dict[str, Any]:
        """Generate FAISS metadata for semantic search filtering"""
        return {
            "type": "conversation",
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def increment_message_count(self):
        """Increment message count and update timestamp"""
        self.message_count += 1
        self.updated_at = datetime.now()


class ConversationWithMessages(BaseModel):
    """
    Conversation with messages (for GET /conversations/{id})

    Not stored directly - composed from Conversation + Messages
    """
    conversation: Conversation
    messages: List[Message] = Field(default_factory=list)

    @property
    def total_messages(self) -> int:
        return len(self.messages)


# ============================================================================
# Request/Response Models (FastAPI)
# ============================================================================

class CreateConversationRequest(BaseModel):
    """Request to create a new conversation"""
    user_id: str
    title: str = "New Chat"
    metadata: Optional[Dict[str, Any]] = None


class CreateConversationResponse(BaseModel):
    """Response after creating conversation"""
    success: bool = True
    conversation: Conversation


class AddMessageRequest(BaseModel):
    """Request to add a message to conversation"""
    user_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    # Multi-speaker support
    speaker_id: Optional[str] = Field(
        None,
        description="Speaker identifier for multi-speaker conversations (e.g., 'SPEAKER_00')"
    )


class AddMessageResponse(BaseModel):
    """Response after adding message"""
    success: bool = True
    message: Message


class SearchMessagesRequest(BaseModel):
    """Request to search messages semantically"""
    user_id: str
    query: str
    top_k: int = 5


class SearchMessagesResponse(BaseModel):
    """Response with search results"""
    success: bool = True
    results: List[Message]
    count: int

    class Config:
        """Configuration settings for """
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GetMessagesRequest(BaseModel):
    """Request to get messages from conversation"""
    user_id: str
    limit: int = 100
    offset: int = 0


class GetMessagesResponse(BaseModel):
    """Response with messages"""
    success: bool = True
    messages: List[Message]
    total: int

    class Config:
        """Configuration settings for """
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
