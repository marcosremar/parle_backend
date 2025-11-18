"""
Memory Store - Basic storage for conversation data
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a conversation message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Session:
    """Represents a conversation session"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class ConversationMemoryStorage:
    """
    In-memory storage for conversation context (conversation memory storage)
    Simple implementation for testing and development
    """

    def __init__(self,
                 max_sessions: int = 100,
                 max_messages_per_session: int = 50,
                 session_ttl_minutes: int = 60):
        """
        Initialize memory store

        Args:
            max_sessions: Maximum number of sessions to store
            max_messages_per_session: Maximum messages per session
            session_ttl_minutes: Session TTL in minutes
        """
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self.session_ttl = timedelta(minutes=session_ttl_minutes)

        self.sessions: Dict[str, Session] = {}
        self.lock = asyncio.Lock()

        logger.info(f"📝 ConversationMemoryStorage initialized (max_sessions={max_sessions})")

    async def save_interaction(self,
                               session_id: str,
                               user_input: str,
                               assistant_response: str,
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save an interaction to memory

        Args:
            session_id: Session identifier
            user_input: User's input
            assistant_response: Assistant's response
            metadata: Optional metadata
        """
        async with self.lock:
            # Get or create session
            if session_id not in self.sessions:
                self.sessions[session_id] = Session(session_id=session_id)
                logger.debug(f"Created new session: {session_id}")

            session = self.sessions[session_id]
            session.last_accessed = datetime.now()

            # Add user message
            user_msg = Message(
                role="user",
                content=user_input,
                metadata=metadata.get('user_metadata') if metadata else None
            )
            session.messages.append(user_msg)

            # Add assistant message
            assistant_msg = Message(
                role="assistant",
                content=assistant_response,
                metadata=metadata.get('assistant_metadata') if metadata else None
            )
            session.messages.append(assistant_msg)

            # Trim if too many messages
            if len(session.messages) > self.max_messages_per_session:
                # Keep only the most recent messages
                session.messages = session.messages[-self.max_messages_per_session:]

            # Clean old sessions if too many
            if len(self.sessions) > self.max_sessions:
                await self._cleanup_old_sessions()

            logger.debug(f"Saved interaction to session {session_id} "
                        f"(total messages: {len(session.messages)})")

    async def get_context(self,
                         session_id: str,
                         max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Get conversation context for a session

        Args:
            session_id: Session identifier
            max_messages: Maximum messages to return

        Returns:
            List of message dictionaries
        """
        async with self.lock:
            if session_id not in self.sessions:
                return []

            session = self.sessions[session_id]
            session.last_accessed = datetime.now()

            # Get recent messages
            messages = session.messages[-max_messages:] if max_messages else session.messages

            # Convert to dict format
            context = []
            for msg in messages:
                context.append({
                    'role': msg.role,
                    'content': msg.content
                })

            return context

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a complete session

        Args:
            session_id: Session identifier

        Returns:
            Session object or None
        """
        async with self.lock:
            return self.sessions.get(session_id)

    async def clear_session(self, session_id: str) -> bool:
        """
        Clear a specific session

        Args:
            session_id: Session identifier

        Returns:
            True if session was cleared
        """
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.debug(f"Cleared session: {session_id}")
                return True
            return False

    async def _cleanup_old_sessions(self) -> None:
        """Remove expired sessions"""
        now = datetime.now()
        expired = []

        for sid, session in self.sessions.items():
            if now - session.last_accessed > self.session_ttl:
                expired.append(sid)

        for sid in expired:
            del self.sessions[sid]
            logger.debug(f"Expired session removed: {sid}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        total_messages = sum(len(s.messages) for s in self.sessions.values())

        return {
            'total_sessions': len(self.sessions),
            'total_messages': total_messages,
            'max_sessions': self.max_sessions,
            'max_messages_per_session': self.max_messages_per_session,
            'is_initialized': True
        }