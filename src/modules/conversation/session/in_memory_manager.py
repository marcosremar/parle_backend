#!/usr/bin/env python3
"""
In-Memory Session Manager
Manages session state in memory (no external dependencies)
"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from .models import LLMType, SessionResponse
except ImportError:
    try:
        from src.services.session.models import LLMType, SessionResponse
    except ImportError:
        # Fallback: define minimal models
        from enum import Enum
        class LLMType(Enum):
            PRIMARY = "primary"
            FALLBACK = "fallback"
        class SessionResponse:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            def dict(self):
                return self.__dict__

logger = logging.getLogger(__name__)


class InMemorySessionManager:
    """
    In-memory session manager for monolith mode
    Stores sessions in a dictionary (no external dependencies)
    """

    def __init__(
        self,
        default_ttl: int = 1800,
    ) -> None:
        """
        Initialize in-memory session manager

        Args:
            default_ttl: Default TTL in seconds (30 minutes)
        """
        self.default_ttl = default_ttl
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info(f"📦 InMemorySessionManager initialized (TTL: {default_ttl}s)")

    async def connect(self) -> None:
        """Connect (no-op for in-memory)"""
        logger.debug("In-memory session manager ready")

    async def disconnect(self) -> None:
        """Disconnect (no-op for in-memory)"""
        logger.debug("In-memory session manager disconnected")

    async def initialize(self) -> None:
        """Initialize (alias for connect)"""
        await self.connect()

    async def create_session(
        self,
        scenario_id: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Create a new session

        Args:
            scenario_id: ID of the scenario to use
            conversation_id: Existing conversation ID (creates new if None)
            user_id: User identifier
            metadata: Additional session metadata
            session_id: Specific session ID to use (generates UUID if None)

        Returns:
            Session ID
        """
        # Use provided session_id or generate a new UUID
        if not session_id:
            session_id = str(uuid.uuid4())

        # If no conversation_id provided, create one
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        now = datetime.utcnow().isoformat()

        session_data = {
            "id": session_id,
            "session_id": session_id,
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "user_id": user_id or "",
            "active_llm": LLMType.PRIMARY.value,
            "failover_count": 0,
            "created_at": now,
            "last_activity": now,
            "ttl_seconds": self.default_ttl,
            "metadata": metadata or {}
        }

        self.sessions[session_id] = session_data
        logger.info(f"📝 Created in-memory session {session_id} with scenario {scenario_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """
        Get session by ID

        Args:
            session_id: Session ID

        Returns:
            SessionResponse or None if not found
        """
        if session_id not in self.sessions:
            return None

        data = self.sessions[session_id]
        return SessionResponse(
            id=data["id"],
            scenario_id=data["scenario_id"],
            conversation_id=data["conversation_id"],
            user_id=data.get("user_id"),
            active_llm=LLMType(data.get("active_llm", "primary")),
            failover_count=data.get("failover_count", 0),
            created_at=data["created_at"],
            last_activity=data["last_activity"],
            ttl_seconds=self.default_ttl,
            metadata=data.get("metadata", {})
        )

    async def update_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        active_llm: Optional[LLMType] = None
    ) -> bool:
        """
        Update session metadata and/or active LLM

        Args:
            session_id: Session ID
            metadata: New metadata (merges with existing)
            active_llm: New active LLM type

        Returns:
            True if updated, False if session not found
        """
        if session_id not in self.sessions:
            return False

        data = self.sessions[session_id]
        data["last_activity"] = datetime.utcnow().isoformat()

        if metadata:
            data.setdefault("metadata", {}).update(metadata)

        if active_llm:
            old_llm = data.get("active_llm")
            data["active_llm"] = active_llm.value

            # Increment failover count if switching to fallback
            if old_llm != active_llm.value and active_llm == LLMType.FALLBACK:
                data["failover_count"] = data.get("failover_count", 0) + 1
                logger.warning(f"🔄 Session {session_id} failed over to {active_llm.value} "
                             f"(count: {data['failover_count']})")

        logger.debug(f"✏️  Updated in-memory session {session_id}")
        return True

    async def heartbeat(self, session_id: str, extend_by: Optional[int] = None) -> bool:
        """
        Send heartbeat to extend session TTL

        Args:
            session_id: Session ID
            extend_by: Seconds to extend (ignored - TTL managed in-memory)

        Returns:
            True if extended, False if session not found
        """
        return await self.update_session(
            session_id=session_id,
            metadata={"last_heartbeat": datetime.utcnow().isoformat()}
        )

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🗑️  Deleted in-memory session {session_id}")
            return True
        return False

    async def list_sessions(self, user_id: Optional[str] = None) -> List[SessionResponse]:
        """
        List sessions (optionally filtered by user_id)

        Args:
            user_id: Optional user ID filter

        Returns:
            List of sessions
        """
        sessions = []
        for data in self.sessions.values():
            if user_id and data.get("user_id") != user_id:
                continue
            sessions.append(SessionResponse(
                id=data["id"],
                scenario_id=data["scenario_id"],
                conversation_id=data["conversation_id"],
                user_id=data.get("user_id"),
                active_llm=LLMType(data.get("active_llm", "primary")),
                failover_count=data.get("failover_count", 0),
                created_at=data["created_at"],
                last_activity=data["last_activity"],
                ttl_seconds=self.default_ttl,
                metadata=data.get("metadata", {})
            ))
        return sessions

    async def get_all_sessions(self) -> List[SessionResponse]:
        """
        Get all active sessions

        Returns:
            List of active sessions
        """
        return await self.list_sessions()

    async def get_active_count(self) -> int:
        """
        Get count of active sessions

        Returns:
            Number of active sessions
        """
        return len(self.sessions)

    async def is_connected(self) -> bool:
        """
        Check if manager is ready

        Returns:
            True (always ready for in-memory)
        """
        return True

    async def cleanup_expired(self) -> None:
        """
        Cleanup expired sessions (basic implementation)
        """
        now = datetime.utcnow()
        expired = []
        for session_id, data in self.sessions.items():
            last_activity = datetime.fromisoformat(data["last_activity"])
            age_seconds = (now - last_activity).total_seconds()
            if age_seconds > self.default_ttl:
                expired.append(session_id)

        for session_id in expired:
            del self.sessions[session_id]
            logger.debug(f"🧹 Cleaned up expired session {session_id}")

        if expired:
            logger.info(f"🧹 Cleaned up {len(expired)} expired sessions")
