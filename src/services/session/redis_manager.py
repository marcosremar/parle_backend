#!/usr/bin/env python3
"""
Database-backed Session Manager
Manages session state via Database Service using HTTP
"""

import logging
import uuid
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from .models import LLMType, SessionResponse
except ImportError:
    from models import LLMType, SessionResponse

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages sessions via Database Service using HTTP

    Since both session and database are separate process services,
    they communicate via HTTP REST API.

    Database Service (port 8205) handles storage (Redis or SQLite fallback)
    """

    def __init__(
        self,
        redis_url: str = None,
        default_ttl: int = 1800,
        redis_db: int = 0,
        comm_manager=None
    ) -> None:
        """
        Initialize session manager

        Args:
            redis_url: DEPRECATED - kept for backwards compatibility
            default_ttl: Default TTL in seconds (30 minutes)
            redis_db: DEPRECATED - kept for backwards compatibility
            comm_manager: DEPRECATED - not used (HTTP direct communication)
        """
        self.default_ttl = default_ttl
        self.database_url = "http://localhost:8205"  # Database Service HTTP endpoint
        self.client = httpx.AsyncClient(timeout=10.0)

        logger.info(f"📡 SessionManager initialized (Database Service: {self.database_url})")

    def set_comm_manager(self, comm_manager) -> None:
        """
        Set Communication Manager after initialization (DEPRECATED)

        This method is kept for backwards compatibility but is not used.
        Session and Database are separate processes, so they use HTTP.
        """
        pass  # Not needed for HTTP communication

    async def connect(self) -> None:
        """Connect to Database Service (health check)"""
        try:
            response = await self.client.get(f"{self.database_url}/health")

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "healthy":
                    logger.info(f"✅ Connected to Database Service for session storage")
                    logger.info(f"   Storage: {result.get('realtime_database_connected', False) and 'Redis' or 'SQLite'}")
                    return
                else:
                    logger.warning(f"⚠️  Database Service health check returned: {result}")
            else:
                logger.warning(f"⚠️  Database Service health check failed: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️  Database Service not available: {e}")
            logger.info("Session service will return degraded status (sessions may not persist)")

    async def disconnect(self) -> None:
        """Disconnect from Database Service"""
        await self.client.aclose()
        logger.info("🔌 Disconnected from Database Service")

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
            "session_id": session_id,
            "id": session_id,
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "user_id": user_id or "",
            "active_llm": LLMType.PRIMARY.value,
            "failover_count": 0,
            "created_at": now,
            "last_activity": now,
            "metadata": metadata or {}
        }

        try:
            # Store via Database Service HTTP API
            response = await self.client.post(
                f"{self.database_url}/sessions",
                json=session_data
            )

            if response.status_code in (200, 201):
                result = response.json()
                if result.get("success"):
                    logger.info(f"📝 Created session {session_id} with scenario {scenario_id}")
                    return session_id
                else:
                    logger.error(f"❌ Failed to create session: {result}")
                    raise Exception(f"Failed to create session: {result}")
            else:
                error_text = response.text
                logger.error(f"❌ Failed to create session: HTTP {response.status_code}: {error_text}")
                raise Exception(f"Failed to create session: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Failed to create session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """
        Get session by ID

        Args:
            session_id: Session ID

        Returns:
            SessionResponse or None if not found
        """
        try:
            response = await self.client.get(f"{self.database_url}/sessions/{session_id}")

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    data = result.get("session", {})

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
                else:
                    logger.warning(f"⚠️  Session {session_id} not found")
                    return None
            else:
                logger.warning(f"⚠️  Session {session_id} not found: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to get session: {e}")
            return None

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
        # Get current session
        session = await self.get_session(session_id)
        if not session:
            return False

        # Get current data via HTTP
        try:
            response = await self.client.get(f"{self.database_url}/sessions/{session_id}")

            if response.status_code != 200:
                return False

            result = response.json()
            if not result.get("success"):
                return False

            data = result.get("session", {})

            # Update fields
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

            # Save back via HTTP
            update_response = await self.client.put(
                f"{self.database_url}/sessions/{session_id}",
                json=data
            )

            if update_response.status_code in (200, 201):
                update_result = update_response.json()
                if update_result.get("success"):
                    logger.info(f"✏️  Updated session {session_id}")
                    return True
                else:
                    logger.error(f"❌ Failed to update session: {update_result}")
                    return False
            else:
                logger.error(f"❌ Failed to update session: HTTP {update_response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to update session: {e}")
            return False

    async def heartbeat(self, session_id: str, extend_by: Optional[int] = None) -> bool:
        """
        Send heartbeat to extend session TTL

        Args:
            session_id: Session ID
            extend_by: Seconds to extend (ignored - TTL managed by Database)

        Returns:
            True if extended, False if session not found
        """
        # Update last_activity timestamp
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
        try:
            response = await self.client.delete(f"{self.database_url}/sessions/{session_id}")

            if response.status_code in (200, 204):
                result = response.json() if response.status_code == 200 else {"success": True}
                if result.get("success"):
                    logger.info(f"🗑️  Deleted session {session_id}")
                    return True
                else:
                    logger.warning(f"⚠️  Failed to delete session: {result}")
                    return False
            else:
                logger.warning(f"⚠️  Failed to delete session: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to delete session: {e}")
            return False

    async def get_all_sessions(self) -> List[SessionResponse]:
        """
        Get all active sessions

        Returns:
            List of active sessions
        """
        try:
            response = await self.client.get(f"{self.database_url}/sessions")

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    sessions_data = result.get("sessions", [])

                    sessions = []
                    for data in sessions_data:
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
                else:
                    logger.error(f"❌ Failed to list sessions: {result}")
                    return []
            else:
                logger.error(f"❌ Failed to list sessions: HTTP {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ Failed to list sessions: {e}")
            return []

    async def get_active_count(self) -> int:
        """
        Get count of active sessions

        Returns:
            Number of active sessions
        """
        sessions = await self.get_all_sessions()
        return len(sessions)

    async def is_connected(self) -> bool:
        """
        Check if Database Service is connected

        Returns:
            True if connected, False otherwise
        """
        try:
            response = await self.client.get(f"{self.database_url}/health", timeout=5.0)
            if response.status_code == 200:
                result = response.json()
                return result.get("status") == "healthy"
            return False
        except (redis.ConnectionError, redis.TimeoutError, OSError):
            return False

    async def cleanup_expired(self) -> None:
        """
        Cleanup task - Database Service handles this automatically
        This method is kept for backwards compatibility
        """
        pass
