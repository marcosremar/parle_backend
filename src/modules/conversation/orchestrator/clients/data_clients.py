"""
Data Service Clients

Clients for session, scenarios, conversation store, and storage services.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from .base import BaseServiceClient, ServiceClientError

logger = logging.getLogger(__name__)


class SessionClient(BaseServiceClient):
    """Session service client"""

    def __init__(self) -> None:
        super().__init__("session", is_module_service=True)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session data"""
        # Validate input
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ServiceClientError("session_id must be a non-empty string")

        # Use direct module call (module services always use direct calls)
        await self._ensure_module_initialized()

        try:
            return await self.direct_module.get_session(session_id)
        except Exception as e:
            self._handle_module_error("get_session", e)

    async def create_session(
        self,
        conversation_id: str | None = None,
        scenario_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new session with optional specific session_id"""
        # Use direct module call (module services always use direct calls)
        await self._ensure_module_initialized()

        try:
            result = await self.direct_module.create_session(
                user_id=user_id or "",
                scenario_id=scenario_id or "default",
                conversation_id=conversation_id,
            )
            logger.debug(f"✅ Created session via module: {result.get('session_id')}")
            return result
        except Exception as e:
            self._handle_module_error("create_session", e)

    async def _create_session_http(
        self,
        conversation_id: str | None = None,
        scenario_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """HTTP fallback for create_session"""
        try:
            session_data: dict[str, Any] = {
                "scenario_id": scenario_id or "default",
            }
            if conversation_id:
                session_data["conversation_id"] = conversation_id
            if session_id:
                session_data["session_id"] = session_id

            result = await self._post("/api/sessions", json_data=session_data)
            logger.debug(f"✅ Created session: {result.get('id')}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ Failed to create session: {e}")
            raise

    async def update_session_llm(self, session_id: str, llm_type: str) -> bool:
        """Update which LLM is serving this session"""
        try:
            await self._put(f"/api/sessions/{session_id}/llm", json_data={"active_llm": llm_type})
            logger.debug(f"✅ Updated session {session_id} LLM to {llm_type}")
            return True
        except ServiceClientError:
            logger.warning("⚠️ Failed to update session LLM")
            return False


class ScenariosClient(BaseServiceClient):
    """Scenarios service client"""

    def __init__(self) -> None:
        super().__init__("scenarios", is_module_service=True)

    async def get_scenario(self, scenario_id: str) -> dict[str, Any] | None:
        """Get scenario configuration"""
        # Validate input
        if not scenario_id or not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ServiceClientError("scenario_id must be a non-empty string")

        # Use direct module call (module services always use direct calls)
        await self._ensure_module_initialized()

        try:
            return await self.direct_module.get_scenario(scenario_id)
        except Exception as e:
            self._handle_module_error("get_scenario", e)

    async def validate_turn(
        self,
        scenario_id: str,
        user_message: str,
        expected_topics: list[str],
        turn_number: int = 1,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate conversation turn against scenario context"""
        try:
            result = await self._post(
                f"/api/scenarios/{scenario_id}/validate-turn",
                json_data={
                    "user_message": user_message,
                    "expected_topics": expected_topics,
                    "turn_number": turn_number,
                    "session_id": session_id,
                },
                timeout=5.0,
            )
            logger.debug(f"✅ Validated turn: coherence={result.get('coherence_score', 0):.2f}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ Turn validation failed: {e}")
            raise

    async def initialize_scenario_state(
        self, scenario_id: str, session_id: str, expected_topics: list[str]
    ) -> dict[str, Any]:
        """Initialize scenario state for a new conversation session"""
        try:
            result = await self._post(
                f"/api/scenarios/{scenario_id}/initialize-state",
                json_data={
                    "session_id": session_id,
                    "scenario_id": scenario_id,
                    "expected_topics": expected_topics,
                },
            )
            logger.debug(f"🎬 Initialized scenario state for session {session_id}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ State initialization failed: {e}")
            raise

    async def get_scenario_state(self, session_id: str) -> dict[str, Any] | None:
        """Get current scenario state for a session"""
        try:
            return await self._get(f"/api/scenarios/state/{session_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ State for session {session_id} not found")
            return None

    async def update_scenario_state(
        self, session_id: str, validation_result: dict[str, Any]
    ) -> bool:
        """Update scenario state with validation metrics"""
        try:
            await self._post(
                f"/api/scenarios/state/{session_id}/update", json_data=validation_result
            )
            logger.debug(f"✅ Updated scenario state for session {session_id}")
            return True
        except ServiceClientError:
            logger.warning("⚠️ Failed to update scenario state")
            return False

    async def delete_scenario_state(self, session_id: str) -> bool:
        """Delete scenario state for a session"""
        try:
            await self._post(f"/api/scenarios/state/{session_id}/delete", json_data={})
            logger.debug(f"✅ Deleted scenario state for session {session_id}")
            return True
        except ServiceClientError:
            return False


class ConversationStoreClient(BaseServiceClient):
    """Conversation store service client"""

    def __init__(self) -> None:
        # Try to use as module service first (for monolith mode)
        # Fallback to HTTP if module not available
        super().__init__("conversation_store", is_module_service=True)

    async def add_turn(
        self,
        conversation_id: str,
        user_audio: bytes | None = None,
        user_text: str = "",
        ai_text: str = "",
        ai_audio: bytes | None = None,
    ) -> bool:
        """Save conversation turn"""
        # Validate input
        if (
            not conversation_id
            or not isinstance(conversation_id, str)
            or not conversation_id.strip()
        ):
            raise ServiceClientError("conversation_id must be a non-empty string")

        # Try direct module call first (monolith mode)
        if self.is_module_service and self.direct_module:
            try:
                await self._ensure_module_initialized()
                # Save user message
                if user_text:
                    await self.direct_module.save_message(
                        conversation_id=conversation_id,
                        role="user",
                        content=user_text,
                        metadata={"has_audio": user_audio is not None} if user_audio else None
                    )
                # Save AI message
                if ai_text:
                    await self.direct_module.save_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=ai_text,
                        metadata={"has_audio": ai_audio is not None} if ai_audio else None
                    )
                return True
            except Exception as e:
                logger.warning(f"⚠️ Direct module call failed for add_turn: {e}, falling back to HTTP")
                # Fall through to HTTP fallback
        
        # Fallback to HTTP if module not available or failed
        # Only try HTTP if we have a session (not in pure monolith mode)
        if self.session is not None:
            try:
                return await self._add_turn_http(conversation_id, user_audio, user_text, ai_text, ai_audio)
            except Exception as e:
                logger.warning(f"⚠️ HTTP call also failed for add_turn: {e}")
                return False
        else:
            # No session available, return False (monolith mode without HTTP)
            logger.debug("No HTTP session available for conversation_store, skipping add_turn")
            return False

    async def _add_turn_http(
        self,
        conversation_id: str,
        user_audio: bytes | None = None,
        user_text: str = "",
        ai_text: str = "",
        ai_audio: bytes | None = None,
    ) -> bool:
        """
        HTTP method for add_turn

        Note: Used by ConversationStoreClient (not a module service).
        """
        try:
            turn_data: dict[str, Any] = {"user_text": user_text, "ai_text": ai_text}

            if user_audio:
                turn_data["user_audio"] = base64.b64encode(user_audio).decode()
            if ai_audio:
                turn_data["ai_audio"] = base64.b64encode(ai_audio).decode()

            await self._post(f"/api/conversations/{conversation_id}/turn", json_data=turn_data)
            logger.debug(f"💾 Saved turn to conversation {conversation_id}")
            return True
        except ServiceClientError:
            logger.warning("⚠️ Failed to save conversation turn")
            return False

    async def get_context(self, conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get conversation history for context"""
        # Validate input
        if (
            not conversation_id
            or not isinstance(conversation_id, str)
            or not conversation_id.strip()
        ):
            raise ServiceClientError("conversation_id must be a non-empty string")
        if not isinstance(limit, int) or limit < 1:
            raise ServiceClientError("limit must be a positive integer")

        # Try direct module call first (monolith mode)
        if self.is_module_service and self.direct_module:
            try:
                await self._ensure_module_initialized()
                result = await self.direct_module.get_context(conversation_id, limit)
                return result if isinstance(result, list) else []
            except Exception as e:
                logger.warning(f"⚠️ Direct module call failed for get_context: {e}, falling back to HTTP")
                # Fall through to HTTP fallback
        
        # Fallback to HTTP if module not available or failed
        # Only try HTTP if we have a session (not in pure monolith mode)
        if self.session is not None:
            try:
                return await self._get_context_http(conversation_id, limit)
            except Exception as e:
                logger.warning(f"⚠️ HTTP call also failed for get_context: {e}, returning empty list")
                return []
        else:
            # No session available, return empty list (monolith mode without HTTP)
            logger.debug("No HTTP session available for conversation_store, returning empty context")
            return []

    async def _get_context_http(
        self, conversation_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        HTTP method for get_context

        Note: Used by ConversationStoreClient (not a module service).
        """
        try:
            result = await self._get(f"/api/conversations/{conversation_id}/messages?limit={limit}")
            return result.get("messages", [])
        except ServiceClientError:
            logger.warning("⚠️ Failed to get conversation context")
            return []

    async def create_conversation(self) -> dict[str, Any]:
        """Create a new conversation"""
        # Try direct module call first (monolith mode)
        if self.is_module_service and self.direct_module:
            try:
                await self._ensure_module_initialized()
                # Generate a conversation_id
                import secrets
                conversation_id = f"conv_{secrets.token_hex(8)}"
                return {"conversation_id": conversation_id, "success": True}
            except Exception as e:
                logger.warning(f"⚠️ Direct module call failed for create_conversation: {e}, falling back to HTTP")
                # Fall through to HTTP fallback
        
        # Fallback to HTTP if module not available or failed
        try:
            result = await self._post("/api/conversations", json_data={})
            logger.debug(f"✅ Created conversation: {result.get('conversation_id')}")
            return result
        except ServiceClientError as e:
            logger.warning(f"⚠️ HTTP call also failed for create_conversation: {e}")
            # Return a minimal conversation_id even if both fail
            import secrets
            return {"conversation_id": f"conv_{secrets.token_hex(8)}", "success": False, "error": str(e)}


class ConversationHistoryClient(BaseServiceClient):
    """Conversation history service client"""

    def __init__(self) -> None:
        super().__init__("conversation_history")

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get conversation by ID"""
        try:
            return await self._get(f"/api/conversations/{conversation_id}")
        except ServiceClientError:
            return None

    async def get_user_conversations(self, user_id: str) -> list[dict[str, Any]]:
        """Get all conversations for a user"""
        try:
            result = await self._get(f"/api/users/{user_id}/conversations")
            return result.get("conversations", [])
        except ServiceClientError:
            return []


class UserClient(BaseServiceClient):
    """User service client"""

    def __init__(self) -> None:
        super().__init__("user")

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get user data"""
        try:
            return await self._get(f"/api/users/{user_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ User {user_id} not found")
            return None

    async def create_user(
        self, user_id: str, name: str | None = None, email: str | None = None
    ) -> dict[str, Any]:
        """Create a new user"""
        try:
            user_data: dict[str, Any] = {"user_id": user_id}
            if name:
                user_data["name"] = name
            if email:
                user_data["email"] = email

            result = await self._post("/api/users", json_data=user_data)
            logger.debug(f"✅ Created user: {user_id}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ Failed to create user: {e}")
            raise

    async def authenticate(self, user_id: str, token: str) -> bool:
        """Authenticate user with token"""
        try:
            result = await self._post(
                "/api/auth/verify", json_data={"user_id": user_id, "token": token}
            )
            return result.get("valid", False)
        except ServiceClientError:
            return False


class DatabaseClient(BaseServiceClient):
    """Database service client"""

    def __init__(self) -> None:
        super().__init__("database", is_module_service=True)

    async def query(
        self, table: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query database"""
        try:
            result = await self._post(
                "/api/query", json_data={"table": table, "filters": filters or {}}
            )
            return result.get("results", [])
        except ServiceClientError as e:
            logger.error(f"❌ Database query failed: {e}")
            return []

    async def insert(self, table: str, data: dict[str, Any]) -> bool:
        """Insert data into database"""
        try:
            await self._post("/api/insert", json_data={"table": table, "data": data})
            return True
        except ServiceClientError:
            return False


class FileStorageClient(BaseServiceClient):
    """File storage service client"""

    def __init__(self) -> None:
        super().__init__("file_storage", is_module_service=True)

    async def upload_file(self, file_data: bytes, file_name: str) -> str | None:
        """Upload file to storage"""
        try:
            file_base64 = base64.b64encode(file_data).decode()
            result = await self._post(
                "/api/files/upload", json_data={"file_data": file_base64, "file_name": file_name}
            )
            return result.get("file_id")
        except ServiceClientError:
            return None

    async def get_file(self, file_id: str) -> bytes | None:
        """Get file from storage"""
        try:
            result = await self._get(f"/api/files/{file_id}")
            file_base64 = result.get("file_data")
            if not file_base64:
                return None
            return base64.b64decode(file_base64)
        except ServiceClientError:
            logger.warning(f"⚠️ File {file_id} not found")
            return None

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from storage"""
        try:
            await self._post(f"/api/files/{file_id}/delete", json_data={})
            logger.debug(f"✅ Deleted file: {file_id}")
            return True
        except ServiceClientError:
            return False
