"""
Data Service Clients

Clients for session, scenarios, conversation store, and storage services.
"""

from __future__ import annotations

import base64
import logging
from typing import Dict, Any, Optional, List

from .base import BaseServiceClient, ServiceClientError

logger = logging.getLogger(__name__)


class SessionClient(BaseServiceClient):
    """Session service client"""

    def __init__(self) -> None:
        super().__init__("session", is_module_service=True)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        # Use direct module call in monolith mode
        if self.monolith_mode and self.direct_module:
            # Lazy initialize module if needed
            if not getattr(self, '_module_initialized', False):
                if hasattr(self.direct_module, 'initialize'):
                    try:
                        await self.direct_module.initialize()
                        self._module_initialized = True
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to initialize session module: {e}")
                        # Fall back to HTTP
                        return await self._get(f"/api/sessions/{session_id}")
            
            try:
                return await self.direct_module.get_session(session_id)
            except Exception as e:
                logger.warning(f"⚠️  Direct module call failed: {e}, falling back to HTTP")
                return await self._get(f"/api/sessions/{session_id}")
        
        try:
            return await self._get(f"/api/sessions/{session_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ Session {session_id} not found")
            return None

    async def create_session(
        self,
        conversation_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new session with optional specific session_id"""
        # Use direct module call in monolith mode
        if self.monolith_mode and self.direct_module:
            # Lazy initialize module if needed
            if not getattr(self, '_module_initialized', False):
                if hasattr(self.direct_module, 'initialize'):
                    try:
                        await self.direct_module.initialize()
                        self._module_initialized = True
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to initialize session module: {e}")
                        # Fall back to HTTP
                        return await self._create_session_http(conversation_id, scenario_id, session_id)
            
            try:
                result = await self.direct_module.create_session(
                    user_id=user_id or "",
                    scenario_id=scenario_id or "default",
                    conversation_id=conversation_id
                )
                logger.debug(f"✅ Created session via module: {result.get('session_id')}")
                return result
            except Exception as e:
                logger.warning(f"⚠️  Direct module call failed: {e}, falling back to HTTP")
                return await self._create_session_http(conversation_id, scenario_id, session_id)
        
        return await self._create_session_http(conversation_id, scenario_id, session_id)
    
    async def _create_session_http(
        self,
        conversation_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """HTTP fallback for create_session"""
        try:
            session_data: Dict[str, Any] = {
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
            await self._put(
                f"/api/sessions/{session_id}/llm",
                json_data={"active_llm": llm_type}
            )
            logger.debug(f"✅ Updated session {session_id} LLM to {llm_type}")
            return True
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to update session LLM")
            return False


class ScenariosClient(BaseServiceClient):
    """Scenarios service client"""

    def __init__(self) -> None:
        super().__init__("scenarios", is_module_service=True)

    async def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get scenario configuration"""
        # Use direct module call in monolith mode
        if self.monolith_mode and self.direct_module:
            # Lazy initialize module if needed
            if not getattr(self, '_module_initialized', False):
                if hasattr(self.direct_module, 'initialize'):
                    try:
                        await self.direct_module.initialize()
                        self._module_initialized = True
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to initialize scenarios module: {e}")
                        # Fall back to HTTP
                        return await self._get(f"/api/scenarios/{scenario_id}")
            
            try:
                return await self.direct_module.get_scenario(scenario_id)
            except Exception as e:
                logger.warning(f"⚠️  Direct module call failed: {e}, falling back to HTTP")
                return await self._get(f"/api/scenarios/{scenario_id}")
        
        try:
            return await self._get(f"/api/scenarios/{scenario_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ Scenario {scenario_id} not found")
            return None

    async def validate_turn(
        self,
        scenario_id: str,
        user_message: str,
        expected_topics: List[str],
        turn_number: int = 1,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate conversation turn against scenario context"""
        try:
            result = await self._post(
                f"/api/scenarios/{scenario_id}/validate-turn",
                json_data={
                    "user_message": user_message,
                    "expected_topics": expected_topics,
                    "turn_number": turn_number,
                    "session_id": session_id
                },
                timeout=5.0
            )
            logger.debug(f"✅ Validated turn: coherence={result.get('coherence_score', 0):.2f}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ Turn validation failed: {e}")
            raise

    async def initialize_scenario_state(
        self,
        scenario_id: str,
        session_id: str,
        expected_topics: List[str]
    ) -> Dict[str, Any]:
        """Initialize scenario state for a new conversation session"""
        try:
            result = await self._post(
                f"/api/scenarios/{scenario_id}/initialize-state",
                json_data={
                    "session_id": session_id,
                    "scenario_id": scenario_id,
                    "expected_topics": expected_topics
                }
            )
            logger.debug(f"🎬 Initialized scenario state for session {session_id}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ State initialization failed: {e}")
            raise

    async def get_scenario_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current scenario state for a session"""
        try:
            return await self._get(f"/api/scenarios/state/{session_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ State for session {session_id} not found")
            return None

    async def update_scenario_state(
        self,
        session_id: str,
        validation_result: Dict[str, Any]
    ) -> bool:
        """Update scenario state with validation metrics"""
        try:
            await self._post(
                f"/api/scenarios/state/{session_id}/update",
                json_data=validation_result
            )
            logger.debug(f"✅ Updated scenario state for session {session_id}")
            return True
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to update scenario state")
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
        super().__init__("conversation_store")

    async def add_turn(
        self,
        conversation_id: str,
        user_audio: Optional[bytes] = None,
        user_text: str = "",
        ai_text: str = "",
        ai_audio: Optional[bytes] = None
    ) -> bool:
        """Save conversation turn"""
        # Use direct module call in monolith mode
        if self.monolith_mode and self.direct_module:
            # Lazy initialize module if needed
            if not getattr(self, '_module_initialized', False):
                if hasattr(self.direct_module, 'initialize'):
                    try:
                        await self.direct_module.initialize()
                        self._module_initialized = True
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to initialize conversation_store module: {e}")
                        # Fall back to HTTP
                        return await self._add_turn_http(conversation_id, user_audio, user_text, ai_text, ai_audio)
            
            try:
                # Save user message
                if user_text:
                    await self.direct_module.save_message(
                        conversation_id=conversation_id,
                        role="user",
                        content=user_text
                    )
                # Save AI message
                if ai_text:
                    await self.direct_module.save_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=ai_text
                    )
                logger.debug(f"💾 Saved turn to conversation {conversation_id} via module")
                return True
            except Exception as e:
                logger.warning(f"⚠️  Direct module call failed: {e}, falling back to HTTP")
                return await self._add_turn_http(conversation_id, user_audio, user_text, ai_text, ai_audio)
        
        return await self._add_turn_http(conversation_id, user_audio, user_text, ai_text, ai_audio)
    
    async def _add_turn_http(
        self,
        conversation_id: str,
        user_audio: Optional[bytes] = None,
        user_text: str = "",
        ai_text: str = "",
        ai_audio: Optional[bytes] = None
    ) -> bool:
        """HTTP fallback for add_turn"""
        try:
            turn_data: Dict[str, Any] = {
                "user_text": user_text,
                "ai_text": ai_text
            }

            if user_audio:
                turn_data["user_audio"] = base64.b64encode(user_audio).decode()
            if ai_audio:
                turn_data["ai_audio"] = base64.b64encode(ai_audio).decode()

            await self._post(
                f"/api/conversations/{conversation_id}/turn",
                json_data=turn_data
            )
            logger.debug(f"💾 Saved turn to conversation {conversation_id}")
            return True
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to save conversation turn")
            return False

    async def get_context(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for context"""
        # Use direct module call in monolith mode
        if self.monolith_mode and self.direct_module:
            # Lazy initialize module if needed
            if not getattr(self, '_module_initialized', False):
                if hasattr(self.direct_module, 'initialize'):
                    try:
                        await self.direct_module.initialize()
                        self._module_initialized = True
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to initialize conversation_store module: {e}")
                        # Fall back to HTTP
                        return await self._get_context_http(conversation_id, limit)
            
            try:
                return await self.direct_module.get_context(conversation_id=conversation_id, limit=limit)
            except Exception as e:
                logger.warning(f"⚠️  Direct module call failed: {e}, falling back to HTTP")
                return await self._get_context_http(conversation_id, limit)
        
        return await self._get_context_http(conversation_id, limit)
    
    async def _get_context_http(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """HTTP fallback for get_context"""
        try:
            result = await self._get(f"/api/conversations/{conversation_id}/messages?limit={limit}")
            return result.get("messages", [])
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to get conversation context")
            return []

    async def create_conversation(self) -> Dict[str, Any]:
        """Create a new conversation"""
        try:
            result = await self._post("/api/conversations", json_data={})
            logger.debug(f"✅ Created conversation: {result.get('conversation_id')}")
            return result
        except ServiceClientError as e:
            logger.error(f"❌ Failed to create conversation: {e}")
            raise


class ConversationHistoryClient(BaseServiceClient):
    """Conversation history service client"""

    def __init__(self) -> None:
        super().__init__("conversation_history")

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID"""
        try:
            return await self._get(f"/api/conversations/{conversation_id}")
        except ServiceClientError:
            return None

    async def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
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

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data"""
        try:
            return await self._get(f"/api/users/{user_id}")
        except ServiceClientError:
            logger.warning(f"⚠️ User {user_id} not found")
            return None

    async def create_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new user"""
        try:
            user_data: Dict[str, Any] = {"user_id": user_id}
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
                "/api/auth/verify",
                json_data={"user_id": user_id, "token": token}
            )
            return result.get("valid", False)
        except ServiceClientError:
            return False


class DatabaseClient(BaseServiceClient):
    """Database service client"""

    def __init__(self) -> None:
        super().__init__("database", is_module_service=True)

    async def query(self, table: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query database"""
        try:
            result = await self._post(
                "/api/query",
                json_data={"table": table, "filters": filters or {}}
            )
            return result.get("results", [])
        except ServiceClientError as e:
            logger.error(f"❌ Database query failed: {e}")
            return []

    async def insert(self, table: str, data: Dict[str, Any]) -> bool:
        """Insert data into database"""
        try:
            await self._post(
                "/api/insert",
                json_data={"table": table, "data": data}
            )
            return True
        except ServiceClientError:
            return False


class FileStorageClient(BaseServiceClient):
    """File storage service client"""

    def __init__(self) -> None:
        super().__init__("file_storage", is_module_service=True)

    async def upload_file(self, file_data: bytes, file_name: str) -> Optional[str]:
        """Upload file to storage"""
        try:
            file_base64 = base64.b64encode(file_data).decode()
            result = await self._post(
                "/api/files/upload",
                json_data={"file_data": file_base64, "file_name": file_name}
            )
            return result.get("file_id")
        except ServiceClientError:
            return None

    async def get_file(self, file_id: str) -> Optional[bytes]:
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
