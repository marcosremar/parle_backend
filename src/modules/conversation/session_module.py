"""
Session Module - Direct Python calls for Session management
"""

from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule


class SessionModule(BaseModule):
    """Session Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("session")
        self.session_manager = None
    
    async def _initialize(self) -> bool:
        """Initialize session manager"""
        try:
            # Import session manager
            from src.services.session.redis_manager import SessionManager
            from config.settings import get_sessions_settings
            
            settings = get_sessions_settings()
            self.session_manager = SessionManager(redis_url=settings.redis_url)
            await self.session_manager.connect()
            
            self.logger.info("✅ Session Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to initialize Session Module with Redis: {e}")
            # Fallback to in-memory storage
            self.session_manager = None
            self._sessions = {}
            return True
    
    async def create_session(
        self,
        user_id: str,
        scenario_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new session"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.session_manager:
                # SessionManager.create_session expects scenario_id first
                session_id = await self.session_manager.create_session(
                    scenario_id=scenario_id or "default",
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata=kwargs
                )
                # Get the created session
                session = await self.session_manager.get_session(session_id)
                return session.dict() if hasattr(session, 'dict') else session
            else:
                # Fallback to in-memory
                import secrets
                from datetime import datetime
                session_id = f"session_{secrets.token_hex(8)}"
                session = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "scenario_id": scenario_id,
                    "conversation_id": conversation_id,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    **kwargs
                }
                self._sessions[session_id] = session
                return session
        except Exception as e:
            self.logger.error(f"❌ Session creation failed: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.session_manager:
                session = await self.session_manager.get_session(session_id)
                if session:
                    return session.dict() if hasattr(session, 'dict') else session
                return None
            else:
                return self._sessions.get(session_id)
        except Exception as e:
            self.logger.error(f"❌ Failed to get session: {e}")
            return None
    
    async def update_session(
        self,
        session_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """Update session"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.session_manager:
                return await self.session_manager.update_session(session_id, **updates)
            else:
                if session_id in self._sessions:
                    from datetime import datetime
                    self._sessions[session_id].update(updates)
                    self._sessions[session_id]["updated_at"] = datetime.now().isoformat()
                    return self._sessions[session_id]
                return None
        except Exception as e:
            self.logger.error(f"❌ Session update failed: {e}")
            raise
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.session_manager:
                return await self.session_manager.delete_session(session_id)
            else:
                if session_id in self._sessions:
                    del self._sessions[session_id]
                    return True
                return False
        except Exception as e:
            self.logger.error(f"❌ Session deletion failed: {e}")
            return False
    
    async def list_sessions(self, user_id: Optional[str] = None) -> list:
        """List sessions, optionally filtered by user_id"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.session_manager:
                sessions = await self.session_manager.get_all_sessions()
                sessions_list = [
                    s.dict() if hasattr(s, 'dict') else s
                    for s in sessions
                ]
                if user_id:
                    return [s for s in sessions_list if s.get("user_id") == user_id]
                return sessions_list
            else:
                if user_id:
                    return [
                        s for s in self._sessions.values()
                        if s.get("user_id") == user_id
                    ]
                return list(self._sessions.values())
        except Exception as e:
            self.logger.error(f"❌ Failed to list sessions: {e}")
            return []
