"""
Session Module - Direct Python calls for session management
"""

import os
from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule


class SessionModule(BaseModule):
    """Session Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("session")
        self.manager = None
        self.use_redis = os.getenv("USE_REDIS_SESSION", "false").lower() == "true"
    
    async def _initialize(self) -> bool:
        """Initialize session manager"""
        try:
            # In monolith mode, use in-memory by default unless USE_REDIS_SESSION=true
            if not self.use_redis:
                from .in_memory_manager import InMemorySessionManager
                self.manager = InMemorySessionManager()
                if hasattr(self.manager, 'initialize'):
                    await self.manager.initialize()
                elif hasattr(self.manager, 'connect'):
                    await self.manager.connect()
                self.logger.info("✅ Session Module initialized (in-memory)")
                return True
            else:
                # Try to use HTTP-based SessionManager (requires Database Service)
                from .manager import SessionManager
                self.manager = SessionManager()
                if hasattr(self.manager, 'initialize'):
                    await self.manager.initialize()
                elif hasattr(self.manager, 'connect'):
                    await self.manager.connect()
                self.logger.info("✅ Session Module initialized (HTTP/Database Service)")
                return True
        except Exception as e:
            self.logger.warning(f"⚠️  Session manager not available: {e}, falling back to in-memory")
            # Fallback to in-memory manager
            try:
                from .in_memory_manager import InMemorySessionManager
                self.manager = InMemorySessionManager()
                if hasattr(self.manager, 'initialize'):
                    await self.manager.initialize()
                elif hasattr(self.manager, 'connect'):
                    await self.manager.connect()
                self.logger.info("✅ Session Module initialized (in-memory fallback)")
                return True
            except Exception as e2:
                self.logger.error(f"❌ Failed to initialize in-memory session manager: {e2}")
                self.manager = None
                return True
    
    async def create_session(
        self,
        user_id: str,
        scenario_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new session"""
        if not self.initialized:
            await self.initialize()
        
        if not self.manager:
            # Fallback: create basic session dict
            import uuid
            from datetime import datetime
            session_id = str(uuid.uuid4())
            return {
                "session_id": session_id,
                "user_id": user_id,
                "scenario_id": scenario_id,
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "created_at": datetime.now().isoformat()
            }
        
        try:
            # SessionManager.create_session pode ter assinatura diferente
            if hasattr(self.manager, 'create_session'):
                # Tentar diferentes assinaturas
                try:
                    session_id = await self.manager.create_session(
                        scenario_id=scenario_id or "default",
                        conversation_id=conversation_id,
                        user_id=user_id
                    )
                    # Se retornou apenas ID, buscar sessão completa
                    if isinstance(session_id, str):
                        session = await self.get_session(session_id)
                        return session or {"session_id": session_id, "user_id": user_id}
                    return session_id
                except TypeError:
                    # Tentar outra assinatura
                    session_id = await self.manager.create_session(
                        user_id=user_id,
                        scenario_id=scenario_id,
                        conversation_id=conversation_id
                    )
                    if isinstance(session_id, str):
                        session = await self.get_session(session_id)
                        return session or {"session_id": session_id, "user_id": user_id}
                    return session_id
            else:
                # Fallback
                import uuid
                from datetime import datetime
                session_id = str(uuid.uuid4())
                return {
                    "session_id": session_id,
                    "user_id": user_id,
                    "scenario_id": scenario_id,
                    "conversation_id": conversation_id or str(uuid.uuid4()),
                    "created_at": datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            # Fallback
            import uuid
            from datetime import datetime
            session_id = str(uuid.uuid4())
            return {
                "session_id": session_id,
                "user_id": user_id,
                "scenario_id": scenario_id,
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "created_at": datetime.now().isoformat()
            }
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        if not self.initialized:
            await self.initialize()
        
        if not self.manager:
            return None
        
        try:
            session = await self.manager.get_session(session_id)
            if not session:
                return None
            
            # Converter para dict se necessário
            if hasattr(session, 'dict'):
                session_dict = session.dict()
            elif hasattr(session, '__dict__'):
                session_dict = session.__dict__
            elif isinstance(session, dict):
                session_dict = session
            else:
                session_dict = {"id": str(session)}
            
            # Garantir que session_id existe (pode ser 'id' no dict)
            if "session_id" not in session_dict and "id" in session_dict:
                session_dict["session_id"] = session_dict["id"]
            
            return session_dict
        except Exception as e:
            self.logger.error(f"Failed to get session: {e}")
            return None
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List sessions"""
        if not self.initialized:
            await self.initialize()
        
        if not self.manager:
            return []
        
        try:
            sessions = await self.manager.list_sessions(user_id=user_id) if hasattr(self.manager, 'list_sessions') else []
            # Converter para lista de dicts
            result = []
            for s in sessions:
                if hasattr(s, 'dict'):
                    result.append(s.dict())
                elif hasattr(s, '__dict__'):
                    result.append(s.__dict__)
                else:
                    result.append(s)
            return result
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return []
