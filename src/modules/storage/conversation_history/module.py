"""
Conversation History Module - Direct Python calls for Conversation history
"""

from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule


class ConversationHistoryModule(BaseModule):
    """Conversation History Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("conversation_history")
        self.history_db = {}
    
    async def _initialize(self) -> bool:
        """Initialize conversation history storage"""
        try:
            # For now, use in-memory storage
            self.history_db = {}
            self.logger.info("✅ Conversation History Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Conversation history service not available: {e}")
            self.history_db = {}
            return True
    
    async def save_turn(
        self,
        conversation_id: str,
        user_input: str,
        ai_response: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Save a conversation turn"""
        if conversation_id not in self.history_db:
            self.history_db[conversation_id] = []
        
        turn = {
            "user_input": user_input,
            "ai_response": ai_response,
            "metadata": metadata or {},
            "timestamp": None
        }
        
        self.history_db[conversation_id].append(turn)
        return {"success": True}
    
    async def get_history(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.history_db.get(conversation_id, [])[-limit:]
