"""
Conversation History Module - Direct Python calls for Conversation history
"""

from typing import Dict, Optional, Any, List

from src.modules.base_module import BaseModule


class ConversationHistoryModule(BaseModule):
    """Conversation History Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("conversation_history")
        self.history_db = {}
    
    async def _initialize(self) -> bool:
        """Initialize conversation history storage"""
        try:
            # Try to import conversation history service
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
        if not self.initialized:
            await self.initialize()
        
        try:
            from datetime import datetime
            import secrets
            
            turn = {
                "turn_id": f"turn_{secrets.token_hex(8)}",
                "conversation_id": conversation_id,
                "user_input": user_input,
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            if conversation_id not in self.history_db:
                self.history_db[conversation_id] = []
            self.history_db[conversation_id].append(turn)
            
            return turn
        except Exception as e:
            self.logger.error(f"❌ Failed to save turn: {e}")
            raise
    
    async def get_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if not self.initialized:
            await self.initialize()
        
        turns = self.history_db.get(conversation_id, [])
        if limit:
            return turns[-limit:]
        return turns
    
    async def get_all_conversations(self, user_id: Optional[str] = None) -> List[str]:
        """Get all conversation IDs, optionally filtered by user_id"""
        if not self.initialized:
            await self.initialize()
        
        # For now, return all conversation IDs
        # In the future, can filter by user_id if metadata includes it
        return list(self.history_db.keys())
