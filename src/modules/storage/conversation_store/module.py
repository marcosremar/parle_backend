"""
Conversation Store Module - Direct Python calls for conversation storage
"""

from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule
from .storage import FastConversationStorage


class ConversationStoreModule(BaseModule):
    """Conversation Store Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("conversation_store")
        self.storage = None
    
    async def _initialize(self) -> bool:
        """Initialize conversation storage"""
        try:
            from .storage import FastConversationStorage
            self.storage = FastConversationStorage()
            # FastConversationStorage pode ter initialize
            if hasattr(self.storage, 'initialize'):
                try:
                    await self.storage.initialize()
                except TypeError:
                    # Se initialize não aceita argumentos, não chamar
                    pass
            self.logger.info("✅ Conversation Store Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Conversation store not available: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            # Fallback to in-memory
            self.storage = None
            return True
    
    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Save a message to conversation"""
        if not self.storage or self.storage == {}:
            return {"success": False, "error": "Storage not initialized"}
        
        try:
            # FastConversationStorage.save_message returns the message dict with message_id
            message = await self.storage.save_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata
            )
            
            # Return message with message_id (should already be a dict)
            if isinstance(message, dict):
                return message
            else:
                # If it's a Pydantic model, convert to dict
                if hasattr(message, 'dict'):
                    return message.dict()
                elif hasattr(message, 'model_dump'):
                    return message.model_dump()
                else:
                    # Fallback: create dict with message_id if available
                    return {
                        "message_id": getattr(message, 'message_id', f"msg_{id(message)}"),
                        "conversation_id": conversation_id,
                        "role": role,
                        "content": content,
                        "success": True
                    }
        except Exception as e:
            self.logger.error(f"Failed to save message: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    async def get_context(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation context"""
        if not self.storage:
            return []
        
        try:
            return await self.storage.get_context(
                conversation_id=conversation_id,
                limit=limit
            )
        except Exception as e:
            self.logger.error(f"Failed to get context: {e}")
            return []
