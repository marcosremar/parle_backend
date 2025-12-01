"""
Conversation Store Module - Direct Python calls for Conversation storage
"""

from typing import Any

from src.modules.base_module import BaseModule


class ConversationStoreModule(BaseModule):
    """Conversation Store Module for direct Python calls"""

    def __init__(self):
        super().__init__("conversation_store")
        self.storage = None
        self.fast_storage = None

    async def _initialize(self) -> bool:
        """Initialize conversation store"""
        try:
            # Import conversation store from local module
            from .conversation_store.storage import FastConversationStorage

            # Initialize fast storage
            self.fast_storage = FastConversationStorage()
            await self.fast_storage.initialize()

            # Fallback to in-memory storage
            self.storage = {}

            self.logger.info("✅ Conversation Store Module initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Conversation Store Module: {e}")
            # Fallback to in-memory
            self.storage = {}
            self.fast_storage = None
            return True

    async def save_message(
        self, conversation_id: str, role: str, content: str, metadata: dict | None = None
    ) -> dict[str, Any]:
        """Save a message to conversation"""
        if not self.initialized:
            await self.initialize()

        try:
            from datetime import datetime
            import secrets

            message = {
                "message_id": f"msg_{secrets.token_hex(8)}",
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            if self.fast_storage:
                await self.fast_storage.save_message(
                    conversation_id=conversation_id, role=role, content=content, metadata=metadata
                )
            else:
                # Fallback to in-memory
                if conversation_id not in self.storage:
                    self.storage[conversation_id] = []
                self.storage[conversation_id].append(message)

            return message
        except Exception as e:
            self.logger.error(f"❌ Failed to save message: {e}")
            raise

    async def get_context(self, conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get conversation context (recent messages)"""
        if not self.initialized:
            await self.initialize()

        try:
            if self.fast_storage:
                return await self.fast_storage.get_context(
                    conversation_id=conversation_id, limit=limit
                )
            else:
                # Fallback to in-memory
                messages = self.storage.get(conversation_id, [])
                return messages[-limit:] if messages else []
        except Exception as e:
            self.logger.error(f"❌ Failed to get context: {e}")
            return []

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get full conversation"""
        if not self.initialized:
            await self.initialize()

        try:
            if self.fast_storage:
                return await self.fast_storage.get_conversation(conversation_id)
            else:
                # Fallback to in-memory
                messages = self.storage.get(conversation_id, [])
                return (
                    {"conversation_id": conversation_id, "messages": messages} if messages else None
                )
        except Exception as e:
            self.logger.error(f"❌ Failed to get conversation: {e}")
            return None

    async def search_conversations(
        self, query: str, user_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search conversations by semantic similarity"""
        if not self.initialized:
            await self.initialize()

        try:
            if self.fast_storage:
                return await self.fast_storage.search(query=query, user_id=user_id, limit=limit)
            else:
                # Fallback: simple text search
                results = []
                for conv_id, messages in self.storage.items():
                    for msg in messages:
                        if query.lower() in msg.get("content", "").lower():
                            results.append({"conversation_id": conv_id, "message": msg})
                            if len(results) >= limit:
                                return results
                return results
        except Exception as e:
            self.logger.error(f"❌ Failed to search conversations: {e}")
            return []
