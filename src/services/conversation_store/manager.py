"""
Conversation Manager - Business Logic Layer

Manages conversations and messages using generic Database Service SDK.
No database logic here - only business rules and data structure management.
"""

import logging
from typing import List, Optional
from datetime import datetime

from src.services.database import DatabaseClientFactory, LoadStrategy
from .models import Conversation, Message

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Business logic for conversations

    Uses generic Database Service SDK for storage.
    Defines conversation structure and business rules.
    """

    def __init__(
        self,
        db_type=None,  # Deprecated - kept for backward compatibility
        load_strategy: LoadStrategy = LoadStrategy.EAGER
    ):
        """
        Initialize conversation manager

        Args:
            db_type: Deprecated - Database Service now uses FAISS+SQLite by default
            load_strategy: Load strategy (EAGER for best performance)
        """
        self.load_strategy = load_strategy
        logger.info(f"ConversationManager initialized ({load_strategy.value})")

    def _get_db_client(self, user_id: str):
        """Get database client for user"""
        return DatabaseClientFactory.create(
            user_id=user_id,
            load_strategy=self.load_strategy
        )

    async def create_conversation(
        self,
        user_id: str,
        title: str = "New Chat",
        metadata: Optional[dict] = None
    ) -> Conversation:
        """
        Create new conversation

        Args:
            user_id: User identifier
            title: Conversation title
            metadata: Optional metadata

        Returns:
            Created conversation
        """
        # Create Pydantic model (validation)
        conv = Conversation(
            user_id=user_id,
            title=title,
            metadata=metadata or {}
        )

        db = self._get_db_client(user_id)

        # Store in Database Service (generic key-value)
        await db.set(
            key=conv.to_storage_key(),
            value=conv.dict(),
            metadata=conv.metadata  # For semantic search with FAISS
        )

        logger.info(f"Created conversation {conv.conversation_id} for user {user_id}")
        return conv

    async def get_conversation(
        self,
        user_id: str,
        conversation_id: str
    ) -> Optional[Conversation]:
        """
        Get conversation by ID

        Args:
            user_id: User identifier
            conversation_id: Conversation ID

        Returns:
            Conversation or None if not found
        """
        db = self._get_db_client(user_id)

        # Get from Database Service
        conv_data = await db.get(key=f"conversation:{conversation_id}")

        if conv_data:
            return Conversation(**conv_data)

        return None

    async def list_conversations(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Conversation]:
        """
        List user conversations

        Args:
            user_id: User identifier
            limit: Max number of conversations
            offset: Offset for pagination

        Returns:
            List of conversations
        """
        db = self._get_db_client(user_id)

        # List all conversation keys
        conv_keys = await db.list_keys(prefix="conversation:")

        # Get all conversations
        conversations = []
        for key in conv_keys:
            conv_data = await db.get(key=key)
            if conv_data:
                conversations.append(Conversation(**conv_data))

        # Sort by updated_at (most recent first)
        conversations.sort(key=lambda c: c.updated_at, reverse=True)

        # Pagination
        return conversations[offset:offset + limit]

    async def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> Message:
        """
        Add message to conversation

        Args:
            user_id: User identifier
            conversation_id: Conversation ID
            role: Message role ("user" | "assistant" | "system")
            content: Message content
            metadata: Optional metadata

        Returns:
            Created message
        """
        # Create Pydantic model (validation)
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )

        db = self._get_db_client(user_id)

        # Store message in Database Service
        # FAISS auto-indexes 'content' for semantic search
        await db.set(
            key=msg.to_storage_key(),
            value=msg.dict(),
            metadata=msg.metadata
        )

        # Update conversation message count and timestamp
        conv_data = await db.get(key=f"conversation:{conversation_id}")
        if conv_data:
            conv = Conversation(**conv_data)
            conv.increment_message_count()
            await db.set(
                key=conv.to_storage_key(),
                value=conv.dict(),
                metadata=conv.metadata
            )

        logger.debug(f"Added message {msg.message_id} to conversation {conversation_id}")
        return msg

    async def get_messages(
        self,
        user_id: str,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """
        Get messages for conversation

        Args:
            user_id: User identifier
            conversation_id: Conversation ID
            limit: Max number of messages
            offset: Offset for pagination

        Returns:
            List of messages
        """
        db = self._get_db_client(user_id)

        # List all message keys
        all_keys = await db.list_keys(prefix="message:")

        # Filter messages for this conversation
        messages = []
        for key in all_keys:
            msg_data = await db.get(key=key)
            if msg_data and msg_data.get("conversation_id") == conversation_id:
                messages.append(Message(**msg_data))

        # Sort by created_at (oldest first)
        messages.sort(key=lambda m: m.created_at)

        # Pagination
        return messages[offset:offset + limit]

    async def search_messages(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Message]:
        """
        Semantic search in conversation messages using FAISS

        Args:
            user_id: User identifier
            conversation_id: Conversation ID
            query: Natural language search query
            top_k: Number of results

        Returns:
            List of matching messages (sorted by similarity)
        """
        db = self._get_db_client(user_id)

        # Semantic search with metadata filter
        # Database Service uses FAISS automatically
        results = await db.search(
            query=query,
            metadata_filter={
                "type": "message",
                "conversation_id": conversation_id
            },
            top_k=top_k
        )

        # Convert to Message models
        messages = []
        for result in results:
            # Remove _similarity_score before creating Message
            similarity = result.pop("_similarity_score", 0.0)
            msg = Message(**result)
            # Add similarity as metadata for context
            msg.metadata["similarity_score"] = similarity
            messages.append(msg)

        logger.debug(f"Semantic search found {len(messages)} messages in {conversation_id}")
        return messages

    async def delete_conversation(
        self,
        user_id: str,
        conversation_id: str
    ) -> bool:
        """
        Delete conversation and all its messages

        Args:
            user_id: User identifier
            conversation_id: Conversation ID

        Returns:
            True if deleted
        """
        db = self._get_db_client(user_id)

        # Get all messages for this conversation
        messages = await self.get_messages(user_id, conversation_id)

        # Delete all messages
        for msg in messages:
            await db.delete(key=msg.to_storage_key())

        # Delete conversation
        await db.delete(key=f"conversation:{conversation_id}")

        logger.info(f"Deleted conversation {conversation_id} with {len(messages)} messages")
        return True

    async def get_stats(self, user_id: str) -> dict:
        """
        Get conversation statistics for user

        Args:
            user_id: User identifier

        Returns:
            Stats dict
        """
        db = self._get_db_client(user_id)

        conv_keys = await db.list_keys(prefix="conversation:")
        msg_keys = await db.list_keys(prefix="message:")

        return {
            "total_conversations": len(conv_keys),
            "total_messages": len(msg_keys),
            "user_id": user_id
        }
