"""
Short-term Memory - Immediate conversation context
Handles recent conversation turns for immediate context building
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .memory_store import ConversationMemoryStorage, Message

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Short-term memory for immediate conversation context
    Optimized for recent interactions and fast access
    """

    def __init__(self,
                 max_turns: int = 10,
                 ttl_minutes: int = 30,
                 memory_store: Optional[ConversationMemoryStorage] = None):
        """
        Initialize short-term memory

        Args:
            max_turns: Maximum conversation turns to keep
            ttl_minutes: Time-to-live for short-term memories
            memory_store: Optional custom memory store
        """
        self.max_turns = max_turns
        self.ttl = timedelta(minutes=ttl_minutes)
        self.memory_store = memory_store or ConversationMemoryStorage()

        logger.info(f"🧠 ShortTermMemory initialized (max_turns={max_turns}, ttl={ttl_minutes}min)")

    async def add_turn(self,
                       session_id: str,
                       user_input: str,
                       assistant_response: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation turn to short-term memory

        Args:
            session_id: Session identifier
            user_input: User's input
            assistant_response: Assistant's response
            metadata: Optional metadata
        """
        # Store in the underlying memory store
        await self.memory_store.save_interaction(
            session_id=session_id,
            user_input=user_input,
            assistant_response=assistant_response,
            metadata=metadata
        )

        logger.debug(f"Added turn to short-term memory: {session_id}")

    async def get_recent_context(self,
                                session_id: str,
                                max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get recent conversation context

        Args:
            session_id: Session identifier
            max_turns: Maximum turns to return (defaults to configured max)

        Returns:
            List of recent message dictionaries
        """
        effective_max = max_turns or self.max_turns

        # Get recent messages (user/assistant pairs = turns)
        messages = await self.memory_store.get_context(
            session_id=session_id,
            max_messages=effective_max * 2  # Account for user/assistant pairs
        )

        # Filter out expired messages
        now = datetime.now()
        recent_messages = []

        for msg_dict in messages:
            # Note: In this simple implementation, we assume messages are recent
            # In a more sophisticated system, we'd check timestamps
            recent_messages.append(msg_dict)

        return recent_messages

    async def get_last_user_input(self, session_id: str) -> Optional[str]:
        """
        Get the last user input from short-term memory

        Args:
            session_id: Session identifier

        Returns:
            Last user input or None
        """
        messages = await self.get_recent_context(session_id, max_turns=1)

        # Find the last user message
        for msg in reversed(messages):
            if msg['role'] == 'user':
                return msg['content']

        return None

    async def get_last_assistant_response(self, session_id: str) -> Optional[str]:
        """
        Get the last assistant response from short-term memory

        Args:
            session_id: Session identifier

        Returns:
            Last assistant response or None
        """
        messages = await self.get_recent_context(session_id, max_turns=1)

        # Find the last assistant message
        for msg in reversed(messages):
            if msg['role'] == 'assistant':
                return msg['content']

        return None

    async def build_immediate_context(self,
                                     session_id: str,
                                     current_input: Optional[str] = None) -> str:
        """
        Build immediate context prompt from short-term memory

        Args:
            session_id: Session identifier
            current_input: Current user input

        Returns:
            Formatted context string
        """
        recent_messages = await self.get_recent_context(session_id)

        if not recent_messages and not current_input:
            return "Nova conversa iniciada."

        context_parts = []

        # Add recent conversation
        for msg in recent_messages[-6:]:  # Last 3 turns
            role = "Usuário" if msg['role'] == "user" else "Assistente"
            content = self._truncate_content(msg['content'], 100)
            context_parts.append(f"{role}: {content}")

        # Add current input
        if current_input:
            context_parts.append(f"Usuário: {current_input}")

        return "\\n".join(context_parts)

    async def clear_session_memory(self, session_id: str) -> bool:
        """
        Clear short-term memory for a session

        Args:
            session_id: Session identifier

        Returns:
            True if cleared successfully
        """
        return await self.memory_store.clear_session(session_id)

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to max length with ellipsis"""
        if len(content) <= max_length:
            return content
        return content[:max_length-3] + "..."

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get short-term memory statistics"""
        base_stats = self.memory_store.get_stats()

        return {
            **base_stats,
            'max_turns': self.max_turns,
            'ttl_minutes': self.ttl.total_seconds() / 60,
            'memory_type': 'short_term'
        }