"""
Sliding Window Memory - Efficient context window management
Maintains a fixed-size window of recent conversation context with intelligent selection
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque

from .memory_store import ConversationMemoryStorage, Message

logger = logging.getLogger(__name__)


class SlidingWindowMemory:
    """
    Sliding window memory for efficient context management
    Maintains optimal context window with intelligent message selection
    """

    def __init__(self,
                 window_size: int = 8,
                 max_tokens_per_window: int = 2000,
                 prioritize_recent: bool = True,
                 memory_store: Optional[ConversationMemoryStorage] = None):
        """
        Initialize sliding window memory

        Args:
            window_size: Number of message pairs to keep in window
            max_tokens_per_window: Maximum tokens in context window
            prioritize_recent: Whether to prioritize recent messages
            memory_store: Optional custom memory store
        """
        self.window_size = window_size
        self.max_tokens_per_window = max_tokens_per_window
        self.prioritize_recent = prioritize_recent
        self.memory_store = memory_store or ConversationMemoryStorage()

        # Active windows per session
        self.active_windows: Dict[str, deque] = {}

        logger.info(f"🪟 SlidingWindowMemory initialized (window_size={window_size}, "
                   f"max_tokens={max_tokens_per_window})")

    async def add_message_pair(self,
                              session_id: str,
                              user_input: str,
                              assistant_response: str,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a user-assistant message pair to the sliding window

        Args:
            session_id: Session identifier
            user_input: User's input
            assistant_response: Assistant's response
            metadata: Optional metadata
        """
        # Ensure window exists
        if session_id not in self.active_windows:
            self.active_windows[session_id] = deque(maxlen=self.window_size)

        window = self.active_windows[session_id]

        # Create message pair
        message_pair = {
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': datetime.now(),
            'metadata': metadata or {},
            'token_count': self._estimate_tokens(user_input + assistant_response)
        }

        # Add to window (automatically removes oldest if full)
        window.append(message_pair)

        # Store in underlying memory store as well
        await self.memory_store.save_interaction(
            session_id=session_id,
            user_input=user_input,
            assistant_response=assistant_response,
            metadata=metadata
        )

        logger.debug(f"Added message pair to sliding window: {session_id} "
                    f"(window size: {len(window)})")

    async def get_context_window(self,
                                session_id: str,
                                max_pairs: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get the current context window for a session

        Args:
            session_id: Session identifier
            max_pairs: Maximum message pairs to return

        Returns:
            List of message dictionaries in context format
        """
        if session_id not in self.active_windows:
            return []

        window = self.active_windows[session_id]
        effective_max = min(max_pairs or self.window_size, len(window))

        # Get the most recent pairs
        recent_pairs = list(window)[-effective_max:] if self.prioritize_recent else list(window)[:effective_max]

        # Convert to context format
        context_messages = []
        for pair in recent_pairs:
            context_messages.extend([
                {'role': 'user', 'content': pair['user']},
                {'role': 'assistant', 'content': pair['assistant']}
            ])

        return context_messages

    async def get_optimized_window(self,
                                  session_id: str,
                                  current_input: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get an optimized context window that fits within token limits

        Args:
            session_id: Session identifier
            current_input: Current user input to account for

        Returns:
            Optimized list of message dictionaries
        """
        if session_id not in self.active_windows:
            return []

        window = self.active_windows[session_id]
        if not window:
            return []

        # Calculate current input tokens
        current_tokens = self._estimate_tokens(current_input) if current_input else 0
        available_tokens = self.max_tokens_per_window - current_tokens

        # Select messages that fit within token limit
        selected_messages = []
        total_tokens = 0

        # Start from most recent if prioritizing recent
        pairs = reversed(list(window)) if self.prioritize_recent else window

        for pair in pairs:
            pair_tokens = pair['token_count']

            if total_tokens + pair_tokens <= available_tokens:
                if self.prioritize_recent:
                    selected_messages.insert(0, pair)  # Insert at beginning to maintain order
                else:
                    selected_messages.append(pair)
                total_tokens += pair_tokens
            else:
                break

        # Convert to context format
        context_messages = []
        for pair in selected_messages:
            context_messages.extend([
                {'role': 'user', 'content': pair['user']},
                {'role': 'assistant', 'content': pair['assistant']}
            ])

        logger.debug(f"Optimized window for {session_id}: {len(selected_messages)} pairs, "
                    f"{total_tokens} tokens")

        return context_messages

    async def get_window_summary(self, session_id: str) -> str:
        """
        Get a summary of the current context window

        Args:
            session_id: Session identifier

        Returns:
            Formatted window summary
        """
        if session_id not in self.active_windows:
            return "Nenhuma conversa no contexto."

        window = self.active_windows[session_id]
        if not window:
            return "Janela de contexto vazia."

        total_pairs = len(window)
        total_tokens = sum(pair['token_count'] for pair in window)

        # Get time range
        oldest_time = window[0]['timestamp']
        newest_time = window[-1]['timestamp']
        time_span = newest_time - oldest_time

        summary = f"Contexto atual: {total_pairs} trocas, {total_tokens} tokens, " \
                 f"período de {time_span.total_seconds()/60:.1f} minutos"

        return summary

    async def slide_window_to_position(self,
                                      session_id: str,
                                      position: int) -> List[Dict[str, str]]:
        """
        Slide window to a specific position in conversation history

        Args:
            session_id: Session identifier
            position: Position to slide to (0 = most recent)

        Returns:
            Context window at specified position
        """
        # Get full conversation history from memory store
        full_context = await self.memory_store.get_context(
            session_id=session_id,
            max_messages=None  # Get all messages
        )

        if not full_context:
            return []

        # Convert to message pairs
        pairs = []
        for i in range(0, len(full_context), 2):
            if i + 1 < len(full_context):
                pairs.append({
                    'user': full_context[i]['content'],
                    'assistant': full_context[i + 1]['content'],
                    'timestamp': datetime.now(),  # Simplified
                    'token_count': self._estimate_tokens(
                        full_context[i]['content'] + full_context[i + 1]['content']
                    )
                })

        # Calculate window position
        start_idx = max(0, len(pairs) - self.window_size - position)
        end_idx = min(len(pairs), start_idx + self.window_size)

        selected_pairs = pairs[start_idx:end_idx]

        # Convert to context format
        context_messages = []
        for pair in selected_pairs:
            context_messages.extend([
                {'role': 'user', 'content': pair['user']},
                {'role': 'assistant', 'content': pair['assistant']}
            ])

        return context_messages

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (simplified)

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Simplified token estimation (rough approximation)
        # In production, use actual tokenizer
        return len(text.split()) * 1.3  # Assume ~1.3 tokens per word

    async def clear_window(self, session_id: str) -> bool:
        """
        Clear the sliding window for a session

        Args:
            session_id: Session identifier

        Returns:
            True if cleared successfully
        """
        if session_id in self.active_windows:
            self.active_windows[session_id].clear()
            logger.info(f"Cleared sliding window for {session_id}")
            return True
        return False

    async def resize_window(self, session_id: str, new_size: int) -> None:
        """
        Resize the sliding window for a session

        Args:
            session_id: Session identifier
            new_size: New window size
        """
        if session_id in self.active_windows:
            old_window = self.active_windows[session_id]
            new_window = deque(old_window, maxlen=new_size)
            self.active_windows[session_id] = new_window

            logger.info(f"Resized sliding window for {session_id}: {new_size}")

    def get_window_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get sliding window statistics

        Args:
            session_id: Optional specific session, or all sessions

        Returns:
            Statistics dictionary
        """
        if session_id:
            if session_id not in self.active_windows:
                return {'error': 'Session not found'}

            window = self.active_windows[session_id]
            total_tokens = sum(pair['token_count'] for pair in window)

            return {
                'session_id': session_id,
                'window_size': len(window),
                'max_window_size': self.window_size,
                'total_tokens': total_tokens,
                'max_tokens_per_window': self.max_tokens_per_window,
                'utilization': len(window) / self.window_size,
                'memory_type': 'sliding_window'
            }

        # Global statistics
        total_sessions = len(self.active_windows)
        total_active_windows = sum(len(window) for window in self.active_windows.values())
        total_tokens = sum(
            sum(pair['token_count'] for pair in window)
            for window in self.active_windows.values()
        )

        return {
            'total_sessions': total_sessions,
            'total_active_windows': total_active_windows,
            'total_tokens': total_tokens,
            'window_size': self.window_size,
            'max_tokens_per_window': self.max_tokens_per_window,
            'average_utilization': total_active_windows / (total_sessions * self.window_size) if total_sessions > 0 else 0,
            'memory_type': 'sliding_window'
        }