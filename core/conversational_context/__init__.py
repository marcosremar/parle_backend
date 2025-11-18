"""
Conversational Context Module
Centralized memory and context management for conversations

This module contains all conversation memory-related functionality:
- Short-term and long-term memory
- Sliding window memory system
- Embeddings search for memory retrieval
- Context prompt building
- Session management
"""

from .context_manager import ConversationalContext, get_conversational_context, initialize_conversational_context
from .memory_store import ConversationMemoryStorage, Message, Session
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .sliding_window import SlidingWindowMemory
from .embeddings_search import EmbeddingsMemorySearch

# Backward compatibility alias
InMemoryStore = ConversationMemoryStorage

__all__ = [
    'ConversationalContext',
    'get_conversational_context',
    'initialize_conversational_context',
    'ConversationMemoryStorage',
    'InMemoryStore',  # Backward compatibility
    'Message',
    'Session',
    'ShortTermMemory',
    'LongTermMemory',
    'SlidingWindowMemory',
    'EmbeddingsMemorySearch'
]