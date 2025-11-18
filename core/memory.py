"""
Conversational Context - Compatibility layer for existing code
Imports from the new conversational_context module for backwards compatibility
"""

import logging

# Import from the new modular structure
from src.core.conversational_context import (
    ConversationalContext,
    get_conversational_context,
    initialize_conversational_context,
    ConversationMemoryStorage,
    Message,
    Session
)

# Backward compatibility alias
InMemoryStore = ConversationMemoryStorage

logger = logging.getLogger(__name__)


# All functionality is now available through the imports above
# This file maintains backward compatibility for existing code