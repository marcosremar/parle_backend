"""
Pipeline module - Conversation Pipelines
"""

from .conversation import LocalConversationPipeline
from .external_conversation import ExternalConversationPipeline

__all__ = ['LocalConversationPipeline', 'ExternalConversationPipeline']