"""
Conversational Context Manager - Main orchestrator for conversation memory
Integrates all memory components: short-term, long-term, sliding window, and embeddings
"""

import logging
from typing import Dict, List, Optional, Any

from .memory_store import ConversationMemoryStorage
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .sliding_window import SlidingWindowMemory
from .embeddings_search import EmbeddingsMemorySearch

logger = logging.getLogger(__name__)


class ConversationalContext:
    """
    Advanced Conversational Context Manager

    Centralizes all conversation context logic including:
    - Memory management (short-term, long-term)
    - Context prompt building
    - Sliding window management
    - Semantic search via embeddings
    - Conversation flow control
    """

    def __init__(self,
                 max_context_messages: int = 10,
                 context_window_size: int = 4,
                 enable_long_term_memory: bool = True,
                 enable_embeddings_search: bool = True,
                 memory_store: Optional[ConversationMemoryStorage] = None):
        """
        Initialize conversational context manager

        Args:
            max_context_messages: Maximum messages to keep in context
            context_window_size: Size of sliding context window
            enable_long_term_memory: Whether to enable long-term memory
            enable_embeddings_search: Whether to enable embeddings search
            memory_store: Optional custom memory store
        """
        self.max_context_messages = max_context_messages
        self.context_window_size = context_window_size
        self.enable_long_term_memory = enable_long_term_memory
        self.enable_embeddings_search = enable_embeddings_search

        # Initialize shared memory store
        self.memory_store = memory_store or ConversationMemoryStorage()

        # Initialize memory components
        self.short_term_memory = ShortTermMemory(
            max_turns=max_context_messages,
            memory_store=self.memory_store
        )

        self.sliding_window = SlidingWindowMemory(
            window_size=context_window_size,
            memory_store=self.memory_store
        )

        # Optional advanced memory components
        self.long_term_memory = None
        self.embeddings_search = None

        if enable_long_term_memory:
            self.long_term_memory = LongTermMemory(memory_store=self.memory_store)

        if enable_embeddings_search:
            self.embeddings_search = EmbeddingsMemorySearch(memory_store=self.memory_store)

        # Context templates
        self.base_prompt = "Você é um assistente útil."
        self.context_separator = "\\n\\nContexto da conversa:\\n"
        self.instruction_suffix = "\\nResponda em português de forma natural e conversacional."

        logger.info(f"🧠 ConversationalContext initialized "
                   f"(max_messages={max_context_messages}, window={context_window_size}, "
                   f"long_term={enable_long_term_memory}, embeddings={enable_embeddings_search})")

    async def add_interaction(self,
                             session_id: str,
                             user_input: str,
                             assistant_response: str,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a complete interaction to all memory systems

        Args:
            session_id: Session identifier
            user_input: User's input
            assistant_response: Assistant's response
            metadata: Optional metadata
        """
        # Add to short-term memory
        await self.short_term_memory.add_turn(
            session_id=session_id,
            user_input=user_input,
            assistant_response=assistant_response,
            metadata=metadata
        )

        # Add to sliding window
        await self.sliding_window.add_message_pair(
            session_id=session_id,
            user_input=user_input,
            assistant_response=assistant_response,
            metadata=metadata
        )

        # Add to embeddings search if enabled
        if self.embeddings_search:
            await self.embeddings_search.add_interaction_embeddings(
                session_id=session_id,
                user_input=user_input,
                assistant_response=assistant_response,
                metadata=metadata
            )

        # Process for long-term memory if enabled
        if self.long_term_memory:
            await self.long_term_memory.process_session_for_summarization(session_id)

        logger.debug(f"Added interaction to all memory systems: {session_id}")

    async def build_context_prompt(self,
                                  session_id: str,
                                  current_input: Optional[str] = None,
                                  use_semantic_search: bool = True) -> str:
        """
        Build a comprehensive context prompt from all memory systems

        Args:
            session_id: Session identifier
            current_input: Current user input (optional)
            use_semantic_search: Whether to include semantic search results

        Returns:
            Formatted context prompt
        """
        # Get sliding window context (most recent)
        recent_context = await self.sliding_window.get_optimized_window(
            session_id=session_id,
            current_input=current_input
        )

        # Start building prompt
        if not recent_context and not current_input:
            # No context - return base prompt
            prompt = self.base_prompt
            if current_input:
                prompt += f"\\n\\nUsuário: {current_input}"
            prompt += self.instruction_suffix
            return prompt

        # Build comprehensive contextual prompt
        prompt_parts = [self.base_prompt]

        # Add long-term memory context if available
        if self.long_term_memory and current_input:
            long_term_context = await self.long_term_memory.build_long_term_context(
                session_id=session_id,
                current_topics=self._extract_topics_from_text(current_input)
            )
            if long_term_context:
                prompt_parts.append(long_term_context)

        # Add semantic search context if enabled
        if self.embeddings_search and current_input and use_semantic_search:
            semantic_context = await self.embeddings_search.build_semantic_context(
                session_id=session_id,
                current_input=current_input
            )
            if semantic_context:
                prompt_parts.append(semantic_context)

        # Add recent conversation context
        if recent_context:
            prompt_parts.append(self.context_separator)
            for msg in recent_context[-self.context_window_size:]:
                role = "Usuário" if msg['role'] == "user" else "Você"
                content = self._truncate_content(msg['content'], 80)
                prompt_parts.append(f"{role}: {content}")

        # Add current input if provided
        if current_input:
            prompt_parts.append(f"\\nUsuário: {current_input}")

        prompt_parts.append(self.instruction_suffix)

        return "\\n".join(prompt_parts)

    async def get_conversation_summary(self,
                                     session_id: str,
                                     include_semantic_topics: bool = True) -> str:
        """
        Get a comprehensive summary of the conversation

        Args:
            session_id: Session identifier
            include_semantic_topics: Whether to include semantic topic analysis

        Returns:
            Conversation summary
        """
        # Get basic summary from short-term memory
        recent_messages = await self.short_term_memory.get_recent_context(
            session_id=session_id,
            max_turns=self.max_context_messages
        )

        if not recent_messages:
            return "Nova conversa iniciada."

        summary_parts = []

        # Basic conversation info
        summary_parts.append(f"Conversa com {len(recent_messages)//2} trocas de mensagens")

        # Get sliding window summary
        window_summary = await self.sliding_window.get_window_summary(session_id)
        summary_parts.append(window_summary)

        # Add long-term memory insights if available
        if self.long_term_memory:
            memories = await self.long_term_memory.get_relevant_memories(session_id, limit=2)
            if memories:
                memory_topics = set()
                for memory in memories:
                    memory_topics.update(memory.topics)
                if memory_topics:
                    summary_parts.append(f"Tópicos recorrentes: {', '.join(memory_topics)}")

        # Add semantic analysis if available
        if self.embeddings_search and include_semantic_topics:
            try:
                clusters = await self.embeddings_search.get_embedding_clusters(session_id, num_clusters=3)
                if clusters:
                    summary_parts.append(f"Análise semântica: {len(clusters)} grupos temáticos identificados")
            except Exception as e:
                logger.debug(f"Semantic analysis failed: {e}")

        return "\\n".join(summary_parts)

    async def get_context_window(self,
                                session_id: str,
                                window_type: str = "sliding",
                                window_size: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get context window from specified memory system

        Args:
            session_id: Session identifier
            window_type: Type of window ("sliding", "recent", "optimized")
            window_size: Size of context window

        Returns:
            List of messages in context window
        """
        if window_type == "sliding":
            return await self.sliding_window.get_context_window(
                session_id=session_id,
                max_pairs=window_size
            )
        elif window_type == "recent":
            return await self.short_term_memory.get_recent_context(
                session_id=session_id,
                max_turns=window_size or self.context_window_size
            )
        elif window_type == "optimized":
            return await self.sliding_window.get_optimized_window(session_id=session_id)
        else:
            raise ValueError(f"Unknown window type: {window_type}")

    async def search_conversation_content(self,
                                         session_id: str,
                                         query: str,
                                         max_results: int = 5) -> List[str]:
        """
        Search conversation content semantically

        Args:
            session_id: Session identifier
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of relevant content snippets
        """
        if not self.embeddings_search:
            logger.warning("Embeddings search not enabled")
            return []

        return await self.embeddings_search.find_relevant_context(
            session_id=session_id,
            current_input=query,
            max_results=max_results
        )

    async def clear_conversation(self, session_id: str) -> bool:
        """
        Clear all context for a conversation session

        Args:
            session_id: Session identifier

        Returns:
            True if session was cleared
        """
        results = []

        # Clear short-term memory
        results.append(await self.short_term_memory.clear_session_memory(session_id))

        # Clear sliding window
        results.append(await self.sliding_window.clear_window(session_id))

        # Clear long-term memory if enabled
        if self.long_term_memory:
            results.append(await self.long_term_memory.clear_session_memories(session_id))

        # Clear embeddings if enabled
        if self.embeddings_search:
            results.append(await self.embeddings_search.clear_session_embeddings(session_id))

        # Clear base memory store
        results.append(await self.memory_store.clear_session(session_id))

        success = all(results)
        if success:
            logger.info(f"Cleared all conversation context for {session_id}")

        return success

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to max length with ellipsis"""
        if len(content) <= max_length:
            return content
        return content[:max_length-3] + "..."

    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract basic topics from text (simplified implementation)"""
        topics = []
        text_lower = text.lower()

        # Simple keyword-based topic extraction
        topic_keywords = {
            'programming': ['código', 'programação', 'python', 'javascript'],
            'ai': ['ia', 'artificial', 'machine learning', 'llm'],
            'audio': ['música', 'audio', 'som', 'voice'],
            'help': ['ajuda', 'como', 'tutorial', 'explicar']
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def configure_prompts(self,
                         base_prompt: Optional[str] = None,
                         context_separator: Optional[str] = None,
                         instruction_suffix: Optional[str] = None) -> None:
        """
        Configure prompt templates

        Args:
            base_prompt: Base system prompt
            context_separator: Separator for context section
            instruction_suffix: Final instructions
        """
        if base_prompt:
            self.base_prompt = base_prompt
        if context_separator:
            self.context_separator = context_separator
        if instruction_suffix:
            self.instruction_suffix = instruction_suffix

        logger.info("🔧 ConversationalContext prompts configured")

    def get_context_stats(self) -> Dict[str, Any]:
        """Get comprehensive context statistics from all memory systems"""
        stats = {
            'max_context_messages': self.max_context_messages,
            'context_window_size': self.context_window_size,
            'base_prompt_length': len(self.base_prompt),
            'context_manager_type': 'ConversationalContext',
            'components': {
                'short_term_memory': True,
                'sliding_window': True,
                'long_term_memory': self.long_term_memory is not None,
                'embeddings_search': self.embeddings_search is not None
            }
        }

        # Add component statistics
        if self.short_term_memory:
            stats['short_term_stats'] = self.short_term_memory.get_memory_stats()

        if self.sliding_window:
            stats['sliding_window_stats'] = self.sliding_window.get_window_stats()

        if self.long_term_memory:
            stats['long_term_stats'] = self.long_term_memory.get_memory_stats()

        if self.embeddings_search:
            stats['embeddings_stats'] = self.embeddings_search.get_embedding_stats()

        return stats


# Global instance for easy access
_conversational_context: Optional[ConversationalContext] = None


def get_conversational_context() -> ConversationalContext:
    """Get global conversational context instance"""
    global _conversational_context
    if _conversational_context is None:
        _conversational_context = ConversationalContext()
    return _conversational_context


def initialize_conversational_context(max_context_messages: int = 10,
                                    context_window_size: int = 4,
                                    enable_long_term_memory: bool = True,
                                    enable_embeddings_search: bool = True) -> ConversationalContext:
    """
    Initialize global conversational context with custom settings

    Args:
        max_context_messages: Maximum messages to keep
        context_window_size: Context window size
        enable_long_term_memory: Whether to enable long-term memory
        enable_embeddings_search: Whether to enable embeddings search

    Returns:
        ConversationalContext instance
    """
    global _conversational_context
    _conversational_context = ConversationalContext(
        max_context_messages=max_context_messages,
        context_window_size=context_window_size,
        enable_long_term_memory=enable_long_term_memory,
        enable_embeddings_search=enable_embeddings_search
    )
    logger.info("🧠 Global ConversationalContext initialized")
    return _conversational_context