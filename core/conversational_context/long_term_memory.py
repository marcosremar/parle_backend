"""
Long-term Memory - Persistent conversation knowledge
Handles long-term storage, summarization, and retrieval of conversation history
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .memory_store import ConversationMemoryStorage, Session, Message

logger = logging.getLogger(__name__)


@dataclass
class MemorySummary:
    """Represents a summarized memory chunk"""
    session_id: str
    summary: str
    topics: List[str]
    importance_score: float
    created_at: datetime
    message_count: int
    metadata: Optional[Dict[str, Any]] = None


class LongTermMemory:
    """
    Long-term memory for persistent conversation knowledge
    Handles summarization, storage, and retrieval of important conversation elements
    """

    def __init__(self,
                 summarization_threshold: int = 20,
                 max_summaries: int = 100,
                 importance_threshold: float = 0.6,
                 memory_store: Optional[ConversationMemoryStorage] = None):
        """
        Initialize long-term memory

        Args:
            summarization_threshold: Number of messages before summarization
            max_summaries: Maximum number of summaries to keep
            importance_threshold: Minimum importance score to keep memories
            memory_store: Optional custom memory store
        """
        self.summarization_threshold = summarization_threshold
        self.max_summaries = max_summaries
        self.importance_threshold = importance_threshold
        self.memory_store = memory_store or ConversationMemoryStorage()

        # In-memory storage for summaries (in production, use persistent storage)
        self.summaries: Dict[str, List[MemorySummary]] = {}

        logger.info(f"🧠 LongTermMemory initialized (threshold={summarization_threshold}, "
                   f"max_summaries={max_summaries})")

    async def process_session_for_summarization(self, session_id: str) -> Optional[MemorySummary]:
        """
        Process a session to create long-term memory summaries

        Args:
            session_id: Session identifier

        Returns:
            Created summary or None
        """
        session = await self.memory_store.get_session(session_id)
        if not session or len(session.messages) < self.summarization_threshold:
            return None

        # Create summary from recent messages
        summary = await self._create_summary(session)

        if summary and summary.importance_score >= self.importance_threshold:
            # Store the summary
            if session_id not in self.summaries:
                self.summaries[session_id] = []

            self.summaries[session_id].append(summary)

            # Cleanup old summaries if needed
            await self._cleanup_old_summaries(session_id)

            logger.info(f"Created long-term memory summary for {session_id} "
                       f"(importance: {summary.importance_score:.2f})")

            return summary

        return None

    async def _create_summary(self, session: Session) -> MemorySummary:
        """
        Create a summary from session messages

        Args:
            session: Session to summarize

        Returns:
            Memory summary
        """
        # Extract topics and create summary (simplified implementation)
        messages = session.messages[-self.summarization_threshold:]

        # Extract key topics (simplified - in production, use NLP)
        topics = self._extract_topics(messages)

        # Create summary text
        summary_text = self._create_summary_text(messages)

        # Calculate importance score (simplified heuristic)
        importance = self._calculate_importance(messages, topics)

        return MemorySummary(
            session_id=session.session_id,
            summary=summary_text,
            topics=topics,
            importance_score=importance,
            created_at=datetime.now(),
            message_count=len(messages),
            metadata={'source': 'auto_summarization'}
        )

    def _extract_topics(self, messages: List[Message]) -> List[str]:
        """
        Extract topics from messages (simplified implementation)

        Args:
            messages: List of messages

        Returns:
            List of extracted topics
        """
        # Simplified topic extraction
        topics = set()

        for msg in messages:
            content = msg.content.lower()

            # Simple keyword-based topic extraction
            if any(word in content for word in ['código', 'programação', 'python']):
                topics.add('programming')
            if any(word in content for word in ['ia', 'artificial', 'machine learning']):
                topics.add('ai')
            if any(word in content for word in ['música', 'audio', 'som']):
                topics.add('audio')
            if any(word in content for word in ['problema', 'erro', 'bug']):
                topics.add('troubleshooting')
            if any(word in content for word in ['ajuda', 'como', 'tutorial']):
                topics.add('help')

        return list(topics)

    def _create_summary_text(self, messages: List[Message]) -> str:
        """
        Create summary text from messages

        Args:
            messages: List of messages

        Returns:
            Summary text
        """
        # Count user vs assistant messages
        user_messages = [m for m in messages if m.role == 'user']
        assistant_messages = [m for m in messages if m.role == 'assistant']

        # Get first and last topics
        first_msg = messages[0].content[:100] if messages else ""
        last_msg = messages[-1].content[:100] if messages else ""

        summary = f"Conversa com {len(user_messages)} perguntas do usuário e " \
                 f"{len(assistant_messages)} respostas. " \
                 f"Iniciou com: {first_msg}... " \
                 f"Terminou com: {last_msg}..."

        return summary

    def _calculate_importance(self, messages: List[Message], topics: List[str]) -> float:
        """
        Calculate importance score for messages

        Args:
            messages: List of messages
            topics: Extracted topics

        Returns:
            Importance score (0.0 to 1.0)
        """
        score = 0.0

        # Base score from message count
        score += min(len(messages) / 50.0, 0.3)

        # Score from topic diversity
        score += min(len(topics) / 5.0, 0.3)

        # Score from message length (indicates detailed conversation)
        avg_length = sum(len(m.content) for m in messages) / len(messages) if messages else 0
        score += min(avg_length / 500.0, 0.2)

        # Score from recent activity
        if messages:
            time_diff = datetime.now() - messages[-1].timestamp
            if time_diff < timedelta(hours=1):
                score += 0.2

        return min(score, 1.0)

    async def get_relevant_memories(self,
                                   session_id: str,
                                   query_topics: Optional[List[str]] = None,
                                   limit: int = 5) -> List[MemorySummary]:
        """
        Get relevant long-term memories

        Args:
            session_id: Session identifier
            query_topics: Topics to search for
            limit: Maximum number of memories to return

        Returns:
            List of relevant memory summaries
        """
        if session_id not in self.summaries:
            return []

        memories = self.summaries[session_id]

        if not query_topics:
            # Return most recent and important memories
            sorted_memories = sorted(memories,
                                   key=lambda m: (m.importance_score, m.created_at),
                                   reverse=True)
            return sorted_memories[:limit]

        # Filter by topic relevance
        relevant_memories = []
        for memory in memories:
            relevance_score = len(set(memory.topics) & set(query_topics)) / len(query_topics)
            if relevance_score > 0:
                relevant_memories.append((memory, relevance_score))

        # Sort by relevance and importance
        relevant_memories.sort(key=lambda x: (x[1], x[0].importance_score), reverse=True)

        return [memory for memory, _ in relevant_memories[:limit]]

    async def build_long_term_context(self,
                                     session_id: str,
                                     current_topics: Optional[List[str]] = None) -> str:
        """
        Build context from long-term memories

        Args:
            session_id: Session identifier
            current_topics: Current conversation topics

        Returns:
            Formatted long-term context
        """
        memories = await self.get_relevant_memories(
            session_id=session_id,
            query_topics=current_topics,
            limit=3
        )

        if not memories:
            return ""

        context_parts = ["Contexto de conversas anteriores:"]

        for i, memory in enumerate(memories, 1):
            context_parts.append(
                f"{i}. {memory.summary} (Tópicos: {', '.join(memory.topics)})"
            )

        return "\\n".join(context_parts)

    async def _cleanup_old_summaries(self, session_id: str) -> None:
        """
        Clean up old summaries if over limit

        Args:
            session_id: Session identifier
        """
        if session_id not in self.summaries:
            return

        summaries = self.summaries[session_id]

        if len(summaries) > self.max_summaries:
            # Keep the most important and recent summaries
            summaries.sort(key=lambda s: (s.importance_score, s.created_at), reverse=True)
            self.summaries[session_id] = summaries[:self.max_summaries]

            logger.debug(f"Cleaned up old summaries for {session_id}")

    async def clear_session_memories(self, session_id: str) -> bool:
        """
        Clear long-term memories for a session

        Args:
            session_id: Session identifier

        Returns:
            True if cleared successfully
        """
        if session_id in self.summaries:
            del self.summaries[session_id]
            logger.info(f"Cleared long-term memories for {session_id}")
            return True
        return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get long-term memory statistics"""
        total_summaries = sum(len(summaries) for summaries in self.summaries.values())
        avg_importance = 0.0

        if total_summaries > 0:
            total_importance = sum(
                sum(s.importance_score for s in summaries)
                for summaries in self.summaries.values()
            )
            avg_importance = total_importance / total_summaries

        return {
            'total_sessions_with_memories': len(self.summaries),
            'total_summaries': total_summaries,
            'average_importance_score': avg_importance,
            'summarization_threshold': self.summarization_threshold,
            'max_summaries': self.max_summaries,
            'memory_type': 'long_term'
        }