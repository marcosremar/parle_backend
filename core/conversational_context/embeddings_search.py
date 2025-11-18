"""
Embeddings Memory Search - Semantic search for conversation memories
Handles embedding generation, storage, and semantic retrieval of conversation content
With optimized caching and incremental updates for performance
"""

import logging
import numpy as np
import json
import pickle
import hashlib
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

from .memory_store import ConversationMemoryStorage, Message

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingEntry:
    """Represents an embedded memory entry"""
    session_id: str
    content: str
    embedding: List[float]
    timestamp: datetime
    message_type: str  # 'user', 'assistant', 'summary'
    content_hash: str  # SHA-256 hash for caching
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create_with_hash(cls, session_id: str, content: str, embedding: List[float],
                        timestamp: datetime, message_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Create entry with content hash for caching"""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return cls(
            session_id=session_id,
            content=content,
            embedding=embedding,
            timestamp=timestamp,
            message_type=message_type,
            content_hash=content_hash,
            metadata=metadata
        )


class EmbeddingsMemorySearch:
    """
    Embeddings-based semantic search for conversation memories
    Uses vector similarity to find relevant conversation context
    With optimized caching, incremental updates, and persistence
    """

    def __init__(self,
                 embedding_dim: int = 384,  # Sentence-BERT dimension
                 max_embeddings: int = 1000,
                 similarity_threshold: float = 0.7,
                 cache_similarity_threshold: float = 0.95,  # For approximate cache hits
                 cache_dir: str = "data/embedding_cache",
                 enable_persistence: bool = True,
                 cache_ttl_hours: int = 24,
                 memory_store: Optional[ConversationMemoryStorage] = None):
        """
        Initialize embeddings memory search with caching

        Args:
            embedding_dim: Dimension of embedding vectors
            max_embeddings: Maximum number of embeddings to store
            similarity_threshold: Minimum similarity score for retrieval
            cache_similarity_threshold: Similarity threshold for cache hits
            cache_dir: Directory for persistent cache storage
            enable_persistence: Whether to enable disk persistence
            cache_ttl_hours: TTL for cache entries in hours
            memory_store: Optional custom memory store
        """
        self.embedding_dim = embedding_dim
        self.max_embeddings = max_embeddings
        self.similarity_threshold = similarity_threshold
        self.cache_similarity_threshold = cache_similarity_threshold
        self.cache_dir = Path(cache_dir)
        self.enable_persistence = enable_persistence
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.memory_store = memory_store or ConversationMemoryStorage()

        # Create cache directory
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage for embeddings
        self.embeddings: Dict[str, List[EmbeddingEntry]] = {}
        self.embedding_model = None

        # Hash-based cache for fast lookups
        self.embedding_cache: Dict[str, Tuple[List[float], datetime]] = {}

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_computations = 0

        # Load existing cache if available
        if self.enable_persistence:
            asyncio.create_task(self._load_cache())

        logger.info(f"🔍 EmbeddingsMemorySearch initialized (dim={embedding_dim}, "
                   f"max={max_embeddings}, threshold={similarity_threshold}, "
                   f"cache_threshold={cache_similarity_threshold}, persistence={enable_persistence})")

    def _initialize_embedding_model(self):
        """Initialize the embedding model (lazy loading)"""
        if self.embedding_model is not None:
            return

        try:
            # Try to import and initialize sentence-transformers
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence-BERT model loaded successfully")
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available, using dummy embeddings")
            self.embedding_model = "dummy"
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.embedding_model = "dummy"

    def _get_text_hash(self, text: str) -> str:
        """Generate SHA-256 hash for text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text with caching optimization

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        self.total_computations += 1
        text_hash = self._get_text_hash(text)

        # Check exact hash cache first
        if text_hash in self.embedding_cache:
            embedding, timestamp = self.embedding_cache[text_hash]
            # Check if cache entry is still valid
            if datetime.now() - timestamp < self.cache_ttl:
                self.cache_hits += 1
                logger.debug(f"Cache hit for text hash: {text_hash[:8]}...")
                return embedding
            else:
                # Remove expired entry
                del self.embedding_cache[text_hash]

        # Check for approximate matches (similar texts)
        similar_embedding = await self._find_similar_cached_embedding(text)
        if similar_embedding:
            self.cache_hits += 1
            return similar_embedding

        # Generate new embedding
        self.cache_misses += 1
        self._initialize_embedding_model()

        if self.embedding_model == "dummy":
            # Dummy embedding for testing
            np.random.seed(hash(text) % 2147483647)
            embedding = np.random.normal(0, 1, self.embedding_dim).tolist()
        else:
            try:
                embedding = self.embedding_model.encode(text, convert_to_tensor=False)
                embedding = embedding.tolist()
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                # Fallback to dummy
                np.random.seed(hash(text) % 2147483647)
                embedding = np.random.normal(0, 1, self.embedding_dim).tolist()

        # Cache the new embedding
        self.embedding_cache[text_hash] = (embedding, datetime.now())

        # Async save to disk
        if self.enable_persistence:
            asyncio.create_task(self._save_cache_async())

        logger.debug(f"Generated new embedding for text hash: {text_hash[:8]}...")
        return embedding

    async def _find_similar_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Find similar cached embedding for approximate matching"""
        if not self.embedding_cache:
            return None

        # For approximate matching, we need to compute similarity with cached texts
        # This is expensive, so we limit it to recent cache entries
        query_embedding = None

        # Only check recent entries to avoid performance hit
        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_entries = [(hash_key, (emb, ts)) for hash_key, (emb, ts) in self.embedding_cache.items()
                         if ts > recent_threshold]

        if len(recent_entries) > 50:  # Limit to 50 most recent
            recent_entries = sorted(recent_entries, key=lambda x: x[1][1], reverse=True)[:50]

        if not recent_entries:
            return None

        # Generate embedding for current text to compare
        self._initialize_embedding_model()
        if self.embedding_model == "dummy":
            np.random.seed(hash(text) % 2147483647)
            query_embedding = np.random.normal(0, 1, self.embedding_dim).tolist()
        else:
            try:
                query_embedding = self.embedding_model.encode(text, convert_to_tensor=False).tolist()
            except (RuntimeError, AttributeError, ValueError, TypeError):
                return None

        # Find most similar cached embedding
        best_similarity = 0
        best_embedding = None

        for _, (cached_embedding, _) in recent_entries:
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity >= self.cache_similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_embedding = cached_embedding

        if best_embedding:
            logger.debug(f"Found approximate cache match with similarity: {best_similarity:.3f}")

        return best_embedding

    async def _load_cache(self) -> None:
        """Load embedding cache from disk"""
        try:
            cache_file = self.cache_dir / "embedding_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)

                # Clean expired entries
                now = datetime.now()
                expired_keys = [k for k, (_, ts) in self.embedding_cache.items()
                               if now - ts > self.cache_ttl]
                for key in expired_keys:
                    del self.embedding_cache[key]

                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings from disk")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}

    async def _save_cache_async(self) -> None:
        """Save embedding cache to disk asynchronously"""
        try:
            cache_file = self.cache_dir / "embedding_cache.pkl"
            # Use a temporary file to avoid corruption
            temp_file = cache_file.with_suffix('.tmp')

            with open(temp_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)

            # Atomic rename
            temp_file.rename(cache_file)

        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    async def _cleanup_cache(self) -> None:
        """Clean up expired cache entries and manage memory pressure"""
        now = datetime.now()

        # Remove expired entries
        expired_keys = [k for k, (_, ts) in self.embedding_cache.items()
                       if now - ts > self.cache_ttl]
        for key in expired_keys:
            del self.embedding_cache[key]

        # If still too many entries, remove oldest ones
        if len(self.embedding_cache) > self.max_embeddings * 2:  # 2x limit for cache
            sorted_entries = sorted(self.embedding_cache.items(),
                                  key=lambda x: x[1][1])  # Sort by timestamp
            keep_count = self.max_embeddings

            # Keep only the most recent entries
            self.embedding_cache = dict(sorted_entries[-keep_count:])

        logger.debug(f"Cache cleanup: {len(self.embedding_cache)} entries remaining")

    async def add_message_embedding(self,
                                   session_id: str,
                                   content: str,
                                   message_type: str,
                                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message embedding to the search index with caching optimization

        Args:
            session_id: Session identifier
            content: Message content to embed
            message_type: Type of message ('user', 'assistant', 'summary')
            metadata: Optional metadata
        """
        # Generate embedding with caching
        embedding = await self._generate_embedding(content)

        # Create embedding entry with hash
        entry = EmbeddingEntry.create_with_hash(
            session_id=session_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.now(),
            message_type=message_type,
            metadata=metadata or {}
        )

        # Store embedding
        if session_id not in self.embeddings:
            self.embeddings[session_id] = []

        self.embeddings[session_id].append(entry)

        # Cleanup if too many embeddings
        await self._cleanup_old_embeddings(session_id)

        # Periodic cache cleanup
        if self.total_computations % 100 == 0:  # Every 100 computations
            await self._cleanup_cache()

        logger.debug(f"Added embedding for {session_id}: {message_type} message")

    async def add_interaction_embeddings(self,
                                        session_id: str,
                                        user_input: str,
                                        assistant_response: str,
                                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add embeddings for a complete user-assistant interaction

        Args:
            session_id: Session identifier
            user_input: User's input
            assistant_response: Assistant's response
            metadata: Optional metadata
        """
        # Add user message embedding
        await self.add_message_embedding(
            session_id=session_id,
            content=user_input,
            message_type='user',
            metadata=metadata
        )

        # Add assistant message embedding
        await self.add_message_embedding(
            session_id=session_id,
            content=assistant_response,
            message_type='assistant',
            metadata=metadata
        )

    async def search_similar_content(self,
                                    session_id: str,
                                    query: str,
                                    top_k: int = 5,
                                    message_types: Optional[List[str]] = None) -> List[Tuple[EmbeddingEntry, float]]:
        """
        Search for similar content using semantic similarity with caching

        Args:
            session_id: Session identifier
            query: Query text to search for
            top_k: Maximum number of results to return
            message_types: Optional filter for message types

        Returns:
            List of (embedding_entry, similarity_score) tuples
        """
        if session_id not in self.embeddings:
            return []

        # Generate query embedding with caching
        query_embedding = await self._generate_embedding(query)

        # Calculate similarities
        similarities = []
        for entry in self.embeddings[session_id]:
            # Filter by message type if specified
            if message_types and entry.message_type not in message_types:
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, entry.embedding)

            if similarity >= self.similarity_threshold:
                similarities.append((entry, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    async def find_relevant_context(self,
                                   session_id: str,
                                   current_input: str,
                                   max_results: int = 3) -> List[str]:
        """
        Find relevant context for current input

        Args:
            session_id: Session identifier
            current_input: Current user input
            max_results: Maximum context results to return

        Returns:
            List of relevant context strings
        """
        similar_entries = await self.search_similar_content(
            session_id=session_id,
            query=current_input,
            top_k=max_results
        )

        relevant_context = []
        for entry, similarity in similar_entries:
            context_text = f"[{entry.message_type.title()}] {entry.content[:200]}..."
            relevant_context.append(context_text)

        return relevant_context

    async def build_semantic_context(self,
                                    session_id: str,
                                    current_input: str) -> str:
        """
        Build semantic context from similar conversations

        Args:
            session_id: Session identifier
            current_input: Current user input

        Returns:
            Formatted semantic context
        """
        relevant_contexts = await self.find_relevant_context(
            session_id=session_id,
            current_input=current_input,
            max_results=3
        )

        if not relevant_contexts:
            return ""

        context_parts = ["Contexto semanticamente relevante:"]
        for i, context in enumerate(relevant_contexts, 1):
            context_parts.append(f"{i}. {context}")

        return "\\n".join(context_parts)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return dot_product / (norm_v1 * norm_v2)

    async def _cleanup_old_embeddings(self, session_id: str) -> None:
        """
        Clean up old embeddings if over limit

        Args:
            session_id: Session identifier
        """
        if session_id not in self.embeddings:
            return

        embeddings = self.embeddings[session_id]

        if len(embeddings) > self.max_embeddings:
            # Keep the most recent embeddings
            embeddings.sort(key=lambda e: e.timestamp, reverse=True)
            self.embeddings[session_id] = embeddings[:self.max_embeddings]

            logger.debug(f"Cleaned up old embeddings for {session_id}")

    async def get_embedding_clusters(self,
                                    session_id: str,
                                    num_clusters: int = 5) -> List[List[EmbeddingEntry]]:
        """
        Cluster embeddings to find conversation topics

        Args:
            session_id: Session identifier
            num_clusters: Number of clusters to create

        Returns:
            List of embedding clusters
        """
        if session_id not in self.embeddings:
            return []

        embeddings = self.embeddings[session_id]
        if len(embeddings) < num_clusters:
            return [[entry] for entry in embeddings]

        try:
            from sklearn.cluster import KMeans

            # Extract embedding vectors
            vectors = np.array([entry.embedding for entry in embeddings])

            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors)

            # Group by clusters
            clusters = [[] for _ in range(num_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(embeddings[i])

            return clusters

        except ImportError:
            logger.warning("sklearn not available for clustering")
            # Simple grouping by timestamp
            chunk_size = len(embeddings) // num_clusters
            return [embeddings[i:i + chunk_size] for i in range(0, len(embeddings), chunk_size)]

    async def clear_session_embeddings(self, session_id: str) -> bool:
        """
        Clear embeddings for a session

        Args:
            session_id: Session identifier

        Returns:
            True if cleared successfully
        """
        if session_id in self.embeddings:
            del self.embeddings[session_id]
            logger.info(f"Cleared embeddings for {session_id}")
            return True
        return False

    def get_embedding_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get embedding statistics with performance metrics

        Args:
            session_id: Optional specific session, or all sessions

        Returns:
            Statistics dictionary with cache performance
        """
        # Calculate cache hit rate
        cache_hit_rate = (self.cache_hits / self.total_computations * 100) if self.total_computations > 0 else 0

        if session_id:
            if session_id not in self.embeddings:
                return {'error': 'Session not found'}

            embeddings = self.embeddings[session_id]
            message_types = {}
            for entry in embeddings:
                message_types[entry.message_type] = message_types.get(entry.message_type, 0) + 1

            return {
                'session_id': session_id,
                'total_embeddings': len(embeddings),
                'message_types': message_types,
                'embedding_dim': self.embedding_dim,
                'memory_type': 'embeddings',
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cached_embeddings': len(self.embedding_cache)
            }

        # Global statistics
        total_sessions = len(self.embeddings)
        total_embeddings = sum(len(embs) for embs in self.embeddings.values())

        all_message_types = {}
        for embeddings in self.embeddings.values():
            for entry in embeddings:
                all_message_types[entry.message_type] = all_message_types.get(entry.message_type, 0) + 1

        return {
            'total_sessions': total_sessions,
            'total_embeddings': total_embeddings,
            'message_types': all_message_types,
            'embedding_dim': self.embedding_dim,
            'max_embeddings': self.max_embeddings,
            'similarity_threshold': self.similarity_threshold,
            'cache_similarity_threshold': self.cache_similarity_threshold,
            'memory_type': 'embeddings',
            'performance': {
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'total_computations': self.total_computations,
                'cached_embeddings': len(self.embedding_cache),
                'persistence_enabled': self.enable_persistence
            }
        }

    def get_cache_performance(self) -> Dict[str, Any]:
        """Get detailed cache performance metrics"""
        cache_hit_rate = (self.cache_hits / self.total_computations * 100) if self.total_computations > 0 else 0

        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_computations': self.total_computations,
            'cached_embeddings': len(self.embedding_cache),
            'cache_efficiency': 'excellent' if cache_hit_rate > 80 else 'good' if cache_hit_rate > 60 else 'needs_improvement',
            'estimated_time_saved_seconds': self.cache_hits * 0.1  # Assume 100ms saved per cache hit
        }