"""
Semantic Cache - Similarity-based cache for LLM responses

Caches LLM responses and retrieves them based on semantic similarity,
not just exact matches. Uses sentence embeddings to find similar queries.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic similarity-based cache for LLM responses

    Features:
    - Similarity threshold (default 0.85)
    - Automatic embedding generation
    - Fast cosine similarity search
    - Fallback to exact match if embeddings unavailable
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_entries: int = 1000,
        model_name: str = "all-MiniLM-L6-v2",
        enable_semantic: bool = True,
    ):
        """
        Initialize semantic cache

        Args:
            similarity_threshold: Minimum similarity for cache hit (0-1)
            max_entries: Maximum number of cached entries
            model_name: Sentence transformer model name
            enable_semantic: Enable semantic matching (disable for exact-match only)
        """
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.enable_semantic = enable_semantic

        # Cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}
        # Key: hash, Value: {prompt, response, embedding, timestamp, hits}

        # Sentence transformer model (lazy loading)
        self._model: Optional[SentenceTransformer] = None
        self._model_name = model_name

        # Statistics
        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "sets": 0,
            "avg_similarity": 0.0,
        }

        if enable_semantic and not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "⚠️  sentence-transformers not available. "
                "Semantic cache will use exact match only. "
                "Install with: pip install sentence-transformers"
            )
            self.enable_semantic = False
        elif enable_semantic:
            logger.info(f"✅ Semantic cache enabled (threshold={similarity_threshold})")

    def _load_model(self) -> bool:
        """
        Lazy load sentence transformer model

        Returns:
            True if model loaded successfully
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False

        if self._model is not None:
            return True

        try:
            logger.info(f"📥 Loading sentence transformer: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("✅ Sentence transformer loaded")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
            return False

    def _get_embedding(self, text: str) -> Optional[Any]:
        """
        Get embedding for text

        Args:
            text: Input text

        Returns:
            Embedding vector or None
        """
        if not self.enable_semantic:
            return None

        if not self._load_model():
            return None

        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"❌ Embedding generation error: {e}")
            return None

    def _cosine_similarity(self, a: Any, b: Any) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (0-1)
        """
        if not NUMPY_AVAILABLE or np is None:
            return 0.0

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        **kwargs
    ) -> Optional[Tuple[str, float, str]]:
        """
        Get cached LLM response

        Args:
            prompt: User prompt
            context: Optional conversation context
            **kwargs: Additional parameters (model, temperature, etc.)

        Returns:
            Tuple of (response, similarity, match_type) or None
            match_type: "exact" or "semantic"
        """
        # Generate cache key
        cache_key = self._make_key(prompt, context, **kwargs)

        # Try exact match first
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            entry["hits"] += 1
            self._stats["exact_hits"] += 1
            logger.debug(f"✅ Exact cache hit: {prompt[:50]}...")
            return (entry["response"], 1.0, "exact")

        # Try semantic match if enabled
        if self.enable_semantic and len(self._cache) > 0:
            query_embedding = self._get_embedding(prompt)

            if query_embedding is not None:
                best_match = None
                best_similarity = 0.0

                # Search for similar prompts
                for key, entry in self._cache.items():
                    if entry.get("embedding") is None:
                        continue

                    similarity = self._cosine_similarity(
                        query_embedding,
                        entry["embedding"]
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry

                # Check if best match exceeds threshold
                if best_match and best_similarity >= self.similarity_threshold:
                    best_match["hits"] += 1
                    self._stats["semantic_hits"] += 1
                    self._stats["avg_similarity"] = (
                        (self._stats["avg_similarity"] * (self._stats["semantic_hits"] - 1) + best_similarity) /
                        self._stats["semantic_hits"]
                    )

                    logger.debug(
                        f"✅ Semantic cache hit: {prompt[:50]}... "
                        f"(similarity={best_similarity:.3f})"
                    )
                    return (best_match["response"], best_similarity, "semantic")

        # Cache miss
        self._stats["misses"] += 1
        return None

    def set(
        self,
        prompt: str,
        response: str,
        context: Optional[List[Dict]] = None,
        **kwargs
    ) -> bool:
        """
        Cache LLM response

        Args:
            prompt: User prompt
            response: LLM response
            context: Optional conversation context
            **kwargs: Additional parameters (model, temperature, etc.)

        Returns:
            True if successfully cached
        """
        # Check max entries limit
        if len(self._cache) >= self.max_entries:
            # Evict least used entry
            self._evict_least_used()

        # Generate cache key
        cache_key = self._make_key(prompt, context, **kwargs)

        # Get embedding if semantic cache enabled
        embedding = None
        if self.enable_semantic:
            embedding = self._get_embedding(prompt)

        # Store entry
        self._cache[cache_key] = {
            "prompt": prompt,
            "response": response,
            "context": context,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
            "hits": 0,
            "kwargs": kwargs,
        }

        self._stats["sets"] += 1
        logger.debug(f"✅ Cached LLM response: {prompt[:50]}...")
        return True

    def _evict_least_used(self) -> None:
        """Evict least recently used entry"""
        if not self._cache:
            return

        # Find entry with fewest hits
        min_hits = float('inf')
        evict_key = None

        for key, entry in self._cache.items():
            if entry["hits"] < min_hits:
                min_hits = entry["hits"]
                evict_key = key

        if evict_key:
            del self._cache[evict_key]
            logger.debug(f"🗑️  Evicted LLM cache entry (hits={min_hits})")

    def clear(self) -> None:
        """Clear all cached entries"""
        self._cache.clear()
        logger.info("🧹 Semantic cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache stats
        """
        total_requests = self._stats["exact_hits"] + self._stats["semantic_hits"] + self._stats["misses"]
        total_hits = self._stats["exact_hits"] + self._stats["semantic_hits"]
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            **self._stats,
            "total_entries": len(self._cache),
            "max_entries": self.max_entries,
            "hit_rate": round(hit_rate, 2),
            "semantic_enabled": self.enable_semantic,
            "similarity_threshold": self.similarity_threshold,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters"""
        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "sets": 0,
            "avg_similarity": 0.0,
        }

    def _make_key(
        self,
        prompt: str,
        context: Optional[List[Dict]],
        **kwargs
    ) -> str:
        """
        Generate cache key

        Args:
            prompt: User prompt
            context: Conversation context
            **kwargs: Additional parameters

        Returns:
            Hashed cache key
        """
        # Create key data
        key_data = {
            "prompt": prompt,
            "context": context or [],
            "kwargs": sorted(kwargs.items())
        }

        key_str = json.dumps(key_data, sort_keys=True, default=str)

        # Hash for consistent key length
        return hashlib.sha256(key_str.encode()).hexdigest()
