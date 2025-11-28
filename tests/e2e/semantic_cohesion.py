"""
LSA Semantic Cohesion
Implements Latent Semantic Analysis for measuring semantic cohesion
Based on: Vajjala & Rama (2021), Arnold et al. (2018)
"""

import re
from typing import List, Dict, Any
import numpy as np

# Try to import scikit-learn (optional)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SemanticCohesionAnalyzer:
    """Analyze semantic cohesion using LSA"""
    
    def __init__(self, n_components: int = 50):
        """
        Initialize analyzer
        
        Args:
            n_components: Number of dimensions for LSA (default 50)
        """
        self.n_components = n_components
    
    def calculate_cohesion(self, text: str) -> Dict[str, Any]:
        """
        Calculate semantic cohesion using LSA
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with cohesion metrics
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_cohesion(text)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return {
                "mean_cohesion": 0.0,
                "cohesion_variance": 0.0,
                "adjacent_cohesion": 0.0,
                "non_adjacent_cohesion": 0.0,
                "num_sentences": len(sentences)
            }
        
        try:
            # Vectorize sentences
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            sentence_vectors = vectorizer.fit_transform(sentences)
            
            # Apply LSA
            lsa = TruncatedSVD(n_components=min(self.n_components, len(sentences) - 1))
            lsa_vectors = lsa.fit_transform(sentence_vectors)
            
            # Calculate cosine similarity between sentences
            similarities = cosine_similarity(lsa_vectors)
            
            # Extract metrics
            # Mean cohesion (excluding diagonal)
            mask = ~np.eye(len(similarities), dtype=bool)
            mean_cohesion = float(np.mean(similarities[mask]))
            
            # Cohesion variance
            cohesion_variance = float(np.var(similarities[mask]))
            
            # Adjacent sentence cohesion
            adjacent_similarities = []
            for i in range(len(similarities) - 1):
                adjacent_similarities.append(similarities[i, i + 1])
            adjacent_cohesion = float(np.mean(adjacent_similarities)) if adjacent_similarities else 0.0
            
            # Non-adjacent sentence cohesion
            non_adjacent_similarities = []
            for i in range(len(similarities)):
                for j in range(i + 2, len(similarities)):
                    non_adjacent_similarities.append(similarities[i, j])
            non_adjacent_cohesion = float(np.mean(non_adjacent_similarities)) if non_adjacent_similarities else 0.0
            
            return {
                "mean_cohesion": round(mean_cohesion, 3),
                "cohesion_variance": round(cohesion_variance, 3),
                "adjacent_cohesion": round(adjacent_cohesion, 3),
                "non_adjacent_cohesion": round(non_adjacent_cohesion, 3),
                "num_sentences": len(sentences),
                "lsa_components": self.n_components
            }
        except Exception as e:
            print(f"⚠️ Error calculating LSA cohesion: {e}")
            return self._fallback_cohesion(text)
    
    def _fallback_cohesion(self, text: str) -> Dict[str, Any]:
        """Fallback cohesion calculation when scikit-learn is not available"""
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return {
                "mean_cohesion": 0.0,
                "cohesion_variance": 0.0,
                "adjacent_cohesion": 0.0,
                "non_adjacent_cohesion": 0.0,
                "num_sentences": len(sentences),
                "method": "fallback"
            }
        
        # Simple word overlap between sentences
        sentence_words = [set(re.findall(r'\b\w+\b', s.lower())) for s in sentences]
        
        similarities = []
        for i in range(len(sentence_words)):
            for j in range(i + 1, len(sentence_words)):
                words1 = sentence_words[i]
                words2 = sentence_words[j]
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    similarities.append(overlap)
        
        mean_cohesion = float(np.mean(similarities)) if similarities else 0.0
        
        return {
            "mean_cohesion": round(mean_cohesion, 3),
            "cohesion_variance": 0.0,
            "adjacent_cohesion": round(mean_cohesion, 3),
            "non_adjacent_cohesion": round(mean_cohesion, 3),
            "num_sentences": len(sentences),
            "method": "fallback_word_overlap"
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

