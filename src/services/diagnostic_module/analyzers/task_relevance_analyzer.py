"""
Task Relevance Analyzer
Assesses task relevance using SBERT embeddings.
Based on Lu et al. (2025) and Reimers & Gurevych (2019) - Sentence-BERT.
"""

from typing import Dict, Optional
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class TaskRelevanceAnalyzer:
    """
    Assess task relevance using SBERT embeddings.
    
    Based on:
    - Lu et al. (2025): Task relevance assessment for speaking evaluation
    - Reimers & Gurevych (2019): Sentence-BERT for semantic similarity
    """
    
    def __init__(self):
        """Initialize SBERT model"""
        if not SBERT_AVAILABLE:
            self.model = None
            logger.warning("SBERT not available. Task relevance will be disabled.")
            return
        
        try:
            # Use Portuguese BERT model
            self.model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
            logger.info("Loaded SBERT model: neuralmind/bert-base-portuguese-cased")
        except Exception as e:
            logger.error(f"Failed to load SBERT model: {e}")
            self.model = None
    
    def calculate_task_relevance(
        self,
        question: str,
        response: str,
        exemplar: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate task relevance and exemplar similarity.
        
        Args:
            question: Task/question prompt
            response: Student's response
            exemplar: Exemplar response (optional)
            
        Returns:
            Dictionary with:
            - task_relevance: Cosine similarity between question and response (0-1)
            - exemplar_similarity: Cosine similarity with exemplar (0-1, if provided)
        """
        if self.model is None:
            logger.warning("SBERT model not available, returning default values")
            return {
                "task_relevance": 0.5,
                "exemplar_similarity": 0.5 if exemplar else 0.0
            }
        
        try:
            # Encode texts
            question_emb = self.model.encode([question], convert_to_numpy=True)
            response_emb = self.model.encode([response], convert_to_numpy=True)
            
            # Task relevance (cosine similarity)
            task_relevance = cosine_similarity(question_emb, response_emb)[0][0]
            
            result = {
                "task_relevance": float(task_relevance),
                "exemplar_similarity": 0.0  # Default when no exemplar
            }
            
            # Exemplar similarity if provided
            if exemplar:
                exemplar_emb = self.model.encode([exemplar], convert_to_numpy=True)
                exemplar_similarity = cosine_similarity(
                    response_emb, 
                    exemplar_emb
                )[0][0]
                result["exemplar_similarity"] = float(exemplar_similarity)
            else:
                result["exemplar_similarity"] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating task relevance: {e}")
            return {
                "task_relevance": 0.5,
                "exemplar_similarity": 0.5 if exemplar else None
            }
    
    def detect_off_topic(
        self,
        question: str,
        response: str,
        threshold: float = 0.3
    ) -> Dict[str, any]:
        """
        Detect if response is off-topic.
        
        Args:
            question: Task/question prompt
            response: Student's response
            threshold: Minimum similarity to be considered on-topic (default: 0.3)
            
        Returns:
            Dictionary with:
            - is_off_topic: Boolean
            - similarity: Task relevance score
            - explanation: Explanation of the result
        """
        relevance = self.calculate_task_relevance(question, response)
        similarity = relevance["task_relevance"]
        
        is_off_topic = similarity < threshold
        
        explanation = (
            f"Resposta {'fora do tópico' if is_off_topic else 'relevante'} "
            f"(similaridade: {similarity:.2f})"
        )
        
        return {
            "is_off_topic": is_off_topic,
            "similarity": similarity,
            "explanation": explanation
        }
    
    def calculate_topic_coverage(
        self,
        question: str,
        response: str
    ) -> Dict[str, any]:
        """
        Calculate how well the response covers the topic.
        
        Args:
            question: Task/question prompt
            response: Student's response
            
        Returns:
            Dictionary with coverage metrics
        """
        relevance = self.calculate_task_relevance(question, response)
        
        # Simple coverage based on similarity
        # Higher similarity = better coverage
        coverage_score = relevance["task_relevance"]
        
        # Categorize coverage
        if coverage_score >= 0.7:
            coverage_level = "excellent"
        elif coverage_score >= 0.5:
            coverage_level = "good"
        elif coverage_score >= 0.3:
            coverage_level = "partial"
        else:
            coverage_level = "poor"
        
        return {
            "coverage_score": coverage_score,
            "coverage_level": coverage_level,
            "task_relevance": relevance["task_relevance"]
        }
