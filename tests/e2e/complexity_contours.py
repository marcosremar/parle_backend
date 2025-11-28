"""
Complexity Contours
Implements sliding window complexity calculation for CEFR classification
Based on: Vajjala & Rama (2021) - RNN with complexity contours
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter


class ComplexityContoursCalculator:
    """Calculate complexity contours using sliding windows"""
    
    def __init__(self, window_size: int = 50):
        """
        Initialize calculator
        
        Args:
            window_size: Size of sliding window in words (default 50)
        """
        self.window_size = window_size
    
    def calculate_contours(self, text: str) -> Dict[str, Any]:
        """
        Calculate complexity contours for entire text
        
        Extracts 6 features per window:
        1. Lexical diversity (TTR)
        2. Syntactic depth (average sentence length)
        3. Word length (average characters)
        4. Subordination (subordinating conjunctions)
        5. Rare words (words not in top 1000)
        6. Sentence length (words per sentence)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with contour data and statistics
        """
        words = self._tokenize(text)
        
        if len(words) < self.window_size:
            # For short texts, return single window
            return self._calculate_single_window(words, text)
        
        # Sliding window analysis
        contours = []
        num_windows = len(words) - self.window_size + 1
        
        for i in range(num_windows):
            window_words = words[i:i + self.window_size]
            window_text = " ".join(window_words)
            
            features = self._extract_window_features(window_words, window_text)
            contours.append({
                "position": i,
                "features": features,
                "complexity_score": self._calculate_complexity_score(features)
            })
        
        # Statistical measures
        complexity_scores = [c["complexity_score"] for c in contours]
        
        # Enhanced statistics (Vajjala & Rama, 2021)
        if complexity_scores:
            mean_complexity = float(np.mean(complexity_scores))
            std_complexity = float(np.std(complexity_scores))
            variance = float(np.var(complexity_scores))
            trend = self._calculate_trend(complexity_scores)
            
            # Calculate variance and coefficient of variation
            cv = std_complexity / mean_complexity if mean_complexity > 0 else 0.0
            
            # Calculate range and interquartile range
            sorted_scores = sorted(complexity_scores)
            q1_idx = len(sorted_scores) // 4
            q3_idx = 3 * len(sorted_scores) // 4
            iqr = sorted_scores[q3_idx] - sorted_scores[q1_idx] if q3_idx > q1_idx else 0.0
        else:
            mean_complexity = std_complexity = variance = trend = cv = iqr = 0.0
        
        return {
            "contours": contours,
            "num_windows": num_windows,
            "window_size": self.window_size,
            "statistics": {
                "mean_complexity": mean_complexity,
                "std_complexity": std_complexity,
                "variance": variance,
                "min_complexity": float(np.min(complexity_scores)) if complexity_scores else 0.0,
                "max_complexity": float(np.max(complexity_scores)) if complexity_scores else 0.0,
                "range": float(np.max(complexity_scores) - np.min(complexity_scores)) if complexity_scores else 0.0,
                "trend": trend,
                "coefficient_of_variation": cv,
                "interquartile_range": iqr
            },
            "features_summary": self._summarize_features(contours)
        }
    
    def _calculate_single_window(self, words: List[str], text: str) -> Dict[str, Any]:
        """Calculate features for a single window (short text)"""
        features = self._extract_window_features(words, text)
        complexity_score = self._calculate_complexity_score(features)
        
        return {
            "contours": [{
                "position": 0,
                "features": features,
                "complexity_score": complexity_score
            }],
            "num_windows": 1,
            "window_size": len(words),
            "statistics": {
                "mean_complexity": complexity_score,
                "std_complexity": 0.0,
                "min_complexity": complexity_score,
                "max_complexity": complexity_score,
                "trend": 0.0
            },
            "features_summary": features
        }
    
    def _extract_window_features(self, words: List[str], window_text: str) -> Dict[str, float]:
        """
        Extract 6 features for a window
        
        Returns:
            Dictionary with feature values
        """
        # 1. Lexical diversity (TTR)
        unique_words = len(set(words))
        ttr = unique_words / len(words) if words else 0.0
        
        # 2. Syntactic depth (average sentence length)
        sentences = re.split(r'[.!?]+', window_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        else:
            avg_sentence_length = len(words)
        
        # 3. Word length (average characters)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0.0
        
        # 4. Subordination (subordinating conjunctions per sentence)
        subordinating_conjunctions = [
            'porque', 'quando', 'se', 'embora', 'enquanto', 'conforme',
            'já que', 'uma vez que', 'caso', 'mesmo que', 'apesar de',
            'ainda que', 'conquanto', 'visto que'
        ]
        subord_count = sum(1 for conj in subordinating_conjunctions
                          if re.search(rf'\b{conj}\b', window_text.lower()))
        subordination_density = subord_count / len(sentences) if sentences else 0.0
        
        # 5. Rare words (words not in top 1000 most common Portuguese words)
        # Simple heuristic: words longer than 7 characters or not in common list
        common_words = {
            'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
            'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
            'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem',
            'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos',
            'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso',
            'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter',
            'seus', 'suas', 'numa', 'pelos', 'pelas', 'havia', 'seja', 'qual',
            'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas',
            'pelos', 'pelas', 'pelas', 'pelas', 'pelas', 'pelas', 'pelas'
        }
        rare_words = [w for w in words if w.lower() not in common_words and len(w) > 5]
        rare_word_ratio = len(rare_words) / len(words) if words else 0.0
        
        # 6. Sentence length (already calculated as avg_sentence_length)
        
        return {
            "lexical_diversity": round(ttr, 3),
            "syntactic_depth": round(avg_sentence_length, 2),
            "word_length": round(avg_word_length, 2),
            "subordination": round(subordination_density, 3),
            "rare_words": round(rare_word_ratio, 3),
            "sentence_length": round(avg_sentence_length, 2)
        }
    
    def _calculate_complexity_score(self, features: Dict[str, float]) -> float:
        """
        Calculate overall complexity score from features
        
        Higher score = more complex
        
        Args:
            features: Dictionary with 6 feature values
            
        Returns:
            Complexity score (0-1)
        """
        # Normalize each feature to 0-1 range and weight them
        weights = {
            "lexical_diversity": 0.20,  # TTR
            "syntactic_depth": 0.25,     # Sentence length
            "word_length": 0.15,         # Word length
            "subordination": 0.20,      # Subordination
            "rare_words": 0.15,          # Rare words
            "sentence_length": 0.05     # Redundant with syntactic_depth, lower weight
        }
        
        # Normalize features (rough ranges based on typical values)
        normalized = {}
        
        # TTR: typically 0.3-0.8
        normalized["lexical_diversity"] = min(1.0, max(0.0, 
            (features["lexical_diversity"] - 0.3) / 0.5))
        
        # Sentence length: typically 5-25 words
        normalized["syntactic_depth"] = min(1.0, max(0.0,
            (features["syntactic_depth"] - 5) / 20))
        
        # Word length: typically 4-8 characters
        normalized["word_length"] = min(1.0, max(0.0,
            (features["word_length"] - 4) / 4))
        
        # Subordination: typically 0-1 per sentence
        normalized["subordination"] = min(1.0, max(0.0, features["subordination"]))
        
        # Rare words: typically 0-0.3
        normalized["rare_words"] = min(1.0, max(0.0, features["rare_words"] / 0.3))
        
        # Sentence length (same as syntactic_depth)
        normalized["sentence_length"] = normalized["syntactic_depth"]
        
        # Weighted sum
        score = sum(weights[key] * normalized.get(key, 0.0) for key in weights)
        
        return round(score, 3)
    
    def _calculate_trend(self, complexity_scores: List[float]) -> float:
        """
        Calculate linear trend (slope) of complexity over text
        
        Positive = increasing complexity
        Negative = decreasing complexity
        
        Args:
            complexity_scores: List of complexity scores
            
        Returns:
            Trend slope
        """
        if len(complexity_scores) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(complexity_scores)
        x = np.arange(n)
        y = np.array(complexity_scores)
        
        # Slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        return float(slope)
    
    def _summarize_features(self, contours: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize features across all contours"""
        if not contours:
            return {}
        
        all_features = [c["features"] for c in contours]
        
        summary = {}
        for feature_name in all_features[0].keys():
            values = [f[feature_name] for f in all_features]
            summary[f"{feature_name}_mean"] = float(np.mean(values))
            summary[f"{feature_name}_std"] = float(np.std(values))
        
        return summary
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words

