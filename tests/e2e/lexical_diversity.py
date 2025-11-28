"""
Advanced Lexical Diversity Metrics
Implements MTLD, MATTR, Zipf-normalized TTR, and hapax legomena
Based on: Vajjala & Rama (2021), NILC-Metrix (2022)
"""

import re
from typing import List, Dict, Any
from collections import Counter
import math


class LexicalDiversityCalculator:
    """Calculate advanced lexical diversity metrics"""
    
    def __init__(self):
        """Initialize calculator"""
        pass
    
    def calculate_mtld(self, text: str, threshold: float = 0.72) -> Dict[str, Any]:
        """
        Calculate MTLD (Measure of Textual Lexical Diversity)
        
        MTLD uses a sliding window approach and is robust to text length.
        It measures the average number of words needed to reach a TTR threshold.
        
        Args:
            text: Input text
            threshold: TTR threshold (default 0.72)
            
        Returns:
            Dictionary with MTLD metrics
        """
        words = self._tokenize(text)
        if len(words) < 2:
            return {
                "mtld": 0.0,
                "mtld_forward": 0.0,
                "mtld_backward": 0.0,
                "num_words": len(words)
            }
        
        # Forward MTLD
        mtld_forward = self._calculate_mtld_direction(words, threshold, forward=True)
        
        # Backward MTLD (reverse word order)
        mtld_backward = self._calculate_mtld_direction(words[::-1], threshold, forward=False)
        
        # Average MTLD
        mtld = (mtld_forward + mtld_backward) / 2.0
        
        return {
            "mtld": mtld,
            "mtld_forward": mtld_forward,
            "mtld_backward": mtld_backward,
            "num_words": len(words)
        }
    
    def calculate_mattr(self, text: str, window_size: int = 50) -> Dict[str, Any]:
        """
        Calculate MATTR (Moving-Average Type-Token Ratio)
        
        MATTR calculates TTR for each sliding window and averages them.
        More robust to text length than simple TTR.
        
        Args:
            text: Input text
            window_size: Size of sliding window (default 50)
            
        Returns:
            Dictionary with MATTR metrics
        """
        words = self._tokenize(text)
        if len(words) < window_size:
            # For short texts, use simple TTR
            unique_words = len(set(words))
            ttr = unique_words / len(words) if words else 0.0
            return {
                "mattr": ttr,
                "ttr": ttr,
                "window_size": len(words),
                "num_windows": 1,
                "num_words": len(words)
            }
        
        # Calculate TTR for each window
        ttrs = []
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            unique = len(set(window))
            ttr = unique / window_size
            ttrs.append(ttr)
        
        mattr = sum(ttrs) / len(ttrs) if ttrs else 0.0
        
        return {
            "mattr": mattr,
            "ttr": mattr,  # For compatibility
            "window_size": window_size,
            "num_windows": len(ttrs),
            "num_words": len(words),
            "ttr_std": self._calculate_std(ttrs) if ttrs else 0.0
        }
    
    def calculate_zipf_normalized_ttr(self, text: str) -> Dict[str, Any]:
        """
        Calculate Zipf-normalized TTR
        
        Weights words by their frequency rank (Zipf distribution).
        Accounts for the fact that common words are less informative.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with Zipf-normalized TTR
        """
        words = self._tokenize(text)
        if not words:
            return {
                "zipf_ttr": 0.0,
                "unique_words": 0,
                "total_words": 0
            }
        
        # Count word frequencies
        word_counts = Counter(words)
        unique_words = len(word_counts)
        
        # Calculate Zipf weights (1/rank)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        zipf_weights = []
        total_weight = 0.0
        
        for rank, (word, count) in enumerate(sorted_words, start=1):
            # Zipf weight: 1 / rank
            weight = 1.0 / rank
            weighted_count = count * weight
            zipf_weights.append(weighted_count)
            total_weight += weighted_count
        
        # Normalized TTR: sum of weighted unique words / total weighted count
        weighted_unique = sum(1.0 / (rank + 1) for rank in range(unique_words))
        zipf_ttr = weighted_unique / total_weight if total_weight > 0 else 0.0
        
        return {
            "zipf_ttr": zipf_ttr,
            "unique_words": unique_words,
            "total_words": len(words),
            "simple_ttr": unique_words / len(words) if words else 0.0
        }
    
    def calculate_hapax_legomena(self, text: str) -> Dict[str, Any]:
        """
        Calculate hapax legomena (words that appear exactly once)
        
        Hapax legomena percentage indicates vocabulary sophistication.
        Higher values suggest more diverse vocabulary.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with hapax legomena metrics
        """
        words = self._tokenize(text)
        if not words:
            return {
                "hapax_count": 0,
                "hapax_percentage": 0.0,
                "unique_words": 0,
                "total_words": 0
            }
        
        word_counts = Counter(words)
        unique_words = len(word_counts)
        
        # Count hapax legomena (words appearing exactly once)
        hapax_words = [word for word, count in word_counts.items() if count == 1]
        hapax_count = len(hapax_words)
        hapax_percentage = (hapax_count / unique_words * 100) if unique_words > 0 else 0.0
        
        return {
            "hapax_count": hapax_count,
            "hapax_percentage": hapax_percentage,
            "unique_words": unique_words,
            "total_words": len(words),
            "hapax_words": hapax_words[:20]  # Sample of hapax words (limit to 20)
        }
    
    def calculate_all_lexical_metrics(self, text: str) -> Dict[str, Any]:
        """
        Calculate all lexical diversity metrics at once
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all metrics
        """
        mtld = self.calculate_mtld(text)
        mattr = self.calculate_mattr(text)
        zipf_ttr = self.calculate_zipf_normalized_ttr(text)
        hapax = self.calculate_hapax_legomena(text)
        
        return {
            "mtld": mtld,
            "mattr": mattr,
            "zipf_ttr": zipf_ttr,
            "hapax_legomena": hapax
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words (lowercase, alphanumeric)"""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _calculate_mtld_direction(self, words: List[str], threshold: float, forward: bool = True) -> float:
        """
        Calculate MTLD in one direction
        
        Args:
            words: List of words
            threshold: TTR threshold
            forward: If True, calculate forward; if False, backward
            
        Returns:
            MTLD score
        """
        if not words:
            return 0.0
        
        factors = 0
        current_pos = 0
        seen_types = set()
        
        while current_pos < len(words):
            # Reset for new factor
            factor_types = set()
            factor_length = 0
            
            # Build factor until threshold is reached
            while current_pos < len(words):
                word = words[current_pos]
                factor_types.add(word)
                factor_length += 1
                current_pos += 1
                
                # Calculate current TTR
                if factor_length > 0:
                    current_ttr = len(factor_types) / factor_length
                    
                    # If TTR drops below threshold, factor is complete
                    if current_ttr < threshold:
                        factors += 1
                        break
            
            # If we reached the end, calculate partial factor
            if current_pos >= len(words) and factor_length > 0:
                final_ttr = len(factor_types) / factor_length
                if final_ttr >= threshold:
                    # Partial factor: scale by how close we are to threshold
                    partial_factor = (1 - final_ttr) / (1 - threshold) if threshold < 1.0 else 0.0
                    factors += partial_factor
        
        # MTLD = total words / number of factors
        mtld = len(words) / factors if factors > 0 else 0.0
        return mtld
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

