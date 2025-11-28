"""
Error Rate Analyzer
Extracts error-rate features by comparing ASR transcription with expected text.
Based on Do et al. (Interspeech 2024) - Acoustic Feature Mixup paper.
"""

import Levenshtein
from typing import Dict, Optional


class ErrorRateAnalyzer:
    """
    Analyze error rates between ASR transcription and expected text.
    
    Based on Do et al. (Interspeech 2024):
    - Character-level match error rate
    - Token-level match error rate
    
    These features provide direct hints for mispronunciation detection.
    """
    
    def extract_error_rate_features(
        self, 
        asr_transcription: str, 
        expected_text: str
    ) -> Dict[str, float]:
        """
        Extract error-rate features by comparing ASR with expected text.
        
        Args:
            asr_transcription: Text transcribed by ASR system
            expected_text: Expected/correct text
            
        Returns:
            Dictionary with error rate features:
            - char_error_rate: Character-level error rate (0-1)
            - token_error_rate: Token-level error rate (0-1)
            - char_distance: Raw Levenshtein distance (characters)
            - token_distance: Raw Levenshtein distance (tokens)
        
        Example:
            >>> analyzer = ErrorRateAnalyzer()
            >>> features = analyzer.extract_error_rate_features(
            ...     "Eu gosto de estudar português",
            ...     "Eu gosto de estudar português"
            ... )
            >>> features['char_error_rate']
            0.0
        """
        # Normalize whitespace
        asr_transcription = " ".join(asr_transcription.split())
        expected_text = " ".join(expected_text.split())
        
        # Character-level error rate
        char_distance = Levenshtein.distance(asr_transcription, expected_text)
        char_error_rate = char_distance / max(len(expected_text), 1)
        
        # Token-level error rate
        asr_tokens = asr_transcription.split()
        expected_tokens = expected_text.split()
        
        # Calculate token distance
        token_distance = Levenshtein.distance(
            ' '.join(asr_tokens), 
            ' '.join(expected_tokens)
        )
        token_error_rate = token_distance / max(len(expected_tokens), 1)
        
        return {
            "char_error_rate": float(char_error_rate),
            "token_error_rate": float(token_error_rate),
            "char_distance": int(char_distance),
            "token_distance": int(token_distance),
            "asr_length": len(asr_transcription),
            "expected_length": len(expected_text),
            "asr_tokens": len(asr_tokens),
            "expected_tokens": len(expected_tokens)
        }
    
    def calculate_pronunciation_score(
        self,
        error_features: Dict[str, float]
    ) -> float:
        """
        Calculate overall pronunciation score from error features.
        
        Args:
            error_features: Output from extract_error_rate_features()
            
        Returns:
            Pronunciation score (0-1), where 1 is perfect
        """
        # Weight character and token errors
        char_weight = 0.4
        token_weight = 0.6
        
        char_score = 1.0 - error_features["char_error_rate"]
        token_score = 1.0 - error_features["token_error_rate"]
        
        overall_score = (char_weight * char_score) + (token_weight * token_score)
        
        return max(0.0, min(1.0, overall_score))
    
    def identify_error_positions(
        self,
        asr_transcription: str,
        expected_text: str
    ) -> Dict[str, any]:
        """
        Identify specific positions where errors occur.
        
        Args:
            asr_transcription: Text transcribed by ASR
            expected_text: Expected/correct text
            
        Returns:
            Dictionary with error positions and types
        """
        asr_tokens = asr_transcription.split()
        expected_tokens = expected_text.split()
        
        errors = []
        
        # Simple token-by-token comparison
        max_len = max(len(asr_tokens), len(expected_tokens))
        
        for i in range(max_len):
            asr_token = asr_tokens[i] if i < len(asr_tokens) else ""
            expected_token = expected_tokens[i] if i < len(expected_tokens) else ""
            
            if asr_token != expected_token:
                error_type = "substitution"
                if not asr_token:
                    error_type = "deletion"
                elif not expected_token:
                    error_type = "insertion"
                
                errors.append({
                    "position": i,
                    "type": error_type,
                    "asr": asr_token,
                    "expected": expected_token
                })
        
        return {
            "num_errors": len(errors),
            "errors": errors,
            "error_rate": len(errors) / max(len(expected_tokens), 1)
        }

