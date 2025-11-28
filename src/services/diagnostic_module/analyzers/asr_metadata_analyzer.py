"""
ASR Metadata Analyzer
Extracts and analyzes ASR metadata (timestamps, confidence).
Based on Mohammadi et al. (2025) and CASPER Dataset (2024).
"""

import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger


class ASRMetadataAnalyzer:
    """
    Extract and analyze ASR metadata for pronunciation and fluency insights.
    
    Based on:
    - Mohammadi et al. (2025): 88 acoustic features for pronunciation assessment
    - CASPER Dataset (2024): Spontaneous speech with timestamps and metadata
    """
    
    def analyze_asr_metadata(
        self,
        asr_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze ASR metadata for pronunciation and fluency insights.
        
        Args:
            asr_output: ASR output dictionary with format:
                {
                    "text": "...",
                    "words": [
                        {"word": "...", "start": 0.0, "end": 0.2, "confidence": 0.98},
                        ...
                    ],
                    "duration": 2.0
                }
        
        Returns:
            Dictionary with analysis results:
            - avg_confidence: Average confidence score
            - low_confidence_words: List of words with low confidence
            - speech_rate_wpm: Words per minute
            - num_pauses: Number of pauses detected
            - avg_pause_duration: Average pause duration
            - fluency_score: Overall fluency score (0-1)
        """
        words = asr_output.get("words", [])
        duration = asr_output.get("duration", 0)
        
        if not words:
            logger.warning("No words in ASR output")
            return {
                "avg_confidence": 0.0,
                "low_confidence_words": [],
                "speech_rate_wpm": 0.0,
                "num_pauses": 0,
                "avg_pause_duration": 0.0,
                "fluency_score": 0.0
            }
        
        # Calculate average confidence
        confidences = [w.get("confidence", 0.0) for w in words]
        avg_confidence = float(np.mean(confidences))
        
        # Identify low confidence words (threshold: 0.8)
        low_confidence_words = [
            {
                "word": w.get("word", ""),
                "confidence": w.get("confidence", 0.0),
                "position": i
            }
            for i, w in enumerate(words)
            if w.get("confidence", 0.0) < 0.8
        ]
        
        # Speech rate (words per minute)
        speech_rate_wpm = (len(words) / duration) * 60 if duration > 0 else 0.0
        
        # Pause detection (gaps > 0.5s)
        pauses = []
        for i in range(len(words) - 1):
            current_end = words[i].get("end", 0.0)
            next_start = words[i + 1].get("start", 0.0)
            gap = next_start - current_end
            
            if gap > 0.5:  # Pause threshold: 0.5 seconds
                pauses.append({
                    "position": i,
                    "duration": gap,
                    "before_word": words[i].get("word", ""),
                    "after_word": words[i + 1].get("word", "")
                })
        
        num_pauses = len(pauses)
        avg_pause_duration = float(np.mean([p["duration"] for p in pauses])) if pauses else 0.0
        
        # Calculate fluency score
        fluency_score = self._calculate_fluency_score(
            avg_confidence,
            speech_rate_wpm,
            num_pauses,
            len(words)
        )
        
        return {
            "avg_confidence": avg_confidence,
            "low_confidence_words": low_confidence_words,
            "speech_rate_wpm": float(speech_rate_wpm),
            "num_pauses": num_pauses,
            "avg_pause_duration": avg_pause_duration,
            "pauses": pauses,
            "fluency_score": fluency_score,
            "total_words": len(words),
            "total_duration": float(duration)
        }
    
    def _calculate_fluency_score(
        self,
        avg_confidence: float,
        speech_rate_wpm: float,
        num_pauses: int,
        total_words: int
    ) -> float:
        """
        Calculate overall fluency score.
        
        Args:
            avg_confidence: Average ASR confidence
            speech_rate_wpm: Words per minute
            num_pauses: Number of pauses
            total_words: Total number of words
            
        Returns:
            Fluency score (0-1)
        """
        # Normalize components
        
        # Confidence component (0-1)
        confidence_component = avg_confidence
        
        # Speech rate component
        # Normal range: 120-180 WPM for native speakers
        # L2 learners: 80-150 WPM is good
        if speech_rate_wpm < 50:
            rate_component = 0.3
        elif speech_rate_wpm < 80:
            rate_component = 0.5
        elif speech_rate_wpm < 120:
            rate_component = 0.7
        elif speech_rate_wpm < 180:
            rate_component = 1.0
        else:
            rate_component = 0.9  # Too fast can indicate rushing
        
        # Pause component
        # Normal: 1-2 pauses per 10 words
        pause_rate = (num_pauses / total_words) * 10 if total_words > 0 else 0
        if pause_rate <= 2:
            pause_component = 1.0
        elif pause_rate <= 4:
            pause_component = 0.7
        elif pause_rate <= 6:
            pause_component = 0.5
        else:
            pause_component = 0.3
        
        # Weighted combination
        fluency_score = (
            0.4 * confidence_component +
            0.4 * rate_component +
            0.2 * pause_component
        )
        
        return float(np.clip(fluency_score, 0.0, 1.0))
    
    def identify_problematic_words(
        self,
        asr_output: Dict[str, Any],
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Identify words with pronunciation issues.
        
        Args:
            asr_output: ASR output dictionary
            threshold: Confidence threshold (words below this are problematic)
            
        Returns:
            List of problematic words with details
        """
        words = asr_output.get("words", [])
        
        problematic = []
        
        for i, word_data in enumerate(words):
            confidence = word_data.get("confidence", 0.0)
            
            if confidence < threshold:
                problematic.append({
                    "word": word_data.get("word", ""),
                    "confidence": confidence,
                    "position": i,
                    "start": word_data.get("start", 0.0),
                    "end": word_data.get("end", 0.0),
                    "severity": "high" if confidence < 0.5 else "medium"
                })
        
        return problematic
