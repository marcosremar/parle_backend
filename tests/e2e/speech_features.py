"""
Speech-Specific Features
Implements Mean Word Span (MWS), repetition analysis, disfluency markers, pause detection
Based on: Arnold et al. (2018) speech corpus analysis
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter


class SpeechFeaturesAnalyzer:
    """Analyze speech-specific features for conversational language"""
    
    def __init__(self):
        """Initialize analyzer"""
        # Disfluency markers in Portuguese
        self.disfluency_markers = [
            "é", "eh", "ah", "uh", "hmm", "hm", "né", "tá", "tipo", "assim",
            "então", "aí", "daí", "tipo assim", "é tipo", "sabe", "entendeu"
        ]
        
        # Fillers
        self.fillers = ["é", "eh", "ah", "uh", "hmm", "hm", "né"]
    
    def calculate_mean_word_span(self, text: str, register: str = "spoken") -> Dict[str, Any]:
        """
        Calculate Mean Word Span (MWS) - average distance between content words
        
        MWS measures the spacing between meaningful words, indicating
        fluency and complexity of speech.
        
        Based on NILC-Metrix (2022): MWS normalized by register.
        
        Args:
            text: Input text
            register: Text register ("spoken", "academic", "fiction", "news")
            
        Returns:
            Dictionary with MWS metrics including normalized scores
        """
        words = self._tokenize(text)
        content_words = self._extract_content_words(words)
        
        if len(content_words) < 2:
            return {
                "mws": 0.0,
                "num_content_words": len(content_words),
                "total_words": len(words),
                "normalized_mws": {},
                "register": register
            }
        
        # Calculate distances between consecutive content words
        distances = []
        last_content_pos = None
        
        for i, word in enumerate(words):
            if word.lower() in content_words:
                if last_content_pos is not None:
                    distance = i - last_content_pos
                    distances.append(distance)
                last_content_pos = i
        
        mws = sum(distances) / len(distances) if distances else 0.0
        
        # Normalize MWS by register (NILC-Metrix, 2022)
        # Expected MWS ranges per register (based on Portuguese corpus analysis)
        register_norms = {
            "spoken": {"mean": 2.5, "std": 0.8, "min": 1.5, "max": 4.5},
            "academic": {"mean": 3.8, "std": 1.2, "min": 2.5, "max": 6.5},
            "fiction": {"mean": 3.2, "std": 1.0, "min": 2.0, "max": 5.5},
            "news": {"mean": 3.5, "std": 1.1, "min": 2.2, "max": 6.0}
        }
        
        norm = register_norms.get(register, register_norms["spoken"])
        
        # Z-score normalization
        z_score = (mws - norm["mean"]) / norm["std"] if norm["std"] > 0 else 0.0
        
        # Percentile score (0-1 scale)
        if mws <= norm["min"]:
            percentile = 0.0
        elif mws >= norm["max"]:
            percentile = 1.0
        else:
            # Linear interpolation
            percentile = (mws - norm["min"]) / (norm["max"] - norm["min"])
        
        # Normalized MWS scores per register
        normalized_mws = {}
        for reg, reg_norm in register_norms.items():
            reg_z = (mws - reg_norm["mean"]) / reg_norm["std"] if reg_norm["std"] > 0 else 0.0
            if mws <= reg_norm["min"]:
                reg_percentile = 0.0
            elif mws >= reg_norm["max"]:
                reg_percentile = 1.0
            else:
                reg_percentile = (mws - reg_norm["min"]) / (reg_norm["max"] - reg_norm["min"])
            normalized_mws[reg] = {
                "z_score": round(reg_z, 3),
                "percentile": round(reg_percentile, 3),
                "expected_range": (reg_norm["min"], reg_norm["max"])
            }
        
        return {
            "mws": round(mws, 3),
            "num_content_words": len(content_words),
            "total_words": len(words),
            "content_word_ratio": round(len(content_words) / len(words), 3) if words else 0.0,
            "register": register,
            "z_score": round(z_score, 3),
            "percentile": round(percentile, 3),
            "normalized_mws": normalized_mws,
            "expected_range": (norm["min"], norm["max"])
        }
    
    def analyze_repetitions(self, text: str) -> Dict[str, Any]:
        """
        Analyze immediate and non-immediate repetitions
        
        Repetition patterns are common in speech and indicate different
        proficiency levels.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with repetition metrics
        """
        words = self._tokenize(text)
        
        if len(words) < 2:
            return {
                "immediate_repetitions": 0,
                "non_immediate_repetitions": 0,
                "repetition_rate": 0.0,
                "repetition_examples": []
            }
        
        # Immediate repetitions (same word consecutively)
        immediate = 0
        immediate_examples = []
        
        for i in range(len(words) - 1):
            if words[i].lower() == words[i + 1].lower():
                immediate += 1
                if len(immediate_examples) < 5:
                    immediate_examples.append(f"{words[i]} {words[i+1]}")
        
        # Non-immediate repetitions (same word within 5 words)
        non_immediate = 0
        non_immediate_examples = []
        
        for i in range(len(words)):
            word = words[i].lower()
            # Check next 5 words
            for j in range(i + 2, min(i + 7, len(words))):
                if words[j].lower() == word:
                    non_immediate += 1
                    if len(non_immediate_examples) < 5:
                        non_immediate_examples.append(f"{word} (pos {i} and {j})")
                    break  # Count each word pair only once
        
        total_repetitions = immediate + non_immediate
        repetition_rate = total_repetitions / len(words) if words else 0.0
        
        return {
            "immediate_repetitions": immediate,
            "non_immediate_repetitions": non_immediate,
            "total_repetitions": total_repetitions,
            "repetition_rate": repetition_rate,
            "repetition_examples": {
                "immediate": immediate_examples,
                "non_immediate": non_immediate_examples[:5]
            }
        }
    
    def detect_disfluencies(self, text: str) -> Dict[str, Any]:
        """
        Detect disfluency markers (false starts, repairs, fillers)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with disfluency metrics
        """
        words = self._tokenize(text)
        text_lower = text.lower()
        
        # Count fillers
        filler_count = 0
        fillers_found = []
        
        for filler in self.fillers:
            # Count occurrences (as whole words)
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                filler_count += len(matches)
                fillers_found.append(filler)
        
        # Detect false starts (incomplete sentences ending with "é", "tipo", etc.)
        false_starts = 0
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Check if sentence ends with disfluency marker
                last_words = sentence.split()[-3:] if len(sentence.split()) >= 3 else sentence.split()
                for word in last_words:
                    if word.lower() in self.disfluency_markers:
                        false_starts += 1
                        break
        
        # Detect repairs (self-corrections: "não, quero dizer...")
        repair_patterns = [
            r'\bnão\b.*\bquero dizer\b',
            r'\bquer dizer\b',
            r'\bdigo\b.*\bquero dizer\b',
            r'\bcorrigindo\b',
            r'\bmelhor dizendo\b'
        ]
        
        repairs = 0
        for pattern in repair_patterns:
            matches = re.findall(pattern, text_lower)
            repairs += len(matches)
        
        total_disfluencies = filler_count + false_starts + repairs
        disfluency_rate = total_disfluencies / len(words) if words else 0.0
        
        return {
            "filler_count": filler_count,
            "false_starts": false_starts,
            "repairs": repairs,
            "total_disfluencies": total_disfluencies,
            "disfluency_rate": disfluency_rate,
            "fillers_found": fillers_found[:10]  # Limit to 10
        }
    
    def analyze_pauses(self, text: str) -> Dict[str, Any]:
        """
        Analyze pause patterns (via punctuation and sentence boundaries)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with pause metrics
        """
        # Count punctuation marks that indicate pauses
        commas = text.count(',')
        periods = text.count('.')
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        ellipses = text.count('...') + text.count('…')
        
        total_pauses = commas + periods + question_marks + exclamation_marks + ellipses
        
        # Calculate pause density
        words = self._tokenize(text)
        pause_density = total_pauses / len(words) if words else 0.0
        
        # Average words between pauses
        if total_pauses > 0:
            avg_words_between_pauses = len(words) / total_pauses
        else:
            avg_words_between_pauses = len(words)  # No pauses
        
        return {
            "commas": commas,
            "periods": periods,
            "question_marks": question_marks,
            "exclamation_marks": exclamation_marks,
            "ellipses": ellipses,
            "total_pauses": total_pauses,
            "pause_density": pause_density,
            "avg_words_between_pauses": avg_words_between_pauses,
            "num_words": len(words)
        }
    
    def analyze_self_corrections(self, text: str) -> Dict[str, Any]:
        """
        Analyze self-correction patterns
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with self-correction metrics
        """
        text_lower = text.lower()
        
        # Patterns indicating self-correction
        correction_patterns = [
            (r'\bnão\b.*\bquero dizer\b', 'explicit_correction'),
            (r'\bquer dizer\b', 'correction_marker'),
            (r'\bdigo\b', 'correction_marker'),
            (r'\bmelhor dizendo\b', 'explicit_correction'),
            (r'\bcorrigindo\b', 'explicit_correction'),
            (r'\bna verdade\b', 'correction_marker'),
            (r'\bna realidade\b', 'correction_marker')
        ]
        
        corrections = []
        for pattern, correction_type in correction_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                corrections.append({
                    "type": correction_type,
                    "text": match.group(),
                    "position": match.start()
                })
        
        return {
            "correction_count": len(corrections),
            "corrections": corrections[:10],  # Limit to 10
            "correction_rate": len(corrections) / len(self._tokenize(text)) if text else 0.0
        }
    
    def calculate_all_speech_features(self, text: str, register: str = "spoken") -> Dict[str, Any]:
        """
        Calculate all speech-specific features
        
        Args:
            text: Input text
            register: Text register ("spoken", "academic", "fiction", "news")
            
        Returns:
            Dictionary with all speech features including normalized MWS
        """
        mws = self.calculate_mean_word_span(text, register)
        repetitions = self.analyze_repetitions(text)
        disfluencies = self.detect_disfluencies(text)
        pauses = self.analyze_pauses(text)
        corrections = self.analyze_self_corrections(text)
        
        return {
            "mean_word_span": mws,
            "repetitions": repetitions,
            "disfluencies": disfluencies,
            "pauses": pauses,
            "self_corrections": corrections,
            "register": register
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _extract_content_words(self, words: List[str]) -> List[str]:
        """
        Extract content words (nouns, verbs, adjectives, adverbs)
        
        Simple heuristic: exclude common function words
        """
        function_words = {
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
            'de', 'da', 'do', 'das', 'dos', 'em', 'na', 'no', 'nas', 'nos',
            'para', 'por', 'com', 'sem', 'sobre', 'entre', 'até',
            'que', 'qual', 'quais', 'quando', 'onde', 'como',
            'e', 'ou', 'mas', 'porque', 'se', 'então',
            'eu', 'tu', 'ele', 'ela', 'nós', 'vocês', 'eles', 'elas',
            'me', 'te', 'se', 'nos', 'vos',
            'meu', 'teu', 'seu', 'nosso', 'vosso',
            'este', 'esse', 'aquele', 'isto', 'isso', 'aquilo',
            'ser', 'estar', 'ter', 'haver', 'fazer', 'ir', 'vir'
        }
        
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        return content_words

