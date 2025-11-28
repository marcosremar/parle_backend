"""
Psycholinguistic Metrics
Implements Age of Acquisition (AoA), concreteness, familiarity, imageability
Uses word frequency and psycholinguistic databases
"""

import re
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import Counter


class PsycholinguisticMetricsCalculator:
    """Calculate psycholinguistic metrics for vocabulary"""
    
    def __init__(self, word_stats_path: Optional[Path] = None):
        """
        Initialize calculator
        
        Args:
            word_stats_path: Path to word statistics database (optional)
        """
        if word_stats_path is None:
            project_root = Path(__file__).parent.parent.parent
            word_stats_path = project_root / "data" / "psycholinguistic" / "word_stats.json"
        
        self.word_stats_path = Path(word_stats_path)
        self.word_stats = self._load_word_stats()
    
    def _load_word_stats(self) -> Dict[str, Dict[str, float]]:
        """Load word statistics database"""
        if self.word_stats_path.exists():
            try:
                with open(self.word_stats_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load word stats: {e}")
        
        # Return empty dict - will use fallback heuristics
        return {}
    
    def calculate_metrics(self, text: str) -> Dict[str, Any]:
        """
        Calculate all psycholinguistic metrics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with psycholinguistic metrics
        """
        words = self._tokenize(text)
        
        if not words:
            return {
                "mean_aoa": 0.0,
                "mean_concreteness": 0.0,
                "mean_familiarity": 0.0,
                "mean_imageability": 0.0,
                "mean_zipf_frequency": 0.0,
                "rare_word_percentage": 0.0,
                "num_words": 0
            }
        
        # Calculate metrics for each word
        aoas = []
        concreteness_scores = []
        familiarity_scores = []
        imageability_scores = []
        zipf_frequencies = []
        rare_word_count = 0
        
        for word in words:
            word_lower = word.lower()
            stats = self.word_stats.get(word_lower, {})
            
            # Age of Acquisition (AoA) - typically 1-18 years
            aoa = stats.get("aoa", self._estimate_aoa(word))
            aoas.append(aoa)
            
            # Concreteness - typically 1-7 scale
            concreteness = stats.get("concreteness", self._estimate_concreteness(word))
            concreteness_scores.append(concreteness)
            
            # Familiarity - typically 1-7 scale
            familiarity = stats.get("familiarity", self._estimate_familiarity(word))
            familiarity_scores.append(familiarity)
            
            # Imageability - typically 1-7 scale
            imageability = stats.get("imageability", self._estimate_imageability(word))
            imageability_scores.append(imageability)
            
            # Zipf frequency - log10 frequency, typically 1-7
            zipf = stats.get("zipf_frequency", self._estimate_zipf_frequency(word))
            zipf_frequencies.append(zipf)
            
            # Rare words (Zipf < 3.0)
            if zipf < 3.0:
                rare_word_count += 1
        
        # Calculate means
        mean_aoa = sum(aoas) / len(aoas) if aoas else 0.0
        mean_concreteness = sum(concreteness_scores) / len(concreteness_scores) if concreteness_scores else 0.0
        mean_familiarity = sum(familiarity_scores) / len(familiarity_scores) if familiarity_scores else 0.0
        mean_imageability = sum(imageability_scores) / len(imageability_scores) if imageability_scores else 0.0
        mean_zipf = sum(zipf_frequencies) / len(zipf_frequencies) if zipf_frequencies else 0.0
        rare_word_percentage = (rare_word_count / len(words)) * 100 if words else 0.0
        
        return {
            "mean_aoa": round(mean_aoa, 2),
            "mean_concreteness": round(mean_concreteness, 2),
            "mean_familiarity": round(mean_familiarity, 2),
            "mean_imageability": round(mean_imageability, 2),
            "mean_zipf_frequency": round(mean_zipf, 2),
            "rare_word_percentage": round(rare_word_percentage, 2),
            "rare_word_count": rare_word_count,
            "num_words": len(words)
        }
    
    def _estimate_aoa(self, word: str) -> float:
        """
        Estimate Age of Acquisition using heuristics
        
        Longer, less common words are typically acquired later
        """
        # Simple heuristic: word length and frequency proxy
        length_factor = min(len(word) / 10.0, 1.0)  # Longer words acquired later
        # Common words (short, frequent) acquired early
        common_words = {
            'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
            'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais'
        }
        if word.lower() in common_words:
            return 3.0  # Very early
        else:
            return 5.0 + length_factor * 10.0  # 5-15 years
    
    def _estimate_concreteness(self, word: str) -> float:
        """
        Estimate concreteness (1-7 scale, higher = more concrete)
        
        Concrete words: objects, actions, body parts
        Abstract words: concepts, emotions, ideas
        """
        # Concrete words (nouns for objects, body parts, actions)
        concrete_patterns = [
            r'\b(casa|carro|mesa|cadeira|livro|água|comida|pão|leite)\b',
            r'\b(mão|pé|olho|cabeça|braço|perna)\b',
            r'\b(correr|andar|comer|beber|dormir|ver|ouvir)\b'
        ]
        
        # Abstract words (concepts, emotions)
        abstract_patterns = [
            r'\b(amor|ódio|felicidade|tristeza|esperança|medo)\b',
            r'\b(ideia|conceito|teoria|filosofia|moral|ética)\b',
            r'\b(liberdade|justiça|verdade|beleza|sabedoria)\b'
        ]
        
        word_lower = word.lower()
        
        for pattern in concrete_patterns:
            if re.search(pattern, word_lower):
                return 6.0  # High concreteness
        
        for pattern in abstract_patterns:
            if re.search(pattern, word_lower):
                return 2.0  # Low concreteness
        
        # Default: medium concreteness
        return 4.0
    
    def _estimate_familiarity(self, word: str) -> float:
        """
        Estimate familiarity (1-7 scale, higher = more familiar)
        
        Common words are more familiar
        """
        common_words = {
            'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
            'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
            'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem',
            'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos'
        }
        
        if word.lower() in common_words:
            return 7.0  # Very familiar
        elif len(word) <= 4:
            return 6.0  # Short words more familiar
        elif len(word) <= 6:
            return 5.0  # Medium familiarity
        else:
            return 4.0  # Less familiar
    
    def _estimate_imageability(self, word: str) -> float:
        """
        Estimate imageability (1-7 scale, higher = more imageable)
        
        Similar to concreteness but focuses on visualizability
        """
        # Highly imageable words
        imageable_patterns = [
            r'\b(casa|carro|árvore|sol|lua|estrela|gato|cachorro)\b',
            r'\b(vermelho|azul|verde|amarelo|preto|branco)\b',
            r'\b(correr|pular|dançar|nadar|voar)\b'
        ]
        
        word_lower = word.lower()
        
        for pattern in imageable_patterns:
            if re.search(pattern, word_lower):
                return 6.5  # High imageability
        
        # Low imageability (abstract concepts)
        abstract_patterns = [
            r'\b(amor|ódio|felicidade|tristeza|esperança)\b',
            r'\b(ideia|conceito|teoria|filosofia)\b'
        ]
        
        for pattern in abstract_patterns:
            if re.search(pattern, word_lower):
                return 2.0  # Low imageability
        
        return 4.0  # Medium imageability
    
    def _estimate_zipf_frequency(self, word: str) -> float:
        """
        Estimate Zipf frequency (log10 frequency, typically 1-7)
        
        Common words have higher Zipf values
        """
        # Very common words
        very_common = {
            'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para'
        }
        if word.lower() in very_common:
            return 6.5
        
        # Common words
        common = {
            'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
            'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem'
        }
        if word.lower() in common:
            return 5.5
        
        # Medium frequency
        if len(word) <= 4:
            return 4.5
        elif len(word) <= 6:
            return 3.5
        else:
            return 2.5  # Rare words
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words

