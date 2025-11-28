"""
Referential Cohesion
Implements pronoun chains, co-reference resolution, and lexical chains
Requires dependency parser from linguistic_analysis service
"""

import re
import aiohttp
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from loguru import logger


class ReferentialCohesionAnalyzer:
    """Analyze referential cohesion in text"""
    
    def __init__(self, linguistic_analysis_url: str = "http://localhost:8901"):
        """
        Initialize analyzer
        
        Args:
            linguistic_analysis_url: URL of linguistic analysis service
        """
        self.linguistic_analysis_url = linguistic_analysis_url
    
    async def analyze_cohesion(self, text: str) -> Dict[str, Any]:
        """
        Analyze referential cohesion
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with cohesion metrics
        """
        # Basic analysis without parser (fallback)
        basic_analysis = self._basic_cohesion_analysis(text)
        
        # Try to get advanced analysis with parser
        try:
            advanced_analysis = await self._advanced_cohesion_analysis(text)
            basic_analysis.update(advanced_analysis)
        except Exception as e:
            logger.warning(f"⚠️ Could not get advanced cohesion analysis: {e}")
        
        return basic_analysis
    
    def _basic_cohesion_analysis(self, text: str) -> Dict[str, Any]:
        """
        Basic cohesion analysis without dependency parser
        
        Args:
            text: Input text
            
        Returns:
            Basic cohesion metrics
        """
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Pronoun tracking (basic)
        pronouns = {
            "eu", "tu", "ele", "ela", "nós", "vocês", "eles", "elas",
            "me", "te", "se", "nos", "vos",
            "meu", "teu", "seu", "nosso", "vosso",
            "este", "esse", "aquele", "isto", "isso", "aquilo"
        }
        
        pronoun_count = sum(1 for w in words if w in pronouns)
        pronoun_density = pronoun_count / len(words) if words else 0.0
        
        # Noun phrase repetition (simple: repeated nouns)
        word_counts = defaultdict(int)
        for word in words:
            if len(word) > 4:  # Focus on content words
                word_counts[word] += 1
        
        repeated_nouns = sum(1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_nouns / len(word_counts) if word_counts else 0.0
        
        return {
            "pronoun_count": pronoun_count,
            "pronoun_density": round(pronoun_density, 3),
            "repeated_nouns": repeated_nouns,
            "repetition_ratio": round(repetition_ratio, 3),
            "num_sentences": len(sentences),
            "num_words": len(words),
            "method": "basic"
        }
    
    async def _advanced_cohesion_analysis(self, text: str) -> Dict[str, Any]:
        """
        Advanced cohesion analysis using dependency parser
        
        Args:
            text: Input text
            
        Returns:
            Advanced cohesion metrics
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.linguistic_analysis_url}/api/parse"
                async with session.post(url, json={"text": text}) as response:
                    if response.status == 200:
                        parsed = await response.json()
                        
                        # Extract pronoun chains
                        pronoun_chains = self._extract_pronoun_chains(parsed)
                        
                        # Extract co-reference (basic via dependency)
                        coreference = self._extract_coreference(parsed)
                        
                        # Calculate distances between coreferent mentions
                        distances = self._calculate_coreference_distances(coreference)
                        
                        return {
                            "pronoun_chains": len(pronoun_chains),
                            "coreference_pairs": len(coreference),
                            "avg_coreference_distance": float(np.mean(distances)) if distances else 0.0,
                            "method": "advanced_parser"
                        }
                    else:
                        return {}
        except Exception as e:
            logger.warning(f"⚠️ Error in advanced cohesion analysis: {e}")
            return {}
    
    def _extract_pronoun_chains(self, parsed: Dict[str, Any]) -> List[List[str]]:
        """Extract pronoun chains from parsed text"""
        # Simple implementation: find sequences of pronouns
        chains = []
        current_chain = []
        
        for sentence in parsed.get("sentences", []):
            for token in sentence.get("tokens", []):
                if token.get("pos") == "PRON":
                    current_chain.append(token.get("text", ""))
                else:
                    if len(current_chain) > 1:
                        chains.append(current_chain.copy())
                    current_chain = []
        
        if len(current_chain) > 1:
            chains.append(current_chain)
        
        return chains
    
    def _extract_coreference(self, parsed: Dict[str, Any]) -> List[tuple]:
        """Extract co-reference pairs (basic implementation)"""
        # This is a simplified version - full co-reference resolution is complex
        # We look for noun phrases that might refer to the same entity
        coreference_pairs = []
        
        # Extract noun phrases
        noun_phrases = []
        for sentence in parsed.get("sentences", []):
            for token in sentence.get("tokens", []):
                if token.get("pos") in ["NOUN", "PROPN"]:
                    noun_phrases.append({
                        "text": token.get("text", ""),
                        "lemma": token.get("lemma", ""),
                        "sentence_idx": parsed.get("sentences", []).index(sentence)
                    })
        
        # Find potential coreference (same lemma, different positions)
        for i, np1 in enumerate(noun_phrases):
            for np2 in noun_phrases[i + 1:]:
                if np1["lemma"] == np2["lemma"]:
                    coreference_pairs.append((np1, np2))
        
        return coreference_pairs
    
    def _calculate_coreference_distances(self, coreference_pairs: List[tuple]) -> List[int]:
        """Calculate distances between coreferent mentions"""
        distances = []
        for np1, np2 in coreference_pairs:
            distance = abs(np2["sentence_idx"] - np1["sentence_idx"])
            distances.append(distance)
        return distances

