"""
Discourse Markers Analysis
Analyzes Portuguese discourse markers by CEFR level
"""

import re
from typing import Dict, Any, List
from collections import Counter


class DiscourseMarkersAnalyzer:
    """Analyze discourse markers in Portuguese text"""
    
    def __init__(self):
        """Initialize analyzer with Portuguese discourse marker lexicon"""
        # Discourse markers by CEFR level
        self.markers_by_level = {
            "A1": {
                "markers": ["e", "mas", "então", "porque", "ou"],
                "description": "Basic connectors"
            },
            "A2": {
                "markers": ["e", "mas", "então", "porque", "ou", "depois", "também", "quando"],
                "description": "Basic connectors with time markers"
            },
            "B1": {
                "markers": [
                    "e", "mas", "então", "porque", "ou", "depois", "também", "quando",
                    "porém", "contudo", "embora", "além disso", "por exemplo", "no entanto"
                ],
                "description": "Intermediate connectors with contrast and addition"
            },
            "B2": {
                "markers": [
                    "e", "mas", "então", "porque", "ou", "depois", "também", "quando",
                    "porém", "contudo", "embora", "além disso", "por exemplo", "no entanto",
                    "não obstante", "todavia", "ainda assim", "por outro lado", "dessa forma",
                    "consequentemente", "portanto", "assim sendo"
                ],
                "description": "Advanced connectors with sophisticated relationships"
            },
            "C1": {
                "markers": [
                    "e", "mas", "então", "porque", "ou", "depois", "também", "quando",
                    "porém", "contudo", "embora", "além disso", "por exemplo", "no entanto",
                    "não obstante", "todavia", "ainda assim", "por outro lado", "dessa forma",
                    "consequentemente", "portanto", "assim sendo",
                    "conquanto", "posto que", "visto que", "já que", "uma vez que",
                    "outrossim", "ademais", "além do mais", "não só... mas também"
                ],
                "description": "Sophisticated connectors with complex relationships"
            },
            "C2": {
                "markers": [
                    "e", "mas", "então", "porque", "ou", "depois", "também", "quando",
                    "porém", "contudo", "embora", "além disso", "por exemplo", "no entanto",
                    "não obstante", "todavia", "ainda assim", "por outro lado", "dessa forma",
                    "consequentemente", "portanto", "assim sendo",
                    "conquanto", "posto que", "visto que", "já que", "uma vez que",
                    "outrossim", "ademais", "além do mais", "não só... mas também",
                    "não obstante", "malgrado", "apesar de que", "tanto... quanto",
                    "quer... quer", "seja... seja", "ora... ora"
                ],
                "description": "Very sophisticated connectors with nuanced relationships"
            }
        }
    
    def analyze_markers(self, text: str) -> Dict[str, Any]:
        """
        Analyze discourse markers in text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with marker analysis
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Count markers by level
        markers_found = {}
        marker_counts = {}
        
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            level_markers = self.markers_by_level[level]["markers"]
            found = []
            count = 0
            
            for marker in level_markers:
                # Handle multi-word markers
                if " " in marker:
                    pattern = r'\b' + re.escape(marker) + r'\b'
                    matches = len(re.findall(pattern, text_lower))
                else:
                    pattern = r'\b' + re.escape(marker) + r'\b'
                    matches = len(re.findall(pattern, text_lower))
                
                if matches > 0:
                    found.append(marker)
                    count += matches
            
            markers_found[level] = found
            marker_counts[level] = count
        
        # Calculate density (markers per sentence)
        total_markers = sum(marker_counts.values())
        marker_density = total_markers / len(sentences) if sentences else 0.0
        
        # Determine sophistication level
        sophistication_level = self._determine_sophistication(marker_counts)
        
        # Marker diversity (unique markers / total markers)
        all_found_markers = set()
        for level_markers in markers_found.values():
            all_found_markers.update(level_markers)
        marker_diversity = len(all_found_markers) / total_markers if total_markers > 0 else 0.0
        
        return {
            "markers_found": markers_found,
            "marker_counts": marker_counts,
            "total_markers": total_markers,
            "marker_density": round(marker_density, 3),
            "marker_diversity": round(marker_diversity, 3),
            "sophistication_level": sophistication_level,
            "num_sentences": len(sentences),
            "num_words": len(words)
        }
    
    def _determine_sophistication(self, marker_counts: Dict[str, int]) -> str:
        """
        Determine overall sophistication level based on marker usage
        
        Args:
            marker_counts: Dictionary with counts per level
            
        Returns:
            Sophistication level (A1-C2)
        """
        # Weight higher levels more
        weighted_score = (
            marker_counts.get("A1", 0) * 1 +
            marker_counts.get("A2", 0) * 2 +
            marker_counts.get("B1", 0) * 3 +
            marker_counts.get("B2", 0) * 4 +
            marker_counts.get("C1", 0) * 5 +
            marker_counts.get("C2", 0) * 6
        )
        
        total = sum(marker_counts.values())
        if total == 0:
            return "A1"
        
        avg_score = weighted_score / total
        
        if avg_score <= 1.5:
            return "A1"
        elif avg_score <= 2.5:
            return "A2"
        elif avg_score <= 3.5:
            return "B1"
        elif avg_score <= 4.5:
            return "B2"
        elif avg_score <= 5.5:
            return "C1"
        else:
            return "C2"

