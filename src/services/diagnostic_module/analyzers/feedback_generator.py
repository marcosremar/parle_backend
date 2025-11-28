"""
Feedback Generator
Generates structured pedagogical feedback.
Based on Lu et al. (2025) and Xiao et al. (2024).
"""

from typing import Dict, Any, List, Optional
from loguru import logger


class FeedbackGenerator:
    """
    Generate structured pedagogical feedback.
    
    Based on:
    - Lu et al. (2025): Multi-aspect feedback with examples
    - Xiao et al. (2024): Explainable feedback generation
    """
    
    async def generate_structured_feedback(
        self,
        analysis: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured feedback with strengths, weaknesses, next steps.
        
        Args:
            analysis: Analysis dictionary from ComplexityAnalyzer
            user_id: User ID for personalized feedback
            
        Returns:
            Dictionary with:
            - strengths: List of student's strengths
            - weaknesses: List of weaknesses with details
            - next_steps: Recommended next steps
            - priority: Priority level (high/medium/low)
            - estimated_time: Estimated practice time in minutes
        """
        breakdown = analysis.get("breakdown", {})
        cefr_level = analysis.get("estimated_cefr_level", "B1")
        grammar_errors = analysis.get("grammar_error_profile", {})
        
        # Identify strengths
        strengths = await self._identify_strengths(analysis)
        
        # Identify weaknesses
        weaknesses = await self._identify_weaknesses(analysis)
        
        # Generate next steps
        next_steps = await self._generate_next_steps(analysis, user_id)
        
        # Calculate priority
        priority = self._calculate_priority(analysis, weaknesses)
        
        # Estimate practice time
        estimated_time = self._estimate_practice_time(analysis, weaknesses)
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "next_steps": next_steps,
            "priority": priority,
            "estimated_time_minutes": estimated_time
        }
    
    async def _identify_strengths(
        self, 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Identify student's strengths.
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            List of strength descriptions
        """
        strengths = []
        breakdown = analysis.get("breakdown", {})
        
        # Check each aspect
        aspect_names = {
            "fluency": "Fluência",
            "grammar": "Gramática",
            "vocabulary": "Vocabulário",
            "coherence": "Coerência",
            "pronunciation": "Pronúncia"
        }
        
        for aspect, score in breakdown.items():
            if score >= 4.0:  # Strong performance (4.0-5.0)
                aspect_name = aspect_names.get(aspect, aspect)
                strengths.append(f"Excelente {aspect_name.lower()} (nota {score:.1f}/5.0)")
            elif score >= 3.5:  # Good performance
                aspect_name = aspect_names.get(aspect, aspect)
                strengths.append(f"Boa {aspect_name.lower()} (nota {score:.1f}/5.0)")
        
        # Check pronunciation score if available
        if "pronunciation_score" in analysis:
            pron_score = analysis["pronunciation_score"]
            if pron_score >= 0.8:
                strengths.append("Pronúncia clara e compreensível")
        
        # If no strengths found, add encouraging message
        if not strengths:
            strengths.append("Bom esforço! Continue praticando")
        
        return strengths
    
    async def _identify_weaknesses(
        self, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify weaknesses with specific examples.
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            List of weakness dictionaries with:
            - type: Type of weakness
            - severity: high/medium/low
            - description: Description of the issue
            - examples: List of examples (if available)
        """
        weaknesses = []
        breakdown = analysis.get("breakdown", {})
        grammar_errors = analysis.get("grammar_error_profile", {})
        
        # Check grammar errors
        if grammar_errors:
            for error_type, count in grammar_errors.items():
                if count > 0:
                    severity = "high" if count > 5 else "medium" if count > 2 else "low"
                    
                    error_names = {
                        "verb": "Erros verbais",
                        "agreement": "Erros de concordância",
                        "morphology": "Erros morfológicos",
                        "word_order": "Erros de ordem de palavras",
                        "preposition": "Erros de preposição"
                    }
                    
                    error_name = error_names.get(error_type, error_type)
                    
                    weaknesses.append({
                        "type": error_type,
                        "category": "grammar",
                        "severity": severity,
                        "description": f"{error_name}: {count} erro(s) encontrado(s)",
                        "count": count,
                        "examples": []  # Could extract from text
                    })
        
        # Check low scores in breakdown
        aspect_names = {
            "fluency": "Fluência",
            "grammar": "Gramática",
            "vocabulary": "Vocabulário",
            "coherence": "Coerência"
        }
        
        for aspect, score in breakdown.items():
            if score < 3.0:  # Below average
                aspect_name = aspect_names.get(aspect, aspect)
                severity = "high" if score < 2.0 else "medium"
                
                weaknesses.append({
                    "type": aspect,
                    "category": "performance",
                    "severity": severity,
                    "description": f"{aspect_name} abaixo da média (nota {score:.1f}/5.0)",
                    "score": score,
                    "examples": []
                })
        
        # Check pronunciation if available
        if "pronunciation_score" in analysis:
            pron_score = analysis["pronunciation_score"]
            if pron_score < 0.7:
                weaknesses.append({
                    "type": "pronunciation",
                    "category": "pronunciation",
                    "severity": "medium" if pron_score < 0.5 else "low",
                    "description": f"Pronúncia precisa melhorar (score {pron_score:.2f})",
                    "score": pron_score,
                    "examples": []
                })
        
        return weaknesses
    
    async def _generate_next_steps(
        self,
        analysis: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[str]:
        """
        Generate recommended next steps.
        
        Args:
            analysis: Analysis dictionary
            user_id: User ID for personalized recommendations
            
        Returns:
            List of recommended next steps
        """
        next_steps = []
        weaknesses = await self._identify_weaknesses(analysis)
        breakdown = analysis.get("breakdown", {})
        
        # Prioritize by severity
        high_priority = [w for w in weaknesses if w["severity"] == "high"]
        medium_priority = [w for w in weaknesses if w["severity"] == "medium"]
        
        # Grammar errors
        grammar_weaknesses = [w for w in weaknesses if w.get("category") == "grammar"]
        if grammar_weaknesses:
            error_types = [w["type"] for w in grammar_weaknesses]
            if "verb" in error_types:
                next_steps.append("Praticar conjugação verbal com exercícios específicos")
            if "agreement" in error_types:
                next_steps.append("Revisar regras de concordância (sujeito-verbo, substantivo-adjetivo)")
            if "preposition" in error_types:
                next_steps.append("Estudar uso de preposições em contextos diferentes")
        
        # Low vocabulary score
        if breakdown.get("vocabulary", 5.0) < 3.0:
            next_steps.append("Expandir vocabulário com leitura e flashcards")
        
        # Low fluency score
        if breakdown.get("fluency", 5.0) < 3.0:
            next_steps.append("Praticar conversação para melhorar fluência")
        
        # Low coherence score
        if breakdown.get("coherence", 5.0) < 3.0:
            next_steps.append("Trabalhar organização de ideias e uso de conectores")
        
        # Pronunciation
        if "pronunciation_score" in analysis and analysis["pronunciation_score"] < 0.7:
            next_steps.append("Praticar pronúncia com exercícios de repetição")
        
        # If no specific steps, add general recommendations
        if not next_steps:
            cefr_level = analysis.get("estimated_cefr_level", "B1")
            if cefr_level in ["A1", "A2"]:
                next_steps.append("Continuar praticando estruturas básicas")
            elif cefr_level in ["B1", "B2"]:
                next_steps.append("Expandir uso de estruturas complexas")
            else:
                next_steps.append("Refinar uso de linguagem sofisticada")
        
        return next_steps[:5]  # Limit to 5 steps
    
    def _calculate_priority(
        self,
        analysis: Dict[str, Any],
        weaknesses: List[Dict[str, Any]]
    ) -> str:
        """
        Calculate overall priority level.
        
        Args:
            analysis: Analysis dictionary
            weaknesses: List of weaknesses
            
        Returns:
            Priority: "high", "medium", or "low"
        """
        if not weaknesses:
            return "low"
        
        # Count high severity weaknesses
        high_count = sum(1 for w in weaknesses if w["severity"] == "high")
        medium_count = sum(1 for w in weaknesses if w["severity"] == "medium")
        
        if high_count >= 2:
            return "high"
        elif high_count >= 1 or medium_count >= 3:
            return "medium"
        else:
            return "low"
    
    def _estimate_practice_time(
        self,
        analysis: Dict[str, Any],
        weaknesses: List[Dict[str, Any]]
    ) -> int:
        """
        Estimate practice time needed in minutes.
        
        Args:
            analysis: Analysis dictionary
            weaknesses: List of weaknesses
            
        Returns:
            Estimated time in minutes
        """
        if not weaknesses:
            return 15  # Minimal practice
        
        # Base time
        base_time = 20
        
        # Add time per weakness
        for weakness in weaknesses:
            if weakness["severity"] == "high":
                base_time += 15
            elif weakness["severity"] == "medium":
                base_time += 10
            else:
                base_time += 5
        
        # Cap at 60 minutes
        return min(60, base_time)
