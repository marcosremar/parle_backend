"""
Grammar Analyzer - Análise de erros gramaticais
"""

from typing import List, Dict, Any, Optional
from ..models import ErrorAnalysis, ErrorType, ErrorCategory
from ..llm_client import DiagnosticLLMClient


class GrammarAnalyzer:
    """Analisador de erros gramaticais"""
    
    def __init__(self, llm_client: DiagnosticLLMClient):
        self.llm_client = llm_client
    
    async def analyze(
        self,
        user_text: str,
        ai_text: Optional[str] = None,
        valid_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analisa erros gramaticais no texto e identifica skills (SINKT)
        
        Args:
            user_text: Texto do usuário
            ai_text: Resposta do AI (para contexto)
            valid_skills: Lista de skill_ids válidas para SINKT semantic tagging
            
        Returns:
            Dicionário com:
            - errors: Lista de erros encontrados
            - correct_skills: Lista de skills usadas corretamente
            - linguistic_features: Features linguísticas extraídas
            - semantic_skill_mapping: Mapeamento semântico SINKT (skill_id -> confidence)
        """
        # Chamar LLM para análise (com valid_skills para skill tagging)
        result = await self.llm_client.analyze_grammar(user_text, ai_text, valid_skills)
        
        errors = []
        for error_data in result.get("errors", []):
            try:
                error = ErrorAnalysis(
                    error_type=ErrorType(error_data.get("error_type", "grammar")),
                    category=ErrorCategory(error_data["category"]) if error_data.get("category") else None,
                    skill_id=error_data.get("skill_id"),
                    original_text=error_data.get("original_text", ""),
                    corrected_text=error_data.get("corrected_text"),
                    explanation=error_data.get("explanation"),
                    severity=error_data.get("severity", "medium"),
                    linguistic_features=error_data.get("linguistic_features", {})
                )
                errors.append(error)
            except Exception as e:
                # Skip invalid error data
                continue
        
        # SINKT semantic tagging (se valid_skills fornecidas)
        semantic_skill_mapping = {}
        if valid_skills:
            semantic_mapping = await self.llm_client.semantic_skill_tagging(
                user_text=user_text,
                valid_skills=valid_skills
            )
            semantic_skill_mapping = semantic_mapping
        
        return {
            "errors": errors,
            "correct_skills": result.get("correct_skills", []),
            "linguistic_features": result.get("linguistic_features", {}),
            "semantic_skill_mapping": semantic_skill_mapping
        }

