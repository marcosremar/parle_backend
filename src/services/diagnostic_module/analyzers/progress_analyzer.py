"""
Progress Analyzer - Análise de progresso temporal
"""

from typing import Dict, Any, Optional, List
from ..models import ProgressAnalysis


class ProgressAnalyzer:
    """Analisador de progresso do estudante"""
    
    def analyze(
        self,
        current_errors: List[Dict[str, Any]],
        previous_errors: Optional[List[Dict[str, Any]]] = None,
        current_mastery: Optional[Dict[str, float]] = None,
        previous_mastery: Optional[Dict[str, float]] = None
    ) -> ProgressAnalysis:
        """
        Analisa progresso comparando estado atual com anterior
        
        Args:
            current_errors: Erros atuais
            previous_errors: Erros anteriores (opcional)
            current_mastery: Mastery atual por skill
            previous_mastery: Mastery anterior por skill
            
        Returns:
            Análise de progresso
        """
        improved = False
        regression = False
        improvement_areas = []
        regression_areas = []
        
        # Comparar mastery
        if current_mastery and previous_mastery:
            for skill_id, current_prob in current_mastery.items():
                previous_prob = previous_mastery.get(skill_id, 0.0)
                if current_prob > previous_prob + 0.1:  # Melhoria significativa
                    improved = True
                    improvement_areas.append(skill_id)
                elif current_prob < previous_prob - 0.1:  # Regressão significativa
                    regression = True
                    regression_areas.append(skill_id)
        
        # Comparar erros
        if previous_errors:
            current_error_count = len(current_errors)
            previous_error_count = len(previous_errors)
            
            if current_error_count < previous_error_count * 0.8:  # 20% menos erros
                improved = True
            elif current_error_count > previous_error_count * 1.2:  # 20% mais erros
                regression = True
        
        # Determinar tendência geral
        if improved and not regression:
            trend = "improving"
        elif regression and not improved:
            trend = "declining"
        else:
            trend = "stable"
        
        return ProgressAnalysis(
            improved=improved,
            regression=regression,
            improvement_areas=improvement_areas,
            regression_areas=regression_areas,
            overall_trend=trend
        )

