"""
Spaced Repetition System (SRS)
Algoritmo para calcular intervalos ideais de revisão
Baseado no algoritmo SuperMemo 2 (SM-2)
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger


class SpacedRepetitionSystem:
    """
    Sistema de Repetição Espaçada (SRS)
    
    Calcula quando uma habilidade deve ser revisada baseado em:
    - Mastery probability atual
    - Tempo desde última prática
    - Histórico de acertos/erros
    """
    
    def __init__(self):
        # Intervalos em dias baseados no nível de domínio
        self.base_intervals = {
            "beginner": 1,      # Revisar em 1 dia
            "learning": 3,      # Revisar em 3 dias
            "mastered": 7,      # Revisar em 7 dias
            "expert": 14        # Revisar em 14 dias
        }
    
    def get_mastery_category(self, mastery_probability: float) -> str:
        """Categoriza mastery probability"""
        if mastery_probability < 0.3:
            return "beginner"
        elif mastery_probability < 0.5:
            return "learning"
        elif mastery_probability < 0.7:
            return "mastered"
        else:
            return "expert"
    
    def calculate_review_interval(
        self,
        mastery_probability: float,
        last_practiced: Optional[datetime] = None,
        success_rate: float = 0.5
    ) -> int:
        """
        Calcula intervalo ideal para revisão em dias
        
        Args:
            mastery_probability: Probabilidade de domínio atual
            last_practiced: Data da última prática
            success_rate: Taxa de sucesso nas últimas tentativas (0.0 a 1.0)
            
        Returns:
            Intervalo em dias até próxima revisão
        """
        category = self.get_mastery_category(mastery_probability)
        base_interval = self.base_intervals[category]
        
        # Ajustar baseado na taxa de sucesso
        if success_rate > 0.8:
            # Alta taxa de sucesso: aumentar intervalo
            interval = int(base_interval * 1.5)
        elif success_rate < 0.5:
            # Baixa taxa de sucesso: diminuir intervalo
            interval = max(1, int(base_interval * 0.7))
        else:
            interval = base_interval
        
        return interval
    
    def should_review(
        self,
        mastery_probability: float,
        last_practiced: Optional[datetime],
        success_rate: float = 0.5
    ):
        """
        Determina se uma habilidade deve ser revisada agora
        
        Args:
            mastery_probability: Probabilidade de domínio
            last_practiced: Data da última prática
            success_rate: Taxa de sucesso
            
        Returns:
            Tupla (deve_revisar, urgência)
        """
        if last_practiced is None:
            return True, "high"  # Nunca praticou
        
        # Calcular intervalo ideal
        ideal_interval = self.calculate_review_interval(
            mastery_probability,
            last_practiced,
            success_rate
        )
        
        # Calcular dias desde última prática
        from datetime import timezone
        days_since = (datetime.now(timezone.utc) - last_practiced).days
        
        if days_since >= ideal_interval:
            # Já passou do intervalo ideal
            if days_since >= ideal_interval * 2:
                urgency = "high"
            elif days_since >= ideal_interval * 1.5:
                urgency = "medium"
            else:
                urgency = "low"
            return True, urgency
        
        return False, "low"
    
    def get_review_urgency(
        self,
        mastery_probability: float,
        days_since_practice: int
    ) -> str:
        """
        Calcula urgência de revisão
        
        Args:
            mastery_probability: Probabilidade de domínio
            days_since_practice: Dias desde última prática
            
        Returns:
            Urgência: high, medium, low
        """
        category = self.get_mastery_category(mastery_probability)
        ideal_interval = self.base_intervals[category]
        
        if days_since_practice >= ideal_interval * 2:
            return "high"
        elif days_since_practice >= ideal_interval:
            return "medium"
        else:
            return "low"

