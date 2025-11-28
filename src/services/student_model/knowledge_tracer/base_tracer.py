"""
Base interface for Knowledge Tracing algorithms
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class KnowledgeTracer(ABC):
    """
    Interface abstrata para algoritmos de Knowledge Tracing
    
    Permite trocar implementações (BKT, AKT, etc.) sem quebrar o código
    """
    
    @abstractmethod
    def update_belief(self, correct: bool, context: Dict[str, Any] = None) -> float:
        """
        Atualiza a crença sobre o conhecimento do estudante baseado em uma observação
        
        Args:
            correct: Se o estudante acertou (True) ou errou (False)
            context: Contexto adicional (opcional)
            
        Returns:
            Nova probabilidade de domínio (0.0 a 1.0)
        """
        pass
    
    @abstractmethod
    def predict_performance(self, skill_id: str) -> float:
        """
        Prediz a probabilidade de o estudante acertar uma questão desta habilidade
        
        Args:
            skill_id: ID da habilidade
            
        Returns:
            Probabilidade de acerto (0.0 a 1.0)
        """
        pass
    
    @abstractmethod
    def get_mastery_probability(self) -> float:
        """
        Retorna a probabilidade atual de domínio
        
        Returns:
            Probabilidade de domínio (0.0 a 1.0)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reseta o estado do tracer para valores iniciais"""
        pass

