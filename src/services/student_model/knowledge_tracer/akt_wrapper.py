"""
AKT (Attentive Knowledge Tracing) Wrapper - DEPRECATED
Este arquivo está deprecated. Use AttentiveKnowledgeTracer de akt_tracer.py

Mantido apenas para compatibilidade. A implementação real está em akt_tracer.py.
"""

from typing import Dict, Any
from .base_tracer import KnowledgeTracer
from .akt_tracer import AttentiveKnowledgeTracer


class AKTTracer(KnowledgeTracer):
    """
    Wrapper deprecated - Use AttentiveKnowledgeTracer diretamente
    
    Esta classe redireciona para AttentiveKnowledgeTracer para manter compatibilidade.
    """
    
    def __init__(self, skill_params: Dict[str, Any] = None):
        """Inicializa wrapper que delega para AttentiveKnowledgeTracer"""
        self._tracer = AttentiveKnowledgeTracer(skill_params)
    
    def update_belief(self, correct: bool, context: Dict[str, Any] = None) -> float:
        """Delega para AttentiveKnowledgeTracer"""
        return self._tracer.update_belief(correct, context)
    
    def predict_performance(self, skill_id: str = None) -> float:
        """Delega para AttentiveKnowledgeTracer"""
        return self._tracer.predict_performance(skill_id)
    
    def get_mastery_probability(self) -> float:
        """Delega para AttentiveKnowledgeTracer"""
        return self._tracer.get_mastery_probability()
    
    def reset(self):
        """Delega para AttentiveKnowledgeTracer"""
        self._tracer.reset()

