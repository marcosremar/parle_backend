"""
Bayesian Knowledge Tracing (BKT) Implementation
Algoritmo probabilístico para rastrear conhecimento do estudante
"""

from typing import Dict, Any, Optional
from .base_tracer import KnowledgeTracer


class BayesianKnowledgeTracer(KnowledgeTracer):
    """
    Implementação do Bayesian Knowledge Tracing (BKT)
    
    BKT modela o conhecimento como uma probabilidade que evolui com cada interação.
    Usa o Teorema de Bayes para atualizar crenças sobre o domínio do estudante.
    
    Parâmetros:
        p_L0: Probabilidade inicial de já saber a habilidade
        p_T: Probabilidade de aprender quando exposto ao conceito
        p_F: Probabilidade de esquecer entre sessões
        p_G: Probabilidade de acertar quando sabe (guess)
        p_S: Probabilidade de errar mesmo sabendo (slip)
    """
    
    def __init__(self, skill_params: Optional[Dict[str, float]] = None):
        """
        Inicializa o BKT tracer com parâmetros
        
        Args:
            skill_params: Dicionário com parâmetros do BKT. Se None, usa valores padrão.
        """
        if skill_params is None:
            skill_params = self._get_default_params()
        
        self.p_L = skill_params.get('p_L0', 0.2)  # Probabilidade inicial
        self.p_T = skill_params.get('p_T', 0.15)   # Probabilidade de aprender
        self.p_F = skill_params.get('p_F', 0.05)   # Probabilidade de esquecer
        self.p_G = skill_params.get('p_G', 0.85)   # Probabilidade de acertar sabendo
        self.p_S = skill_params.get('p_S', 0.3)    # Probabilidade de errar sabendo
        
        # Estado inicial
        self.initial_p_L = self.p_L
        self.interaction_count = 0
    
    @staticmethod
    def _get_default_params() -> Dict[str, float]:
        """Retorna parâmetros padrão do BKT"""
        return {
            'p_L0': 0.2,   # 20% chance de já saber
            'p_T': 0.15,   # 15% chance de aprender por tentativa
            'p_F': 0.05,   # 5% chance de esquecer
            'p_G': 0.85,   # 85% chance de acertar sabendo
            'p_S': 0.3     # 30% chance de errar sabendo (slip/guess)
        }
    
    def update_belief(self, correct: bool, context: Dict[str, Any] = None) -> float:
        """
        Atualiza a crença sobre o conhecimento usando Teorema de Bayes
        
        Args:
            correct: True se o estudante acertou, False se errou
            context: Contexto adicional (não usado no BKT básico, mas mantido para compatibilidade)
            
        Returns:
            Nova probabilidade de domínio (0.0 a 1.0)
        """
        # Calcular probabilidades condicionais
        if correct:
            # P(correto | sabe) e P(correto | não sabe)
            p_correct_given_L1 = self.p_G
            p_correct_given_L0 = self.p_S  # Chance de acertar por chute
        else:
            # P(errado | sabe) e P(errado | não sabe)
            p_correct_given_L1 = 1 - self.p_G  # Slip: erra mesmo sabendo
            p_correct_given_L0 = 1 - self.p_S  # Erra porque não sabe
        
        # Teorema de Bayes: P(L₁ | observação) = P(observação | L₁) * P(L₁) / P(observação)
        # P(observação) = P(observação | L₁) * P(L₁) + P(observação | L₀) * P(L₀)
        p_observation = (
            p_correct_given_L1 * self.p_L +
            p_correct_given_L0 * (1 - self.p_L)
        )
        
        # Evitar divisão por zero
        if p_observation == 0:
            p_observation = 0.0001
        
        # Atualizar crença usando Bayes
        p_L1_given_obs = (p_correct_given_L1 * self.p_L) / p_observation
        
        # Transição de estado: considerar aprendizado e esquecimento
        # P(L₁|t+1) = P(L₁|t) * (1 - p_F) + P(L₀|t) * p_T
        self.p_L = p_L1_given_obs * (1 - self.p_F) + (1 - p_L1_given_obs) * self.p_T
        
        # Garantir que está no range [0, 1]
        self.p_L = max(0.0, min(1.0, self.p_L))
        
        self.interaction_count += 1
        
        return self.p_L
    
    def predict_performance(self, skill_id: str = None) -> float:
        """
        Prediz a probabilidade de o estudante acertar uma questão desta habilidade
        
        Args:
            skill_id: ID da habilidade (não usado no BKT básico, mas mantido para compatibilidade)
            
        Returns:
            Probabilidade de acerto: P(correto) = P(correto|sabe) * P(sabe) + P(correto|não sabe) * P(não sabe)
        """
        p_correct = self.p_G * self.p_L + self.p_S * (1 - self.p_L)
        return max(0.0, min(1.0, p_correct))
    
    def get_mastery_probability(self) -> float:
        """
        Retorna a probabilidade atual de domínio
        
        Returns:
            Probabilidade de domínio (0.0 a 1.0)
        """
        return self.p_L
    
    def reset(self):
        """Reseta o estado do tracer para valores iniciais"""
        self.p_L = self.initial_p_L
        self.interaction_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do tracer"""
        return {
            'mastery_probability': self.p_L,
            'initial_probability': self.initial_p_L,
            'interaction_count': self.interaction_count,
            'parameters': {
                'p_L0': self.initial_p_L,
                'p_T': self.p_T,
                'p_F': self.p_F,
                'p_G': self.p_G,
                'p_S': self.p_S
            }
        }

