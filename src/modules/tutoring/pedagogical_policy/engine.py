"""
Policy Engine - Decisão de estratégias pedagógicas
Baseado no mastery probability do estudante
"""

from typing import Dict, Any, Optional
try:
    from .models import Strategy, ScaffoldingType, EmotionalState, PromptContext
except ImportError:
    # Fallback: import from services
    try:
        from src.services.pedagogical_policy.models import Strategy, ScaffoldingType, EmotionalState, PromptContext
    except ImportError:
        # Last resort: define minimal stubs
        from enum import Enum
        class Strategy(str, Enum):
            TEACH = "teach"
            REINFORCE = "reinforce"
            CHALLENGE = "challenge"
        class ScaffoldingType(str, Enum):
            IMPLICIT = "implicit"
            EXPLICIT = "explicit"
        class EmotionalState(str, Enum):
            MOTIVATED = "motivated"
            FRUSTRATED = "frustrated"
            CONFUSED = "confused"
            CONFIDENT = "confident"
            NEUTRAL = "neutral"
        from pydantic import BaseModel
        from typing import Optional, Dict, Any
        class PromptContext(BaseModel):
            scenario: Optional[Dict[str, Any]] = None
            cefr_level: str = "A1"
            native_language: str = "en"
            target_skill: Optional[Dict[str, Any]] = None
            mastery_probability: float = 0.0
            cefr_details: Dict[str, Any] = {}
            strategy: Optional[Strategy] = None


class PolicyEngine:
    """
    Motor de decisão de estratégias pedagógicas
    
    Decide qual estratégia usar baseado no estado do estudante:
    - mastery_probability < 0.3: TEACH (ensino explícito)
    - mastery_probability 0.3-0.7: REINFORCE (prática guiada)
    - mastery_probability > 0.7: CHALLENGE (desafio avançado)
    """
    
    @staticmethod
    def decide_strategy(mastery_probability: float) -> Strategy:
        """
        Decide estratégia pedagógica baseado na probabilidade de domínio
        
        Args:
            mastery_probability: Probabilidade de domínio (0.0 a 1.0)
            
        Returns:
            Estratégia pedagógica apropriada
        """
        if mastery_probability < 0.3:
            return Strategy.TEACH
        elif mastery_probability < 0.7:
            return Strategy.REINFORCE
        else:
            return Strategy.CHALLENGE
    
    @staticmethod
    def decide_scaffolding(
        mastery_probability: float,
        emotional_state: EmotionalState = EmotionalState.NEUTRAL
    ) -> ScaffoldingType:
        """
        Decide tipo de scaffolding baseado no nível e estado emocional
        
        Args:
            mastery_probability: Probabilidade de domínio
            emotional_state: Estado emocional do estudante
            
        Returns:
            Tipo de scaffolding (implicit ou explicit)
        """
        # Se frustrado ou confuso, usar correção explícita
        if emotional_state in [EmotionalState.FRUSTRATED, EmotionalState.CONFUSED]:
            return ScaffoldingType.EXPLICIT
        
        # Se iniciante, usar correção explícita
        if mastery_probability < 0.3:
            return ScaffoldingType.EXPLICIT
        
        # Se intermediário ou avançado, usar correção implícita (recast)
        return ScaffoldingType.IMPLICIT
    
    @staticmethod
    def get_strategy_instructions(strategy: Strategy, mastery: float) -> Dict[str, Any]:
        """
        Retorna instruções detalhadas para uma estratégia
        
        Args:
            strategy: Estratégia pedagógica
            mastery: Probabilidade de domínio
            
        Returns:
            Dicionário com instruções para o prompt
        """
        if strategy == Strategy.TEACH:
            return {
                "approach": "explicit_teaching",
                "instruction": "Ensine explicitamente o conceito. Use exemplos simples e claros.",
                "correction_style": "direct",
                "pace": "slow",
                "complexity": "simple",
                "feedback": "immediate_and_detailed"
            }
        elif strategy == Strategy.REINFORCE:
            return {
                "approach": "guided_practice",
                "instruction": "Faça perguntas para testar o conhecimento. Dê dicas apenas se o aluno errar.",
                "correction_style": "recast",
                "pace": "normal",
                "complexity": "moderate",
                "feedback": "encouraging"
            }
        else:  # CHALLENGE
            return {
                "approach": "advanced_challenge",
                "instruction": "Desafie com casos complexos e exceções. Fale naturalmente como um nativo.",
                "correction_style": "subtle",
                "pace": "natural",
                "complexity": "high",
                "feedback": "minimal"
            }
    
    @staticmethod
    def get_emotional_modulation(emotional_state: EmotionalState) -> Dict[str, str]:
        """
        Retorna modulações baseadas no estado emocional
        
        Args:
            emotional_state: Estado emocional do estudante
            
        Returns:
            Dicionário com instruções de modulação
        """
        modulations = {
            EmotionalState.MOTIVATED: {
                "tone": "encouraging",
                "instruction": "O aluno está motivado. Pode aumentar levemente a complexidade.",
                "pace": "normal_to_fast"
            },
            EmotionalState.FRUSTRATED: {
                "tone": "patient_and_supportive",
                "instruction": "O aluno está frustrado. Seja muito paciente e encorajador. Simplifique a linguagem.",
                "pace": "slow"
            },
            EmotionalState.CONFUSED: {
                "tone": "clarifying",
                "instruction": "O aluno parece confuso. Explique de forma mais clara e use exemplos simples.",
                "pace": "slow"
            },
            EmotionalState.CONFIDENT: {
                "tone": "challenging",
                "instruction": "O aluno está confiante. Pode introduzir desafios e complexidade.",
                "pace": "normal"
            },
            EmotionalState.NEUTRAL: {
                "tone": "neutral",
                "instruction": "Mantenha um tom natural e equilibrado.",
                "pace": "normal"
            }
        }
        return modulations.get(emotional_state, modulations[EmotionalState.NEUTRAL])
    
    @staticmethod
    def process_context(context: PromptContext) -> Dict[str, Any]:
        """
        Processa contexto completo e retorna todas as decisões pedagógicas
        
        Args:
            context: Contexto completo do prompt
            
        Returns:
            Dicionário com todas as decisões e instruções
        """
        # Decidir estratégia se não fornecida
        strategy = context.strategy or PolicyEngine.decide_strategy(context.mastery_probability)
        
        # Decidir scaffolding
        scaffolding = PolicyEngine.decide_scaffolding(
            context.mastery_probability,
            context.emotional_state
        )
        
        # Obter instruções da estratégia
        strategy_instructions = PolicyEngine.get_strategy_instructions(
            strategy,
            context.mastery_probability
        )
        
        # Obter modulação emocional
        emotional_modulation = PolicyEngine.get_emotional_modulation(
            context.emotional_state
        )
        
        return {
            "strategy": strategy,
            "scaffolding_type": scaffolding,
            "strategy_instructions": strategy_instructions,
            "emotional_modulation": emotional_modulation,
            "cefr_level": context.cefr_level,
            "target_skill": context.target_skill,
            "mastery_probability": context.mastery_probability
        }

