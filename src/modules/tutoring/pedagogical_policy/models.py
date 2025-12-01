"""
Pydantic models for Pedagogical Policy Service
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Strategy(str, Enum):
    """Estratégias pedagógicas"""

    TEACH = "teach"  # Ensino explícito (mastery < 0.3)
    REINFORCE = "reinforce"  # Reforço com prática (mastery 0.3-0.7)
    CHALLENGE = "challenge"  # Desafio avançado (mastery > 0.7)


class ScaffoldingType(str, Enum):
    """Tipos de scaffolding (suporte pedagógico)"""

    IMPLICIT = "implicit"  # Correção sutil (recast)
    EXPLICIT = "explicit"  # Correção direta


class CEFRLevel(str, Enum):
    """Níveis CEFR"""

    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class EmotionalState(str, Enum):
    """Estados emocionais do estudante"""

    MOTIVATED = "motivated"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    CONFIDENT = "confident"
    NEUTRAL = "neutral"


class PromptContext(BaseModel):
    """Contexto completo para composição de prompt"""

    scenario: dict[str, Any] | None = Field(None, description="Dados do cenário")
    cefr_level: CEFRLevel = Field(CEFRLevel.A1, description="Nível CEFR do estudante")
    native_language: str = Field("en", description="Língua nativa")
    target_skill: dict[str, Any] | None = Field(None, description="Habilidade em foco")
    mastery_probability: float = Field(0.0, ge=0.0, le=1.0, description="Probabilidade de domínio")
    cefr_details: dict[str, Any] = Field(
        default_factory=dict, description="Detalhes do progresso CEFR"
    )
    strategy: Strategy | None = Field(None, description="Estratégia pedagógica")
    emotional_state: EmotionalState = Field(EmotionalState.NEUTRAL, description="Estado emocional")
    conversation_history: list | None = Field(None, description="Histórico de conversa")
    interpretable_knowledge_state: dict[str, Any] | None = Field(
        None, description="Estado interpretável com recomendações e breakdown por dimensão"
    )
    current_turn_analysis: dict[str, Any] | None = Field(
        None, description="Análise do turno atual (erros, features, skills identificadas)"
    )
    session_analysis: dict[str, Any] | None = Field(
        None, description="Análise agregada da sessão (tendências, padrões, progresso)"
    )


class ComposePromptRequest(BaseModel):
    """Request para compor prompt"""

    context: dict[str, Any]  # Accept dict for flexibility


class ComposePromptResponse(BaseModel):
    """Response com prompt composto"""

    prompt: str
    strategy: Strategy
    scaffolding_type: ScaffoldingType
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyInfo(BaseModel):
    """Informação sobre uma estratégia"""

    strategy: Strategy
    description: str
    mastery_range: str
    use_cases: list[str]
