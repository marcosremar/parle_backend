"""
Pydantic models for Student Model Service API
Request/Response models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class CEFRLevel(str, Enum):
    """Níveis CEFR"""
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class SkillCategory(str, Enum):
    """Categorias de habilidades"""
    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    PRONUNCIATION = "pronunciation"
    COMPREHENSION = "comprehension"
    PRODUCTION = "production"


# Request Models
class AssessRequest(BaseModel):
    """Request para avaliar uma resposta do estudante"""
    skill_id: str = Field(..., description="ID da habilidade sendo praticada")
    correct: bool = Field(..., description="Se o estudante acertou ou errou")
    context: Optional[Dict[str, Any]] = Field(None, description="Contexto adicional da interação")
    user_text: Optional[str] = Field(None, description="Texto do estudante")
    ai_text: Optional[str] = Field(None, description="Resposta do AI")
    difficulty: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dificuldade IRT do skill (opcional, será calculada se não fornecida)")
    linguistic_features: Optional[Dict[str, Any]] = Field(None, description="Features linguísticas extraídas (tense, person, number, register, domain, etc.)")


class CreateUserRequest(BaseModel):
    """Request para criar um novo usuário"""
    user_id: str = Field(..., description="ID único do usuário")
    cefr_level: Optional[CEFRLevel] = Field(CEFRLevel.A1, description="Nível CEFR inicial")
    native_language: Optional[str] = Field("en", description="Língua nativa do estudante")


class CreateSkillRequest(BaseModel):
    """Request para criar uma nova habilidade"""
    skill_id: str = Field(..., description="ID único da habilidade")
    name: str = Field(..., description="Nome da habilidade")
    category: SkillCategory = Field(..., description="Categoria da habilidade")
    difficulty: Optional[str] = Field("beginner", description="Dificuldade")
    description: Optional[str] = Field(None, description="Descrição da habilidade")


# Response Models
class SkillMasteryResponse(BaseModel):
    """Resposta com domínio de uma habilidade"""
    skill_id: str
    skill_name: str
    mastery_probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidade de domínio (0-1)")
    difficulty: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dificuldade IRT do skill")
    attempts: int
    successes: int
    last_practiced: Optional[datetime]
    
    class Config:
        from_attributes = True


class StudentProfileResponse(BaseModel):
    """Perfil completo do estudante"""
    user_id: str
    cefr_level: CEFRLevel
    native_language: str
    total_skills: int
    mastered_skills: int  # Skills com mastery > 0.7
    learning_skills: int  # Skills com mastery 0.3-0.7
    beginner_skills: int  # Skills com mastery < 0.3
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class FocusAreaResponse(BaseModel):
    """Área de foco recomendada para o estudante"""
    skill_id: str
    skill_name: str
    category: str
    mastery_probability: float
    priority: str = Field(..., description="high, medium, low")
    reason: str = Field(..., description="Por que esta habilidade foi recomendada")


class AssessResponse(BaseModel):
    """Resposta após avaliar uma interação"""
    skill_id: str
    new_mastery_probability: float = Field(..., ge=0.0, le=1.0)
    previous_mastery_probability: float = Field(..., ge=0.0, le=1.0)
    improvement: float = Field(..., description="Mudança na probabilidade")
    mastery_status: str = Field(..., description="beginner, learning, mastered")


class SkillsListResponse(BaseModel):
    """Lista de todas as habilidades do estudante"""
    skills: List[SkillMasteryResponse]
    total: int


class FocusAreasResponse(BaseModel):
    """Top 3 áreas de foco"""
    focus_areas: List[FocusAreaResponse]
    total_recommended: int


class CEFRProgressDetailedResponse(BaseModel):
    """Resposta detalhada do progresso CEFR com breakdown interpretável"""
    current_estimated_level: str
    dimension_progress: Dict[str, float] = Field(..., description="Progresso por dimensão (grammar, vocabulary, pronunciation)")
    strong_skills: List[Dict[str, Any]] = Field(..., description="Top skills fortes (skill_id, mastery, level)")
    weak_skills: List[Dict[str, Any]] = Field(..., description="Top skills fracas (skill_id, mastery, level)")
    recommendations: List[str] = Field(..., description="Recomendações human-readable")
    cefr_details: Dict[str, Any] = Field(..., description="Detalhes completos por nível CEFR")
    linguistic_error_patterns: Optional[Dict[str, Any]] = Field(None, description="Análise de padrões de erro por feature linguística")


class LinguisticErrorPatternResponse(BaseModel):
    """Resposta com análise de padrões de erro por feature linguística"""
    total_interactions_analyzed: int = Field(..., description="Total de interações analisadas")
    features_analyzed: int = Field(..., description="Número de features linguísticas analisadas")
    problematic_features: List[Dict[str, Any]] = Field(..., description="Top features causando mais erros (error_rate > 50%, >= 5 tentativas)")
    mastered_features: List[Dict[str, Any]] = Field(..., description="Top features dominadas (error_rate < 20%, >= 5 tentativas)")
    all_features: List[Dict[str, Any]] = Field(..., description="Todas as features analisadas ordenadas por taxa de erro")
    summary: str = Field(..., description="Resumo human-readable dos padrões encontrados")

