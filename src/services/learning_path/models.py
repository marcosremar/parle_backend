"""
Pydantic models for Learning Path Service
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class NextSkillResponse(BaseModel):
    """Próxima habilidade recomendada"""
    skill_id: str
    skill_name: str
    category: str
    priority: str = Field(..., description="high, medium, low")
    reason: str = Field(..., description="Por que esta habilidade foi recomendada")
    mastery_probability: float = Field(..., ge=0.0, le=1.0)
    zpd_ready: bool = Field(False, description="Se está na Zona de Desenvolvimento Proximal")


class ReviewSkillResponse(BaseModel):
    """Habilidade para revisão"""
    skill_id: str
    skill_name: str
    mastery_probability: float
    last_practiced: Optional[datetime]
    days_since_practice: int
    review_urgency: str = Field(..., description="high, medium, low")
    reason: str


class ReviewSkillsResponse(BaseModel):
    """Lista de habilidades para revisar"""
    review_skills: List[ReviewSkillResponse]
    total: int


class LearningPathResponse(BaseModel):
    """Caminho de aprendizado completo"""
    current_skill: Optional[NextSkillResponse] = None
    next_skills: List[NextSkillResponse] = Field(default_factory=list)
    review_skills: List[ReviewSkillResponse] = Field(default_factory=list)
    estimated_completion: Optional[int] = Field(None, description="Dias estimados para completar")

