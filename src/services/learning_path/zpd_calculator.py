"""
Zone of Proximal Development (ZPD) Calculator
Calcula quais habilidades o aluno está pronto para aprender
"""

from typing import List, Dict, Any, Optional
from loguru import logger


class ZPDCalculator:
    """
    Calculadora de Zona de Desenvolvimento Proximal (ZPD)
    
    ZPD = Habilidades que o aluno pode aprender com ajuda
    Baseado em:
    - Habilidades já dominadas (pré-requisitos)
    - Dificuldade da habilidade
    - Nível CEFR do aluno
    """
    
    # Pré-requisitos de habilidades (skill_id -> lista de pré-requisitos)
    PREREQUISITES = {
        "verb_conjugation_past": ["verb_conjugation_present"],
        "verb_conjugation_future": ["verb_conjugation_present", "verb_conjugation_past"],
        "articles_definite": ["vocabulary_basic"],
        "prepositions_basic": ["articles_definite"],
        "vocabulary_intermediate": ["vocabulary_basic"],
        "vocabulary_advanced": ["vocabulary_intermediate"],
        "pronunciation_advanced": ["pronunciation_basic"]
    }
    
    # Dificuldade por nível CEFR
    CEFR_DIFFICULTY_MAP = {
        "A1": ["beginner"],
        "A2": ["beginner", "intermediate"],
        "B1": ["beginner", "intermediate"],
        "B2": ["intermediate", "advanced"],
        "C1": ["intermediate", "advanced"],
        "C2": ["advanced"]
    }
    
    def is_zpd_ready(
        self,
        skill_id: str,
        skill_difficulty: str,
        user_cefr_level: str,
        mastered_skills: List[str],
        skill_mastery: Optional[Dict[str, float]] = None
    ):
        """
        Verifica se uma habilidade está na ZPD do aluno
        
        Args:
            skill_id: ID da habilidade
            skill_difficulty: Dificuldade da habilidade
            user_cefr_level: Nível CEFR do usuário
            mastered_skills: Lista de habilidades dominadas (mastery > 0.7)
            skill_mastery: Dicionário com mastery de todas as habilidades
            
        Returns:
            Tupla (está_pronto, razão)
        """
        # Verificar se dificuldade é apropriada para o nível CEFR
        allowed_difficulties = self.CEFR_DIFFICULTY_MAP.get(user_cefr_level, [])
        if skill_difficulty not in allowed_difficulties:
            return False, f"Dificuldade '{skill_difficulty}' não é apropriada para nível {user_cefr_level}"
        
        # Verificar pré-requisitos
        prerequisites = self.PREREQUISITES.get(skill_id, [])
        if prerequisites:
            missing_prereqs = [p for p in prerequisites if p not in mastered_skills]
            if missing_prereqs:
                return False, f"Faltam pré-requisitos: {', '.join(missing_prereqs)}"
        
        # Verificar se já dominou (não precisa mais praticar)
        if skill_mastery and skill_id in skill_mastery:
            if skill_mastery[skill_id] > 0.9:
                return False, "Habilidade já dominada (mastery > 90%)"
        
        return True, "Habilidade está na ZPD - aluno está pronto para aprender"
    
    def get_zpd_skills(
        self,
        all_skills: List[Dict[str, Any]],
        user_cefr_level: str,
        mastered_skills: List[str],
        skill_mastery: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Retorna lista de habilidades na ZPD do aluno
        
        Args:
            all_skills: Lista de todas as habilidades disponíveis
            user_cefr_level: Nível CEFR do usuário
            mastered_skills: Habilidades dominadas
            skill_mastery: Mastery de todas as habilidades
            
        Returns:
            Lista de habilidades na ZPD
        """
        zpd_skills = []
        
        for skill in all_skills:
            skill_id = skill.get("skill_id")
            difficulty = skill.get("difficulty", "beginner")
            
            is_ready, reason = self.is_zpd_ready(
                skill_id,
                difficulty,
                user_cefr_level,
                mastered_skills,
                skill_mastery
            )
            
            if is_ready:
                skill["zpd_reason"] = reason
                zpd_skills.append(skill)
        
        return zpd_skills

