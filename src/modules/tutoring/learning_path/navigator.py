"""
Learning Path Navigator
Lógica principal para navegação de caminhos de aprendizado
"""

from datetime import datetime
from typing import Any

from .spaced_repetition import SpacedRepetitionSystem
from .zpd_calculator import ZPDCalculator


class LearningPathNavigator:
    """
    Navegador de caminhos de aprendizado

    Decide qual habilidade o aluno deve praticar agora baseado em:
    - Habilidades com baixo domínio (precisam de atenção)
    - Habilidades na ZPD (prontas para aprender)
    - Spaced repetition (revisão de habilidades antigas)
    """

    def __init__(self):
        self.srs = SpacedRepetitionSystem()
        self.zpd_calculator = ZPDCalculator()

    def get_next_skill(
        self,
        user_id: str,
        skill_masteries: list[dict[str, Any]],
        user_cefr_level: str = "A1",
        all_skills: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """
        Retorna próxima habilidade recomendada para o aluno

        Args:
            user_id: ID do usuário
            skill_masteries: Lista de skill masteries do aluno
            user_cefr_level: Nível CEFR do usuário
            all_skills: Lista de todas as habilidades disponíveis

        Returns:
            Dicionário com próxima habilidade recomendada ou None
        """
        if not skill_masteries:
            # Se não tem histórico, retornar primeira habilidade básica
            if all_skills:
                beginner_skills = [s for s in all_skills if s.get("difficulty") == "beginner"]
                if beginner_skills:
                    return {
                        "skill_id": beginner_skills[0]["skill_id"],
                        "skill_name": beginner_skills[0].get(
                            "name", beginner_skills[0]["skill_id"]
                        ),
                        "category": beginner_skills[0].get("category", "unknown"),
                        "priority": "high",
                        "reason": "Primeira habilidade para iniciante",
                        "mastery_probability": 0.0,
                        "zpd_ready": True,
                    }
            return None

        # Criar dicionário de mastery por skill_id
        mastery_dict = {m["skill_id"]: m["mastery_probability"] for m in skill_masteries}

        # Identificar habilidades dominadas
        mastered_skills = [m["skill_id"] for m in skill_masteries if m["mastery_probability"] > 0.7]

        # Prioridade 1: Habilidades com mastery baixo (< 0.3) que precisam de atenção
        low_mastery = [m for m in skill_masteries if m["mastery_probability"] < 0.3]
        if low_mastery:
            # Ordenar por mastery (menor primeiro)
            low_mastery.sort(key=lambda x: x["mastery_probability"])
            skill = low_mastery[0]
            return {
                "skill_id": skill["skill_id"],
                "skill_name": skill.get("skill_name", skill["skill_id"]),
                "category": skill.get("category", "unknown"),
                "priority": "high",
                "reason": f"Mastery baixo ({skill['mastery_probability']:.0%}) - precisa de prática",
                "mastery_probability": skill["mastery_probability"],
                "zpd_ready": True,
            }

        # Prioridade 2: Habilidades na ZPD (prontas para aprender)
        if all_skills:
            # Filtrar habilidades que o aluno ainda não tem mastery
            available_skills = [s for s in all_skills if s["skill_id"] not in mastery_dict]

            zpd_skills = self.zpd_calculator.get_zpd_skills(
                available_skills, user_cefr_level, mastered_skills, mastery_dict
            )

            if zpd_skills:
                # Retornar primeira habilidade na ZPD
                skill = zpd_skills[0]
                return {
                    "skill_id": skill["skill_id"],
                    "skill_name": skill.get("name", skill["skill_id"]),
                    "category": skill.get("category", "unknown"),
                    "priority": "medium",
                    "reason": f"Pronto para aprender: {skill.get('zpd_reason', '')}",
                    "mastery_probability": 0.0,
                    "zpd_ready": True,
                }

        # Prioridade 3: Habilidades em aprendizado (0.3-0.7) que precisam de reforço
        learning_skills = [m for m in skill_masteries if 0.3 <= m["mastery_probability"] < 0.7]
        if learning_skills:
            # Ordenar por mastery (menor primeiro)
            learning_skills.sort(key=lambda x: x["mastery_probability"])
            skill = learning_skills[0]
            return {
                "skill_id": skill["skill_id"],
                "skill_name": skill.get("skill_name", skill["skill_id"]),
                "category": skill.get("category", "unknown"),
                "priority": "medium",
                "reason": f"Em aprendizado ({skill['mastery_probability']:.0%}) - continue praticando",
                "mastery_probability": skill["mastery_probability"],
                "zpd_ready": False,
            }

        # Se não há habilidades prioritárias, retornar None
        return None

    def get_review_skills(
        self, skill_masteries: list[dict[str, Any]], limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retorna habilidades que devem ser revisadas (spaced repetition)

        Args:
            skill_masteries: Lista de skill masteries
            limit: Número máximo de habilidades para retornar

        Returns:
            Lista de habilidades para revisar
        """
        review_skills = []

        for mastery in skill_masteries:
            mastery_prob = mastery.get("mastery_probability", 0.0)
            last_practiced = mastery.get("last_practiced")

            if last_practiced:
                if isinstance(last_practiced, str):
                    try:
                        from dateutil.parser import parse

                        last_practiced = parse(last_practiced)
                    except ImportError:
                        # Fallback: try ISO format
                        last_practiced = datetime.fromisoformat(
                            last_practiced.replace("Z", "+00:00")
                        )
                from datetime import timezone

                days_since = (datetime.now(timezone.utc) - last_practiced).days
            else:
                days_since = 999  # Nunca praticou

            # Calcular se deve revisar
            should_review, urgency = self.srs.should_review(
                mastery_prob, last_practiced, success_rate=mastery.get("success_rate", 0.5)
            )

            if should_review:
                review_skills.append(
                    {
                        "skill_id": mastery["skill_id"],
                        "skill_name": mastery.get("skill_name", mastery["skill_id"]),
                        "mastery_probability": mastery_prob,
                        "last_practiced": last_practiced,
                        "days_since_practice": days_since,
                        "review_urgency": urgency,
                        "reason": f"Revisão espaçada: {days_since} dias desde última prática",
                    }
                )

        # Ordenar por urgência e dias desde prática
        review_skills.sort(
            key=lambda x: (
                {"high": 0, "medium": 1, "low": 2}[x["review_urgency"]],
                -x["days_since_practice"],
            )
        )

        return review_skills[:limit]
