"""
Knowledge Analyzer Engine

Analyzes student knowledge and updates skill mastery.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class KnowledgeAnalyzer:
    """Analyzes turn and updates student knowledge."""

    def __init__(self, clients: dict[str, Any], get_skill_difficulty: Any | None = None) -> None:
        """
        Initialize knowledge analyzer.

        Args:
            clients: Dictionary of service clients
            get_skill_difficulty: Optional function to get skill difficulty
        """
        self.clients = clients
        self._get_skill_difficulty = get_skill_difficulty

    async def analyze_and_update_knowledge(
        self, user_id: str, target_skill: dict[str, Any], user_text: str, ai_text: str
    ) -> None:
        """
        Background task: Analyze turn and update student knowledge.

        This runs asynchronously after the main response is sent,
        so it doesn't block the conversation flow.

        Args:
            user_id: User ID
            target_skill: Target skill being practiced
            user_text: User's input text
            ai_text: AI's response text
        """
        try:
            logger.info(
                f"🔍 Background: Analyzing turn for user {user_id}, skill {target_skill.get('skill_id')}"
            )

            # Step 1: Get valid skills from SKILL_CEFR_MAP for SINKT semantic tagging
            valid_skills: list[str] = []
            try:
                from src.modules.tutoring.student_model.skill_registry import SKILL_CEFR_MAP

                for level_skills in SKILL_CEFR_MAP.values():
                    valid_skills.extend(level_skills)
            except Exception as e:
                logger.warning(f"Could not fetch valid skills: {e}")

            # Step 2: Analyze errors AND successes using speech_grader
            if "speech_grader" not in self.clients:
                logger.warning("speech_grader client not available")
                return

            analysis = await self.clients["speech_grader"].analyze_turn(
                user_text=user_text,
                ai_text=ai_text,
                valid_skills=valid_skills if valid_skills else None,
            )

            skill_id = target_skill.get("skill_id")

            # Extract linguistic features and semantic mapping from analysis
            linguistic_features = analysis.get("linguistic_features", {})
            semantic_skill_mapping = analysis.get("semantic_skill_mapping", {})
            correct_skills = analysis.get("correct_skills", [])

            # Process explicit successes
            success_skills = set(correct_skills)
            from ..constants import HIGH_CONFIDENCE_THRESHOLD

            for skill_id_item, confidence in semantic_skill_mapping.items():
                if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                    success_skills.add(skill_id_item)

            # Update student model for successes
            if "student_model" in self.clients:
                for usage_skill_id in success_skills:
                    skill_difficulty = None
                    if self._get_skill_difficulty:
                        try:
                            skill_difficulty = self._get_skill_difficulty(usage_skill_id)
                        except Exception:
                            pass

                    await self.clients["student_model"].assess(
                        user_id=user_id,
                        skill_id=usage_skill_id,
                        correct=True,
                        context={"type": "explicit_success", "source": "diagnostic_analysis"},
                        user_text=user_text,
                        ai_text=ai_text,
                        difficulty=skill_difficulty,
                        linguistic_features=linguistic_features,
                    )
                    logger.info(f"✅ Registered SUCCESS for skill {usage_skill_id}")

            # Process errors
            errors = analysis.get("errors", [])
            error_skills_processed = set()

            for error in errors:
                error_skill_id = error.get("skill_id")
                if error_skill_id:
                    skill_difficulty = None
                    if self._get_skill_difficulty:
                        try:
                            skill_difficulty = self._get_skill_difficulty(error_skill_id)
                        except Exception:
                            pass

                    error_linguistic_features = error.get(
                        "linguistic_features", linguistic_features
                    )

                    await self.clients["student_model"].assess(
                        user_id=user_id,
                        skill_id=error_skill_id,
                        correct=False,
                        context={
                            "type": "error",
                            "error_type": error.get("error_type"),
                            "explanation": error.get("explanation"),
                            "severity": error.get("severity"),
                        },
                        user_text=user_text,
                        ai_text=ai_text,
                        difficulty=skill_difficulty,
                        linguistic_features=error_linguistic_features,
                    )
                    error_skills_processed.add(error_skill_id)
                    logger.info(f"❌ Registered ERROR for skill {error_skill_id}")

            # Handle target skill logic (fallback/reinforcement)
            if skill_id:
                is_error = skill_id in error_skills_processed
                is_success = skill_id in success_skills

                if not is_error and not is_success:
                    # Inferred success: If user spoke and didn't make a mistake
                    target_skill_correct = analysis.get("target_skill_correct", False)
                    if not errors:
                        target_skill_correct = True

                    if target_skill_correct:
                        skill_difficulty = None
                        if self._get_skill_difficulty:
                            try:
                                skill_difficulty = self._get_skill_difficulty(skill_id)
                            except Exception:
                                pass

                        await self.clients["student_model"].assess(
                            user_id=user_id,
                            skill_id=skill_id,
                            correct=True,
                            context={"type": "implicit_success"},
                            user_text=user_text,
                            ai_text=ai_text,
                            difficulty=skill_difficulty,
                            linguistic_features=linguistic_features,
                        )
                        logger.info(f"✅ Registered IMPLICIT SUCCESS for target skill {skill_id}")

        except Exception as e:
            logger.error(f"❌ Error in background analysis: {e}", exc_info=True)
            # Don't raise - this is a background task, errors shouldn't affect main flow
