"""
Diagnostic Module - Direct Python calls for Diagnostic analysis
"""

from typing import Any

from src.modules.base_module import BaseModule


class DiagnosticModule(BaseModule):
    """Diagnostic Module for direct Python calls"""

    def __init__(self):
        super().__init__("diagnostic_module")
        self.grammar_analyzer = None
        self.vocabulary_analyzer = None
        self.complexity_analyzer = None
        self.progress_analyzer = None
        self.session_analyzer = None
        self.error_rate_analyzer = None
        self.feedback_generator = None
        self.task_relevance_analyzer = None
        self.asr_metadata_analyzer = None
        self.llm_client = None

    async def _initialize(self) -> bool:
        """Initialize diagnostic analyzers"""
        try:
            # Diagnostic analyzers not yet fully migrated - using fallback
            # TODO: Migrate analyzers to modules/tutoring/diagnostic/analyzers/
            self.logger.warning("⚠️  Diagnostic analyzers not fully implemented, using fallback")
            self.llm_client = None
            self.grammar_analyzer = None
            self.vocabulary_analyzer = None
            self.complexity_analyzer = None
            self.progress_analyzer = None
            self.session_analyzer = None
            self.error_rate_analyzer = None
            self.feedback_generator = None
            self.task_relevance_analyzer = None
            self.asr_metadata_analyzer = None
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize diagnostic module: {e}")
            return False

    async def analyze_turn(
        self,
        user_text: str,
        ai_text: str | None = None,
        valid_skills: list[str] | None = None,
    ) -> dict[str, Any]:
        """Analyze a conversation turn"""
        if not self.initialized:
            await self.initialize()

        try:
            # Analyze grammar errors
            grammar_result = await self.grammar_analyzer.analyze(
                user_text=user_text, ai_text=ai_text, valid_skills=valid_skills
            )

            grammar_errors = grammar_result.get("errors", [])
            correct_skills = grammar_result.get("correct_skills", [])
            linguistic_features = grammar_result.get("linguistic_features", {})

            # Analyze vocabulary
            vocab_result = await self.vocabulary_analyzer.analyze(user_text)

            # Analyze complexity
            complexity_result = await self.complexity_analyzer.analyze(user_text)

            # Analyze progress
            progress_result = await self.progress_analyzer.analyze(
                user_text=user_text, errors=grammar_errors
            )

            return {
                "errors": grammar_errors,
                "correct_skills": correct_skills,
                "linguistic_features": linguistic_features,
                "vocabulary": vocab_result,
                "complexity": complexity_result,
                "progress": progress_result,
            }
        except Exception as e:
            self.logger.error(f"❌ Turn analysis failed: {e}")
            raise

    async def estimate_level(self, user_text: str, metadata: dict | None = None) -> dict[str, Any]:
        """Estimate CEFR level from text"""
        if not self.initialized:
            await self.initialize()

        try:
            # Fallback: analyzer not available
            if not self.complexity_analyzer:
                return {"estimated_level": "A1", "confidence": 0.5, "features": {}}

            # Use complexity analyzer to estimate level
            result = await self.complexity_analyzer.analyze(user_text)
            estimated_level = result.get("estimated_cefr_level", "A1")

            return {
                "estimated_level": estimated_level,
                "confidence": result.get("confidence", 0.5),
                "features": result,
            }
        except Exception as e:
            self.logger.error(f"❌ Level estimation failed: {e}")
            raise

    async def extract_skills(
        self,
        user_text: str,
        valid_skills: list[str] | None = None,
        ai_text: str | None = None,
    ) -> dict[str, Any]:
        """Extract skills from text"""
        if not self.initialized:
            await self.initialize()

        try:
            # Fallback: analyzer not available
            if not self.grammar_analyzer:
                return {"skills": [], "errors": [], "linguistic_features": {}}

            # Use grammar analyzer to extract skills
            result = await self.grammar_analyzer.analyze(
                user_text=user_text, ai_text=ai_text, valid_skills=valid_skills
            )

            return {
                "skills": result.get("correct_skills", []),
                "errors": result.get("errors", []),
                "linguistic_features": result.get("linguistic_features", {}),
            }
        except Exception as e:
            self.logger.error(f"❌ Skill extraction failed: {e}")
            raise

    async def analyze_session(self, turns: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze a complete session"""
        if not self.initialized:
            await self.initialize()

        try:
            # Fallback: analyzer not available
            if not self.session_analyzer:
                return {}

            result = await self.session_analyzer.analyze(turns)
            return result
        except Exception as e:
            self.logger.error(f"❌ Session analysis failed: {e}")
            raise
