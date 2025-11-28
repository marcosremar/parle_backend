"""
Intelligent Tutoring System Service Clients

Clients for student model, pedagogical policy, diagnostic module, and learning path services.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

from .base import BaseServiceClient, ServiceClientError

logger = logging.getLogger(__name__)


class StudentModelClient(BaseServiceClient):
    """Student Model service client"""

    def __init__(self) -> None:
        super().__init__("student_model", is_module_service=True)

    async def assess(
        self,
        user_id: str,
        skill_id: str,
        correct: bool,
        context: Optional[Dict[str, Any]] = None,
        user_text: Optional[str] = None,
        ai_text: Optional[str] = None,
        difficulty: Optional[float] = None,
        linguistic_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess student response and update knowledge.
        
        Args:
            user_id: User identifier
            skill_id: Skill identifier
            correct: Whether the answer was correct
            context: Optional context dictionary
            user_text: Optional user input text
            ai_text: Optional AI response text
            difficulty: Optional skill difficulty
            linguistic_features: Optional linguistic features
            
        Returns:
            Assessment result dictionary
        """
        try:
            json_data: Dict[str, Any] = {
                "skill_id": skill_id,
                "correct": correct,
                "context": context,
                "user_text": user_text,
                "ai_text": ai_text
            }
            if difficulty is not None:
                json_data["difficulty"] = difficulty
            if linguistic_features:
                json_data["linguistic_features"] = linguistic_features
            
            result = await self._post(
                f"/api/student/{user_id}/assess",
                json_data=json_data
            )
            return result
        except ServiceClientError as e:
            logger.warning(f"⚠️ Failed to assess student response: {e}")
            return {}

    async def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get student profile"""
        try:
            return await self._get(f"/api/student/{user_id}/profile")
        except ServiceClientError:
            logger.warning(f"⚠️ Student profile not found for {user_id}")
            return None

    async def get_focus_areas(self, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get focus areas for student"""
        try:
            result = await self._get(f"/api/student/{user_id}/focus_areas?limit={limit}")
            return result.get("focus_areas", [])
        except ServiceClientError:
            return []

    async def get_cefr_progress(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed CEFR progress for student"""
        try:
            return await self._get(f"/api/student/{user_id}/cefr_progress")
        except ServiceClientError:
            logger.warning(f"⚠️ CEFR progress not found for {user_id}")
            return None
    
    async def get_interpretable_knowledge_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get interpretable knowledge state with recommendations"""
        try:
            return await self._get(f"/api/student/{user_id}/interpretable_knowledge_state")
        except ServiceClientError:
            logger.warning(f"⚠️ Interpretable knowledge state not found for {user_id}")
            return None
    
    async def get_linguistic_error_patterns(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis of error patterns by linguistic features"""
        try:
            return await self._get(f"/api/student/{user_id}/linguistic_error_patterns")
        except ServiceClientError:
            logger.warning(f"⚠️ Linguistic error patterns not found for {user_id}")
            return None


class PedagogicalPolicyClient(BaseServiceClient):
    """Pedagogical Policy service client"""

    def __init__(self) -> None:
        super().__init__("pedagogical_policy", is_module_service=True)

    async def compose_prompt(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compose pedagogical prompt based on context.
        
        Args:
            context: Context dictionary with scenario, CEFR level, etc.
            
        Returns:
            Dictionary with prompt, strategy, and scaffolding_type
        """
        try:
            result = await self._post(
                "/api/prompt/compose",
                json_data={"context": context}
            )
            return result
        except ServiceClientError as e:
            logger.warning(f"⚠️ Failed to compose prompt: {e}")
            return {"prompt": "", "strategy": "teach", "scaffolding_type": "explicit"}


class DiagnosticModuleClient(BaseServiceClient):
    """Speech Grader service client"""

    def __init__(self) -> None:
        super().__init__("speech_grader", is_module_service=True)

    async def analyze_turn(
        self,
        user_text: str,
        ai_text: Optional[str] = None,
        language: str = "pt-BR",
        valid_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a conversation turn with SINKT semantic tagging.
        
        Args:
            user_text: User input text
            ai_text: Optional AI response text
            language: Language code
            valid_skills: Optional list of valid skill IDs
            
        Returns:
            Dictionary with errors, correct_skills, linguistic_features, etc.
        """
        try:
            json_data: Dict[str, Any] = {
                "user_text": user_text,
                "ai_text": ai_text,
                "language": language
            }
            if valid_skills:
                json_data["valid_skills"] = valid_skills
            
            result = await self._post(
                "/api/diagnostic/analyze_turn",
                json_data=json_data
            )
            return result
        except ServiceClientError as e:
            logger.warning(f"⚠️ Failed to analyze turn: {e}")
            return {
                "errors": [],
                "correct_skills": [],
                "linguistic_features": {},
                "semantic_skill_mapping": {},
                "summary": "Analysis failed"
            }

    async def estimate_level(self, text: str, language: str = "pt-BR") -> Dict[str, Any]:
        """Estimate CEFR level"""
        try:
            result = await self._post(
                "/api/diagnostic/estimate_level",
                json_data={"text": text, "language": language}
            )
            return result
        except ServiceClientError:
            return {"cefr_level": "A1", "confidence": 0.5}

    async def extract_skills(
        self,
        user_text: str,
        valid_skills: List[str],
        ai_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract skills from user text using dedicated LLM call.
        
        Args:
            user_text: User input text
            valid_skills: List of valid skill IDs
            ai_text: Optional AI response text for context
            
        Returns:
            Dictionary with skills, overall_linguistic_features, and summary
        """
        try:
            json_data: Dict[str, Any] = {
                "user_text": user_text,
                "valid_skills": valid_skills
            }
            if ai_text:
                json_data["ai_text"] = ai_text
            
            result = await self._post(
                "/api/diagnostic/extract_skills",
                json_data=json_data
            )
            return result
        except ServiceClientError as e:
            logger.warning(f"⚠️ Failed to extract skills: {e}")
            return {
                "skills": [],
                "overall_linguistic_features": {},
                "summary": "Skill extraction failed"
            }


class LearningPathClient(BaseServiceClient):
    """Learning Path Navigator service client"""

    def __init__(self) -> None:
        super().__init__("learning_path", is_module_service=True)

    async def get_next_skill(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get next recommended skill"""
        try:
            result = await self._get(f"/api/path/{user_id}/next")
            return result
        except ServiceClientError:
            logger.warning(f"⚠️ Failed to get next skill for {user_id}")
            return None

    async def get_review_skills(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get skills for review"""
        try:
            result = await self._get(f"/api/path/{user_id}/review?limit={limit}")
            return result.get("review_skills", [])
        except ServiceClientError:
            return []
