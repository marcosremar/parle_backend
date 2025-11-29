"""
Learning Path Module - Direct Python calls for Learning path navigation
"""

from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule


class LearningPathModule(BaseModule):
    """Learning Path Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("learning_path")
        self.navigator = None
        self.student_model_module = None
    
    async def _initialize(self) -> bool:
        """Initialize learning path navigator"""
        try:
            # Import navigator from local module
            from .learning_path.navigator import LearningPathNavigator
            
            self.navigator = LearningPathNavigator()
            
            # Try to get student model module for direct calls
            try:
                from src.modules import create
                self.student_model_module = create("student_model")
            except:
                self.student_model_module = None
            
            self.logger.info("✅ Learning Path Module initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Learning Path Module: {e}")
            return False
    
    async def get_next_skill(
        self,
        user_id: str,
        cefr_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get next recommended skill for student"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get student skills
            skill_masteries = []
            if self.student_model_module:
                # Use direct module call
                cefr_progress = await self.student_model_module.get_cefr_progress(user_id)
                # For now, return a placeholder
                # In production, would fetch actual skills from student model
                skill_masteries = []
            else:
                # Fallback: fetch from HTTP (if needed)
                skill_masteries = []
            
            # Get student profile for CEFR level
            if not cefr_level:
                if self.student_model_module:
                    cefr_progress = await self.student_model_module.get_cefr_progress(user_id)
                    cefr_level = cefr_progress.get("level", "A1")
                else:
                    cefr_level = "A1"
            
            # Get next skill
            next_skill = self.navigator.get_next_skill(
                user_id=user_id,
                mastery_list=skill_masteries,
                cefr_level=cefr_level
            )
            
            # Convert to dict if needed
            if hasattr(next_skill, 'dict'):
                return next_skill.dict()
            return next_skill
        except Exception as e:
            self.logger.error(f"❌ Failed to get next skill: {e}")
            raise
    
    async def get_review_skills(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get skills that need review (spaced repetition)"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get student skills
            skill_masteries = []
            if self.student_model_module:
                # Use direct module call
                skill_masteries = []
            else:
                skill_masteries = []
            
            # Get review skills
            review_skills = self.navigator.get_review_skills(
                user_id=user_id,
                mastery_list=skill_masteries
            )
            
            # Convert to dict if needed
            if hasattr(review_skills, 'dict'):
                return review_skills.dict()
            return review_skills
        except Exception as e:
            self.logger.error(f"❌ Failed to get review skills: {e}")
            raise
