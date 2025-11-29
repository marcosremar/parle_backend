"""
Learning Path Module - Direct Python calls for learning path navigation
"""

from typing import Dict, Optional, Any

from src.modules.base_module import BaseModule


class LearningPathModule(BaseModule):
    """Learning Path Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("learning_path")
        self.navigator = None
    
    async def _initialize(self) -> bool:
        """Initialize learning path navigator"""
        try:
            # Import navigator from local module
            from .navigator import LearningPathNavigator
            
            self.navigator = LearningPathNavigator()
            self.logger.info("✅ Learning Path Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Learning path navigator not available: {e}")
            return True
    
    async def get_next_skill(
        self,
        user_id: str,
        cefr_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get next skill for user"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.navigator:
                return await self.navigator.get_next_skill(user_id=user_id, cefr_level=cefr_level)
            else:
                # Fallback: return default skill
                return {
                    "skill_id": "default_skill",
                    "cefr_level": cefr_level or "A1",
                    "reason": "Default skill (navigator not available)"
                }
        except Exception as e:
            self.logger.error(f"❌ Failed to get next skill: {e}")
            raise
