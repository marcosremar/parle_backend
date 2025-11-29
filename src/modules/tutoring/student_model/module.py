"""
Student Model Module - Direct Python calls for student model operations
"""

from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule


class StudentModelModule(BaseModule):
    """Student Model Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("student_model")
        self.service = None
    
    async def _initialize(self) -> bool:
        """Initialize student model service"""
        try:
            # Student model service not yet fully migrated - using fallback profile
            self.logger.warning("⚠️  Student model service not fully implemented, using fallback profile")
            self.service = None
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Student model service not available: {e}")
            self.service = None
            return True
    
    async def get_profile(self, user_id: str) -> Dict[str, Any]:
        """Get student profile"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service:
                return await self.service.get_profile(user_id)
            else:
                # Fallback: return basic profile
                return {
                    "user_id": user_id,
                    "cefr_level": "A1",
                    "skills": {}
                }
        except Exception as e:
            self.logger.error(f"❌ Failed to get profile: {e}")
            raise
    
    async def get_cefr_progress(self, user_id: str) -> Dict[str, Any]:
        """Get CEFR level progress for student"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.service and hasattr(self.service, 'get_cefr_progress'):
                return await self.service.get_cefr_progress(user_id)
            else:
                # Fallback: return basic CEFR progress
                profile = await self.get_profile(user_id)
                return {
                    "user_id": user_id,
                    "level": profile.get("cefr_level", "A1"),
                    "progress": 0.0,
                    "skills": profile.get("skills", {})
                }
        except Exception as e:
            self.logger.error(f"❌ Failed to get CEFR progress: {e}")
            raise
