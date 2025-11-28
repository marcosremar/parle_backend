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
            # Import student model service (with fallback)
            try:
                from src.services.student_model.app_complete import StudentModelService
                self.service = StudentModelService()
                self.logger.info("✅ Student Model Module initialized")
                return True
            except ImportError:
                # Fallback: service not available
                self.logger.warning("⚠️  Student model service not available, using fallback")
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
