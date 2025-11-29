"""
Diagnostic Module - Direct Python calls for Diagnostic analysis
"""

from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule


class DiagnosticModule(BaseModule):
    """Diagnostic Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("diagnostic_module")
        self.analyzers = {}
    
    async def _initialize(self) -> bool:
        """Initialize diagnostic analyzers"""
        try:
            # Diagnostic analyzers not yet fully migrated - using fallback
            # TODO: Migrate analyzers to modules/tutoring/diagnostic/analyzers/
            self.logger.warning("⚠️  Diagnostic analyzers not fully implemented, using fallback")
            self.analyzers = {}
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Diagnostic module not fully available: {e}")
            self.analyzers = {}
            return True
    
    async def analyze_turn(
        self,
        user_text: str,
        ai_text: Optional[str] = None,
        valid_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze a conversation turn"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if not self.analyzers:
                return {"errors": [], "correct_skills": []}
            
            # Analyze grammar errors
            grammar_result = await self.analyzers["grammar"].analyze(
                user_text=user_text,
                ai_text=ai_text,
                valid_skills=valid_skills
            )
            
            return {
                "errors": grammar_result.get("errors", []),
                "correct_skills": grammar_result.get("correct_skills", []),
                "linguistic_features": grammar_result.get("linguistic_features", {})
            }
        except Exception as e:
            self.logger.error(f"❌ Turn analysis failed: {e}")
            return {"errors": [], "correct_skills": []}
