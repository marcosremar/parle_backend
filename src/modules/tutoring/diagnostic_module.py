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
            # Import diagnostic analyzers
            from src.services.diagnostic_module.app_complete import (
                llm_client,
                grammar_analyzer,
                vocabulary_analyzer,
                complexity_analyzer,
                progress_analyzer,
                session_analyzer,
                error_rate_analyzer,
                feedback_generator,
                task_relevance_analyzer,
                asr_metadata_analyzer
            )
            
            # Initialize LLM client
            import aiohttp
            session = aiohttp.ClientSession()
            await llm_client.initialize(session)
            
            self.llm_client = llm_client
            self.grammar_analyzer = grammar_analyzer
            self.vocabulary_analyzer = vocabulary_analyzer
            self.complexity_analyzer = complexity_analyzer
            self.progress_analyzer = progress_analyzer
            self.session_analyzer = session_analyzer
            self.error_rate_analyzer = error_rate_analyzer
            self.feedback_generator = feedback_generator
            self.task_relevance_analyzer = task_relevance_analyzer
            self.asr_metadata_analyzer = asr_metadata_analyzer
            
            self.logger.info("✅ Diagnostic Module initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Diagnostic Module: {e}")
            return False
    
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
            # Analyze grammar errors
            grammar_result = await self.grammar_analyzer.analyze(
                user_text=user_text,
                ai_text=ai_text,
                valid_skills=valid_skills
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
                user_text=user_text,
                errors=grammar_errors
            )
            
            return {
                "errors": grammar_errors,
                "correct_skills": correct_skills,
                "linguistic_features": linguistic_features,
                "vocabulary": vocab_result,
                "complexity": complexity_result,
                "progress": progress_result
            }
        except Exception as e:
            self.logger.error(f"❌ Turn analysis failed: {e}")
            raise
    
    async def estimate_level(
        self,
        user_text: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Estimate CEFR level from text"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use complexity analyzer to estimate level
            result = await self.complexity_analyzer.analyze(user_text)
            estimated_level = result.get("estimated_cefr_level", "A1")
            
            return {
                "estimated_level": estimated_level,
                "confidence": result.get("confidence", 0.5),
                "features": result
            }
        except Exception as e:
            self.logger.error(f"❌ Level estimation failed: {e}")
            raise
    
    async def extract_skills(
        self,
        user_text: str,
        valid_skills: Optional[List[str]] = None,
        ai_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract skills from text"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use grammar analyzer to extract skills
            result = await self.grammar_analyzer.analyze(
                user_text=user_text,
                ai_text=ai_text,
                valid_skills=valid_skills
            )
            
            return {
                "skills": result.get("correct_skills", []),
                "errors": result.get("errors", []),
                "linguistic_features": result.get("linguistic_features", {})
            }
        except Exception as e:
            self.logger.error(f"❌ Skill extraction failed: {e}")
            raise
    
    async def analyze_session(
        self,
        turns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze a complete session"""
        if not self.initialized:
            await self.initialize()
        
        try:
            result = await self.session_analyzer.analyze(turns)
            return result
        except Exception as e:
            self.logger.error(f"❌ Session analysis failed: {e}")
            raise
