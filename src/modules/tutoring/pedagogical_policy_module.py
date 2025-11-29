"""
Pedagogical Policy Module - Direct Python calls for Pedagogical policy
"""

from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule


class PedagogicalPolicyModule(BaseModule):
    """Pedagogical Policy Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("pedagogical_policy")
        self.policy_engine = None
        self.prompt_composer = None
    
    async def _initialize(self) -> bool:
        """Initialize pedagogical policy engine"""
        try:
            # Import from local module
            from .pedagogical_policy.module import PedagogicalPolicyModule
            
            # Get policy engine from module
            policy_module = PedagogicalPolicyModule()
            await policy_module.initialize()
            self.policy_engine = policy_module.engine if hasattr(policy_module, 'engine') else None
            self.prompt_composer = None  # Not yet implemented in module
            
            self.logger.info("✅ Pedagogical Policy Module initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Pedagogical Policy Module: {e}")
            return False
    
    async def compose_prompt(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compose a pedagogical prompt based on context"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Convert dict to PromptContext if needed
            from .pedagogical_policy.models import (
                PromptContext,
                CEFRLevel,
                EmotionalState
            )
            
            if isinstance(context, dict):
                prompt_context = PromptContext(
                    scenario=context.get("scenario"),
                    cefr_level=CEFRLevel(context.get("cefr_level", "A1")),
                    native_language=context.get("native_language", "en"),
                    target_skill=context.get("target_skill"),
                    mastery_probability=context.get("mastery_probability", 0.0),
                    strategy=None,  # Will be decided by policy engine
                    emotional_state=EmotionalState(context.get("emotional_state", "neutral")),
                    conversation_history=context.get("conversation_history"),
                    cefr_details=context.get("cefr_details") or {},
                    interpretable_knowledge_state=context.get("interpretable_knowledge_state"),
                    current_turn_analysis=context.get("current_turn_analysis"),
                    session_analysis=context.get("session_analysis")
                )
            else:
                prompt_context = context
            
            # Compose prompt
            result = self.prompt_composer.compose(prompt_context)
            
            # Convert to dict if needed
            if hasattr(result, 'dict'):
                return result.dict()
            return result
        except Exception as e:
            self.logger.error(f"❌ Prompt composition failed: {e}")
            raise
    
    async def get_strategies(self) -> list:
        """Get available pedagogical strategies"""
        if not self.initialized:
            await self.initialize()
        
        try:
            from .pedagogical_policy.models import StrategyInfo, Strategy
            
            strategies = [
                StrategyInfo(
                    strategy=Strategy.TEACH,
                    description="Ensino explícito para alunos iniciantes (mastery < 30%)",
                    mastery_range="0% - 30%",
                    use_cases=[
                        "Aluno está aprendendo um conceito pela primeira vez",
                        "Aluno demonstra dificuldade consistente",
                        "Introdução de novo tópico"
                    ]
                ),
                StrategyInfo(
                    strategy=Strategy.REINFORCE,
                    description="Reforço com prática guiada para alunos intermediários (mastery 30-70%)",
                    mastery_range="30% - 70%",
                    use_cases=[
                        "Aluno está praticando um conceito conhecido",
                        "Consolidação de conhecimento",
                        "Aumento de fluência"
                    ]
                ),
                StrategyInfo(
                    strategy=Strategy.CHALLENGE,
                    description="Desafio avançado para alunos proficientes (mastery > 70%)",
                    mastery_range="70% - 100%",
                    use_cases=[
                        "Aluno domina o conceito básico",
                        "Introdução de exceções e casos complexos",
                        "Avanço para próximo nível"
                    ]
                )
            ]
            
            # Convert to list of dicts
            return [
                s.dict() if hasattr(s, 'dict') else s
                for s in strategies
            ]
        except Exception as e:
            self.logger.error(f"❌ Failed to get strategies: {e}")
            return []
