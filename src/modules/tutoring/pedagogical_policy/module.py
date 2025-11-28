"""
Pedagogical Policy Module - Direct Python calls for prompt composition
"""

from typing import Dict, Optional, Any
from loguru import logger

from src.modules.base_module import BaseModule
from .engine import PolicyEngine


class PedagogicalPolicyModule(BaseModule):
    """Pedagogical Policy Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("pedagogical_policy")
        self.engine = None
    
    async def _initialize(self) -> bool:
        """Initialize policy engine"""
        try:
            from .engine import PolicyEngine
            self.engine = PolicyEngine()
            self.logger.info("✅ Pedagogical Policy Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Policy engine not available: {e}")
            return True
    
    async def compose_prompt(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compose pedagogical prompt"""
        if not self.initialized:
            await self.initialize()
        
        try:
            if self.engine:
                result = await self.engine.compose_prompt(context=context)
                return result
            else:
                # Fallback: return basic prompt
                return {
                    "system_prompt": context.get("system_prompt", "You are a helpful language tutor."),
                    "user_prompt": context.get("user_message", "")
                }
        except Exception as e:
            self.logger.error(f"❌ Prompt composition failed: {e}")
            raise
