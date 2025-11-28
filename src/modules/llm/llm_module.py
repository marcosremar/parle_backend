"""
LLM Module - Direct Python calls for Language Model
"""

import os
from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule


class LLMModule(BaseModule):
    """LLM Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("llm")
        self.client = None
        self.default_model = "groq/llama-3.1-8b-instant"
    
    async def _initialize(self) -> bool:
        """Initialize LLM client"""
        try:
            # Import LiteLLM
            import litellm
            
            # Configure LiteLLM
            litellm.set_verbose = False
            
            self.client = litellm
            self.default_model = os.getenv("LLM_DEFAULT_MODEL", "groq/llama-3.1-8b-instant")
            
            self.logger.info(f"✅ LLM Module initialized with model: {self.default_model}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM Module: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            model: Model to use (default: groq/llama-3.1-8b-instant)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Dict with text, model, usage, etc.
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            model_name = model or self.default_model
            
            response = await self.client.acompletion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return {
                "text": response.choices[0].message.content,
                "model": model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            self.logger.error(f"❌ LLM generation failed: {e}")
            raise
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion
        
        Args:
            messages: List of message dicts with role and content
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Dict with message, model, usage
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            model_name = model or self.default_model
            
            response = await self.client.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return {
                "message": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "model": model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            self.logger.error(f"❌ LLM chat failed: {e}")
            raise
    
    async def get_models(self) -> Dict[str, Any]:
        """Get available models"""
        return {
            "models": [
                "groq/llama-3.1-8b-instant",
                "groq/llama-3.1-70b-versatile",
                "groq/mixtral-8x7b-32768"
            ],
            "default": self.default_model
        }
