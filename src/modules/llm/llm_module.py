"""
LLM Module - Direct Python calls for Language Model
"""

import os
from typing import Any

from src.modules.base_module import BaseModule
from src.core.config import get_config


class LLMModule(BaseModule):
    """LLM Module for direct Python calls"""

    def __init__(self):
        super().__init__("llm")
        self.client = None
        self.default_model = "groq/llama-3.1-8b-instant"
        self.config = None

    async def _initialize(self) -> bool:
        """Initialize LLM client"""
        try:
            # Get config to access API keys and settings
            self.config = get_config()
            
            # Import LiteLLM
            import litellm

            # Configure LiteLLM
            litellm.set_verbose = False

            # Get provider from config
            provider = self.config.llm.provider.lower()
            
            # Set API key from config or environment
            api_key = self.config.llm.api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
            if api_key:
                # Set API key for the provider
                if provider == "groq":
                    os.environ["GROQ_API_KEY"] = api_key
                elif provider == "openai":
                    os.environ["OPENAI_API_KEY"] = api_key
                # LiteLLM will pick up the API key from environment
                self.logger.debug(f"✅ API key configured for provider: {provider}")
            else:
                self.logger.warning("⚠️  No LLM API key found in config or environment. Set LLM_API_KEY or GROQ_API_KEY environment variable.")

            self.client = litellm
            
            # Get model from config or environment
            self.default_model = (
                self.config.llm.model or 
                os.getenv("LLM_DEFAULT_MODEL", "llama-3.1-8b-instant")
            )
            
            # Format model name with provider prefix if needed
            if "/" not in self.default_model and provider:
                self.default_model = f"{provider}/{self.default_model}"

            self.logger.info(f"✅ LLM Module initialized with model: {self.default_model}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM Module: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            model: Model to use (default: groq/llama-3.1-8b-instant)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt (converted to system message)
            **kwargs: Additional parameters

        Returns:
            Dict with text, model, usage, etc.
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Get config if not already loaded
            if not self.config:
                self.config = get_config()
            
            model_name = model or self.default_model
            
            # Use config values if not provided
            if max_tokens == 500:  # Default value
                max_tokens = self.config.llm.max_tokens
            if temperature == 0.7:  # Default value
                temperature = self.config.llm.temperature

            # Build messages list - Groq doesn't support system_prompt parameter
            # Convert system_prompt to system message in messages array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Remove system_prompt from kwargs if present (Groq doesn't support it)
            kwargs.pop("system_prompt", None)
            
            # Add timeout from config
            timeout = kwargs.pop("timeout", self.config.llm.timeout)

            response = await self.client.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )

            return {
                "text": response.choices[0].message.content,
                "model": model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            self.logger.error(f"❌ LLM generation failed: {e}")
            raise

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict[str, Any]:
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
            # Get config if not already loaded
            if not self.config:
                self.config = get_config()
            
            model_name = model or self.default_model
            
            # Use config values if not provided
            if max_tokens == 500:  # Default value
                max_tokens = self.config.llm.max_tokens
            if temperature == 0.7:  # Default value
                temperature = self.config.llm.temperature
            
            # Add timeout from config
            timeout = kwargs.pop("timeout", self.config.llm.timeout)

            response = await self.client.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )

            return {
                "message": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "model": model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            self.logger.error(f"❌ LLM chat failed: {e}")
            raise

    async def get_models(self) -> dict[str, Any]:
        """Get available models"""
        return {
            "models": [
                "groq/llama-3.1-8b-instant",
                "groq/llama-3.1-70b-versatile",
                "groq/mixtral-8x7b-32768",
            ],
            "default": self.default_model,
        }
