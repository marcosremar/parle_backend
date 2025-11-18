"""
Base class for LLM providers
All LLM providers must implement this interface

NOTE: This class follows Liskov Substitution Principle (LSP)
- All providers must implement ALL abstract methods
- Behavior must be consistent across implementations
- No mixed responsibilities (LLM only, not STT)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncIterator
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract base class for Language Model providers.
    All LLM providers (LiteLLM, Ultravox, etc.) must implement these methods.

    Follows SOLID principles:
    - Single Responsibility: Only LLM text generation
    - Open/Closed: Extend via subclasses, not modification
    - Liskov Substitution: All providers are interchangeable
    - Interface Segregation: Minimal, focused interface
    - Dependency Inversion: Depend on abstraction, not concrete classes
    """

    def __init__(self, **kwargs):
        """
        Initialize provider with configuration.

        Args:
            **kwargs: Provider-specific configuration
        """
        self.config = kwargs
        logger.info(f"Initializing {self.__class__.__name__} provider")

    # ========================================================================
    # Abstract Methods (MUST be implemented)
    # ========================================================================

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text response from prompt.

        This is the core LLM functionality. All providers MUST implement this.

        Args:
            prompt: User input prompt
            system_prompt: System context/instructions
            temperature: Sampling temperature (0-1, lower = more deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information:
            - name: Model name
            - provider: Provider name
            - max_tokens: Maximum context tokens
            - supports_streaming: Whether streaming is supported
            - capabilities: List of capabilities
        """
        pass

    # ========================================================================
    # Optional Methods (have default implementations)
    # ========================================================================

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate text response with streaming (optional).

        Default implementation falls back to non-streaming generate().
        Override for providers that support streaming.

        Args:
            prompt: User input prompt
            system_prompt: System context/instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they are generated

        Raises:
            NotImplementedError: If provider doesn't support streaming
        """
        # Default: fall back to non-streaming
        result = await self.generate(prompt, system_prompt, temperature, max_tokens, **kwargs)
        yield result

    async def is_available(self) -> bool:
        """
        Check if the provider is available and configured (async).

        Override for providers that need async health checks (API calls, etc.)

        Returns:
            True if provider is ready to use
        """
        return self.is_available_sync()

    def is_available_sync(self) -> bool:
        """
        Synchronous availability check.

        Use for quick checks without I/O. Override is_available() for async checks.

        Returns:
            True if provider is ready to use
        """
        return True

    def supports_streaming(self) -> bool:
        """
        Check if this provider supports streaming generation.

        Returns:
            True if streaming is supported
        """
        return False

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.

        Returns:
            List of language codes (ISO 639-1)
        """
        return ["pt", "en", "es", "fr", "de", "it", "ja", "ko", "zh"]

    def get_capabilities(self) -> List[str]:
        """
        Get list of provider capabilities.

        Returns:
            List of capabilities (e.g., ["chat", "completion", "function_calling"])
        """
        return ["chat", "completion"]

    async def initialize(self) -> None:
        """
        Initialize provider resources (async).

        Override if provider needs async initialization (loading models, etc.)
        Called by ProviderSelector before first use.
        """
        pass

    async def cleanup(self) -> None:
        """
        Cleanup provider resources (async).

        Override if provider needs cleanup (closing connections, etc.)
        Called by ProviderSelector on shutdown.
        """
        pass

    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(config={self.config})"