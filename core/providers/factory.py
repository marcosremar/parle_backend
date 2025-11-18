"""
Provider Factory - Dynamic loading of providers based on configuration

This factory provides backward compatibility while using the new Registry system.
It automatically registers providers and uses the Registry for instantiation.

New code should use ProviderSelector for advanced features like fallback chains.
"""

import logging
from typing import Dict, Any, Optional, Type

# Import base classes
from .llm.base import BaseLLMProvider
from .tts.base import BaseTTSProvider
from .stt.base import BaseSTTProvider

# Import registry system
from .registry import ProviderRegistry, ProviderType, get_registry

logger = logging.getLogger(__name__)

# ============================================================================
# Auto-Registration of Built-in Providers
# ============================================================================

def _register_builtin_providers():
    """
    Register all built-in providers with the registry.
    Called automatically on first factory use.
    """
    registry = get_registry()

    # Import and register LLM providers
    try:
        from .llm.litellm_provider import LiteLLMProvider
        registry.register(
            ProviderType.LLM, "litellm", LiteLLMProvider,
            aliases=["groq", "openai", "anthropic", "gpt", "claude"],
            metadata={"priority": 100, "supports_transcription": True}
        )
    except ImportError as e:
        logger.warning(f"Failed to register LiteLLMProvider: {e}")

    try:
        from .llm.ultravox_http_provider import UltravoxHTTPProvider
        registry.register(
            ProviderType.LLM, "ultravox_http", UltravoxHTTPProvider,
            aliases=["ultravox"],
            metadata={"priority": 50}
        )
    except ImportError as e:
        logger.warning(f"Failed to register UltravoxHTTPProvider: {e}")

    # Import and register TTS providers
    try:
        from .tts.kokoro_http_provider import KokoroHTTPProvider
        registry.register(
            ProviderType.TTS, "kokoro_http", KokoroHTTPProvider,
            aliases=["kokoro"],
            metadata={"priority": 100}
        )
    except ImportError as e:
        logger.warning(f"Failed to register KokoroHTTPProvider: {e}")

    # NOTE: EdgeTTSProvider disabled - edge_tts dependency not used
    # try:
    #     from .tts.edge_tts_provider import EdgeTTSProvider
    #     registry.register(
    #         ProviderType.TTS, "edge_tts", EdgeTTSProvider,
    #         aliases=["edge"],
    #         metadata={"priority": 50}
    #     )
    # except ImportError as e:
    #     logger.warning(f"Failed to register EdgeTTSProvider: {e}")

    # STT providers use LLM providers that support transcription
    # No separate STT provider classes yet

    logger.info("✅ Built-in providers registered with ProviderRegistry")


# Auto-register on module load
_register_builtin_providers()


class ProviderFactory:
    """
    Factory for creating provider instances based on configuration.

    Now uses ProviderRegistry internally for Open/Closed Principle.
    Maintains backward compatibility with existing code.

    Usage:
        # Old way (still works)
        provider = ProviderFactory.create_llm_provider({"provider": "groq", ...})

        # New way (recommended)
        from .registry import get_registry, ProviderType
        registry = get_registry()
        provider_class = registry.get(ProviderType.LLM, "groq")
        provider = provider_class(...)
    """

    def __init__(self, registry: Optional[ProviderRegistry] = None):
        """
        Initialize factory.

        Args:
            registry: Provider registry (uses global if None)
        """
        self.registry = registry or get_registry()

    # ========================================================================
    # Backward Compatibility Properties (for existing code)
    # ========================================================================

    @property
    def LLM_PROVIDERS(self) -> Dict[str, Type]:
        """
        Legacy property for backward compatibility.
        Returns dict of registered LLM providers.
        """
        providers = {}
        for name in self.registry.list_providers(ProviderType.LLM):
            providers[name] = self.registry.get(ProviderType.LLM, name)
        return providers

    @property
    def TTS_PROVIDERS(self) -> Dict[str, Type]:
        """
        Legacy property for backward compatibility.
        Returns dict of registered TTS providers.
        """
        providers = {}
        for name in self.registry.list_providers(ProviderType.TTS):
            providers[name] = self.registry.get(ProviderType.TTS, name)
        return providers

    @property
    def STT_PROVIDERS(self) -> Dict[str, Type]:
        """
        Legacy property for backward compatibility.
        Returns dict of registered STT providers (LLM providers with transcription).
        """
        # For STT, we reuse LLM providers that support transcription
        providers = {}
        for name in self.registry.list_providers(ProviderType.LLM):
            metadata = self.registry.get_metadata(ProviderType.LLM, name)
            if metadata.get("supports_transcription", False):
                providers[name] = self.registry.get(ProviderType.LLM, name)
        return providers

    @classmethod
    def create_llm_provider(cls, config: Dict[str, Any]) -> BaseLLMProvider:
        """
        Create LLM provider based on configuration.

        Now uses ProviderRegistry internally for extensibility.

        Args:
            config: Configuration dictionary with 'provider' and provider-specific settings

        Example config:
            {
                "provider": "litellm",  # or "groq", "openai", "ultravox"
                "model": "groq/llama3-70b-8192",
                "api_key": "...",
                "temperature": 0.7
            }

        Returns:
            Configured LLM provider instance

        Raises:
            ValueError: If provider not registered
        """
        registry = get_registry()
        provider_name = config.get("provider", "litellm").lower()

        # Get provider class from registry
        try:
            provider_class = registry.get(ProviderType.LLM, provider_name)
        except ValueError as e:
            # Better error message with available providers
            available = registry.list_providers(ProviderType.LLM)
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. "
                f"Available providers: {available}"
            ) from e

        # Import for type checking (registry returns Type, not instance)
        from .llm.litellm_provider import LiteLLMProvider

        # Handle LiteLLM-based providers (groq, openai, anthropic)
        if provider_class == LiteLLMProvider or provider_name in ["litellm", "groq", "openai", "anthropic", "gpt", "claude"]:
            model = config.get("model", "groq/llama3-70b-8192")

            # Auto-prefix model name if using specific provider
            if provider_name == "groq" and not model.startswith("groq/"):
                model = f"groq/{model}"
            elif provider_name in ["openai", "gpt"] and not model.startswith("gpt"):
                model = "gpt-3.5-turbo"  # Default OpenAI model
            elif provider_name in ["anthropic", "claude"] and not model.startswith("claude"):
                model = "claude-3-sonnet-20240229"  # Default Anthropic model

            return provider_class(
                model=model,
                api_key=config.get("api_key"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 1000),
                verbose=config.get("verbose", False)
            )

        # For other providers, pass config directly
        return provider_class(**config)

    @classmethod
    def create_tts_provider(cls, config: Dict[str, Any]) -> BaseTTSProvider:
        """
        Create TTS provider based on configuration.

        Now uses ProviderRegistry internally for extensibility.

        Args:
            config: Configuration dictionary

        Example config:
            {
                "provider": "kokoro_http",  # or "kokoro", "edge_tts"
                "voice": "pf_dora",
                "language": "pt-BR",
                "base_url": "http://localhost:8101"
            }

        Returns:
            Configured TTS provider instance

        Raises:
            ValueError: If provider not registered
        """
        registry = get_registry()
        provider_name = config.get("provider", "kokoro_http").lower()

        # Get provider class from registry
        try:
            provider_class = registry.get(ProviderType.TTS, provider_name)
        except ValueError as e:
            # Better error message with available providers
            available = registry.list_providers(ProviderType.TTS)
            raise ValueError(
                f"Unknown TTS provider: {provider_name}. "
                f"Available providers: {available}"
            ) from e

        # Pass config directly to provider
        # Each provider knows how to handle its own config
        return provider_class(config=config)

    @classmethod
    def create_stt_provider(cls, config: Dict[str, Any]) -> BaseSTTProvider:
        """
        Create STT provider based on configuration.

        NOTE: Currently, STT uses LLM providers that support transcription
        (e.g., Groq Whisper, OpenAI Whisper via LiteLLM).

        Args:
            config: Configuration dictionary

        Example config:
            {
                "provider": "groq",  # or "openai", "litellm"
                "model": "whisper-large-v3",
                "api_key": "..."
            }

        Returns:
            Configured STT provider instance (LLM provider with transcription)

        Raises:
            ValueError: If provider not registered or doesn't support transcription
        """
        registry = get_registry()
        provider_name = config.get("provider", "groq").lower()

        # Check if it's an LLM provider with transcription support
        if registry.is_registered(ProviderType.LLM, provider_name):
            metadata = registry.get_metadata(ProviderType.LLM, provider_name)
            if metadata.get("supports_transcription", False):
                # Use LLM provider for transcription
                return cls.create_llm_provider(config)
            else:
                raise ValueError(
                    f"LLM provider '{provider_name}' does not support transcription. "
                    f"Try 'groq' or 'openai' instead."
                )

        # Future: dedicated STT providers
        # For now, only LLM providers with transcription are supported
        raise ValueError(
            f"Unknown STT provider: {provider_name}. "
            f"Currently supported: groq, openai, litellm"
        )

    @classmethod
    def create_from_settings(cls, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create all providers from a settings dictionary

        Args:
            settings: Complete settings with 'providers' section

        Returns:
            Dictionary with 'llm', 'tts', 'stt' provider instances
        """
        providers_config = settings.get("providers", {})

        providers = {}

        # Create LLM provider (simple)
        if "llm" in providers_config:
            try:
                providers["llm"] = cls.create_llm_provider(providers_config["llm"])
                logger.info(f"Created LLM provider: {providers_config['llm'].get('provider')}")
            except Exception as e:
                logger.error(f"Failed to create LLM provider: {e}")

        # Create complex LLM provider (for analysis/validation)
        if "llm_complex" in providers_config:
            try:
                providers["llm_complex"] = cls.create_llm_provider(providers_config["llm_complex"])
                logger.info(f"Created complex LLM provider: {providers_config['llm_complex'].get('provider')}")
            except Exception as e:
                logger.error(f"Failed to create complex LLM provider: {e}")

        # Create TTS provider
        if "tts" in providers_config:
            try:
                providers["tts"] = cls.create_tts_provider(providers_config["tts"])
                logger.info(f"Created TTS provider: {providers_config['tts'].get('provider')}")
            except Exception as e:
                logger.error(f"Failed to create TTS provider: {e}")

        # Create STT provider
        if "stt" in providers_config:
            try:
                providers["stt"] = cls.create_stt_provider(providers_config["stt"])
                logger.info(f"Created STT provider: {providers_config['stt'].get('provider')}")
            except Exception as e:
                logger.error(f"Failed to create STT provider: {e}")
        elif "llm" in providers:
            # Use LLM provider for STT if it supports transcription
            providers["stt"] = providers["llm"]
            logger.info("Using LLM provider for STT")

        return providers