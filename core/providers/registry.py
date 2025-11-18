"""
Provider Registry - Dynamic registration system for providers
Implements Open/Closed Principle - extend without modifying
"""

import logging
from typing import Dict, Type, Optional, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Provider types"""
    LLM = "llm"
    STT = "stt"
    TTS = "tts"


class ProviderRegistry:
    """
    Central registry for all providers.
    Implements Open/Closed Principle - new providers can be added without modifying existing code.

    Usage:
        # Register a provider
        registry = ProviderRegistry()
        registry.register(ProviderType.LLM, "groq", GroqProvider)

        # Get provider class
        provider_class = registry.get(ProviderType.LLM, "groq")

        # List all providers
        llm_providers = registry.list_providers(ProviderType.LLM)
    """

    _instance = None  # Singleton

    def __new__(cls):
        """Singleton pattern - only one registry instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry"""
        if self._initialized:
            return

        self._providers: Dict[ProviderType, Dict[str, Type]] = {
            ProviderType.LLM: {},
            ProviderType.STT: {},
            ProviderType.TTS: {},
        }

        self._aliases: Dict[ProviderType, Dict[str, str]] = {
            ProviderType.LLM: {},
            ProviderType.STT: {},
            ProviderType.TTS: {},
        }

        self._metadata: Dict[str, Dict] = {}
        self._initialized = True

        logger.info("✅ ProviderRegistry initialized")

    def register(
        self,
        provider_type: ProviderType,
        name: str,
        provider_class: Type,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Register a provider.

        Args:
            provider_type: Type of provider (LLM, STT, TTS)
            name: Provider name (e.g., "groq", "openai")
            provider_class: Provider class
            aliases: Optional list of aliases (e.g., ["gpt"] for "openai")
            metadata: Optional metadata (priority, capabilities, etc.)

        Raises:
            ValueError: If provider already registered
        """
        # Validate provider type
        if not isinstance(provider_type, ProviderType):
            raise ValueError(f"Invalid provider type: {provider_type}")

        # Check if already registered
        if name in self._providers[provider_type]:
            logger.warning(f"⚠️ Provider {name} already registered for {provider_type.value}, overwriting")

        # Register provider
        self._providers[provider_type][name] = provider_class

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[provider_type][alias] = name
                logger.debug(f"Registered alias: {alias} → {name}")

        # Store metadata
        key = f"{provider_type.value}:{name}"
        self._metadata[key] = metadata or {}

        logger.info(f"✅ Registered provider: {provider_type.value}/{name} ({provider_class.__name__})")

    def get(
        self,
        provider_type: ProviderType,
        name: str,
        raise_if_not_found: bool = True
    ) -> Optional[Type]:
        """
        Get provider class by name.

        Args:
            provider_type: Type of provider
            name: Provider name or alias
            raise_if_not_found: Raise error if not found

        Returns:
            Provider class or None

        Raises:
            ValueError: If provider not found and raise_if_not_found=True
        """
        # Check aliases first
        if name in self._aliases[provider_type]:
            name = self._aliases[provider_type][name]

        # Get provider
        provider_class = self._providers[provider_type].get(name)

        if provider_class is None and raise_if_not_found:
            available = self.list_providers(provider_type)
            raise ValueError(
                f"Provider '{name}' not found for type {provider_type.value}. "
                f"Available: {available}"
            )

        return provider_class

    def list_providers(self, provider_type: ProviderType) -> List[str]:
        """
        List all registered providers of a type.

        Args:
            provider_type: Type of provider

        Returns:
            List of provider names
        """
        return list(self._providers[provider_type].keys())

    def get_metadata(self, provider_type: ProviderType, name: str) -> Dict:
        """
        Get provider metadata.

        Args:
            provider_type: Type of provider
            name: Provider name

        Returns:
            Metadata dictionary
        """
        key = f"{provider_type.value}:{name}"
        return self._metadata.get(key, {})

    def is_registered(self, provider_type: ProviderType, name: str) -> bool:
        """
        Check if provider is registered.

        Args:
            provider_type: Type of provider
            name: Provider name or alias

        Returns:
            True if registered
        """
        # Check aliases
        if name in self._aliases[provider_type]:
            return True

        return name in self._providers[provider_type]

    def unregister(self, provider_type: ProviderType, name: str) -> None:
        """
        Unregister a provider.

        Args:
            provider_type: Type of provider
            name: Provider name
        """
        if name in self._providers[provider_type]:
            del self._providers[provider_type][name]
            logger.info(f"Unregistered provider: {provider_type.value}/{name}")

        # Remove aliases
        aliases_to_remove = [
            alias for alias, target in self._aliases[provider_type].items()
            if target == name
        ]
        for alias in aliases_to_remove:
            del self._aliases[provider_type][alias]

        # Remove metadata
        key = f"{provider_type.value}:{name}"
        if key in self._metadata:
            del self._metadata[key]

    def clear(self, provider_type: Optional[ProviderType] = None) -> None:
        """
        Clear all providers or providers of a specific type.

        Args:
            provider_type: Type to clear (None = clear all)
        """
        if provider_type:
            self._providers[provider_type].clear()
            self._aliases[provider_type].clear()
            # Clear metadata for this type
            keys_to_remove = [k for k in self._metadata.keys() if k.startswith(f"{provider_type.value}:")]
            for key in keys_to_remove:
                del self._metadata[key]
            logger.info(f"Cleared all {provider_type.value} providers")
        else:
            for ptype in ProviderType:
                self._providers[ptype].clear()
                self._aliases[ptype].clear()
            self._metadata.clear()
            logger.info("Cleared all providers")


# ============================================================================
# Decorator for Auto-Registration
# ============================================================================

def register_provider(
    provider_type: ProviderType,
    name: str,
    aliases: Optional[List[str]] = None,
    **metadata
) -> Callable:
    """
    Decorator to auto-register providers.

    Usage:
        @register_provider(ProviderType.LLM, "groq", aliases=["groq-llm"])
        class GroqProvider(BaseLLMProvider):
            pass

    Args:
        provider_type: Type of provider
        name: Provider name
        aliases: Optional aliases
        **metadata: Additional metadata (priority, capabilities, etc.)

    Returns:
        Decorator function
    """
    def decorator(cls: Type) -> Type:
        """Register class"""
        registry = ProviderRegistry()
        registry.register(
            provider_type=provider_type,
            name=name,
            provider_class=cls,
            aliases=aliases,
            metadata=metadata
        )
        return cls

    return decorator


# ============================================================================
# Global Registry Instance
# ============================================================================

# Singleton instance
_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    """Get global registry instance"""
    return _registry
