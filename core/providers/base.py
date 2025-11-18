"""
Base classes for provider system
Defines interfaces that all providers must implement

This module provides:
- BaseLLMProvider: Language Model providers (text generation)
- BaseSTTProvider: Speech-to-Text providers (audio transcription)
- BaseTTSProvider: Text-to-Speech providers (audio synthesis)
- ProviderRegistry: Dynamic provider registration system
- ProviderSelector: Strategy pattern with fallback chain

All providers follow SOLID principles:
- Single Responsibility: Each provider type has one clear purpose
- Open/Closed: Extend via subclasses, registry, and decorators
- Liskov Substitution: All providers of same type are interchangeable
- Interface Segregation: Minimal, focused interfaces
- Dependency Inversion: Depend on abstractions (base classes)
"""

# Import base classes
from src.core.providers.llm.base import BaseLLMProvider
from src.core.providers.stt.base import BaseSTTProvider
from src.core.providers.tts.base import BaseTTSProvider

# Import registry and selector
from src.core.providers.registry import (
    ProviderRegistry,
    ProviderType,
    register_provider,
    get_registry
)

from src.core.providers.selector import (
    ProviderSelector,
    ProviderConfig,
    create_selector
)

# Export all public APIs
__all__ = [
    # Base classes
    "BaseLLMProvider",
    "BaseSTTProvider",
    "BaseTTSProvider",
    # Registry
    "ProviderRegistry",
    "ProviderType",
    "register_provider",
    "get_registry",
    # Selector
    "ProviderSelector",
    "ProviderConfig",
    "create_selector",
]