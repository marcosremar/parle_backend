"""
Provider Selector - Strategy Pattern implementation with fallback chain
Implements Open/Closed Principle - easy to add new selection strategies
"""

import logging
from typing import List, Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
import asyncio

from .registry import ProviderRegistry, ProviderType

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a provider in the chain"""
    name: str
    priority: int = 0  # Higher = higher priority
    config: Dict[str, Any] = None  # Provider-specific config
    health_check: Optional[Callable] = None  # Optional health check function

    def __post_init__(self):
        if self.config is None:
            self.config = {}


class ProviderSelector:
    """
    Selects providers based on priority and availability.
    Implements fallback chain: Primary → Secondary → Tertiary

    Usage:
        selector = ProviderSelector(ProviderType.LLM)

        # Add providers to chain
        selector.add_provider("groq", priority=100, config={...})
        selector.add_provider("openai", priority=50, config={...})
        selector.add_provider("ultravox", priority=10, config={...})

        # Get best available provider
        provider = await selector.get_provider()

        # Use with automatic fallback
        result = await selector.execute(lambda p: p.generate("Hello"))
    """

    def __init__(
        self,
        provider_type: ProviderType,
        registry: Optional[ProviderRegistry] = None,
        auto_retry: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize selector.

        Args:
            provider_type: Type of provider to select
            registry: Provider registry (uses global if None)
            auto_retry: Automatically retry with fallback on failure
            max_retries: Maximum retries per provider
        """
        from .registry import get_registry

        self.provider_type = provider_type
        self.registry = registry or get_registry()
        self.auto_retry = auto_retry
        self.max_retries = max_retries

        # Chain of providers (sorted by priority)
        self._chain: List[ProviderConfig] = []

        # Instantiated providers cache
        self._instances: Dict[str, Any] = {}

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "provider_usage": {}
        }

        logger.info(f"✅ ProviderSelector initialized for {provider_type.value}")

    def add_provider(
        self,
        name: str,
        priority: int = 0,
        config: Optional[Dict[str, Any]] = None,
        health_check: Optional[Callable] = None
    ) -> "ProviderSelector":
        """
        Add provider to the chain.

        Args:
            name: Provider name
            priority: Priority (higher = preferred)
            config: Provider configuration
            health_check: Optional async health check function

        Returns:
            Self (for chaining)

        Raises:
            ValueError: If provider not registered
        """
        # Validate provider exists
        if not self.registry.is_registered(self.provider_type, name):
            raise ValueError(
                f"Provider '{name}' not registered for {self.provider_type.value}"
            )

        # Create config
        provider_config = ProviderConfig(
            name=name,
            priority=priority,
            config=config or {},
            health_check=health_check
        )

        # Add to chain
        self._chain.append(provider_config)

        # Sort by priority (descending)
        self._chain.sort(key=lambda p: p.priority, reverse=True)

        logger.info(
            f"Added provider to chain: {name} (priority={priority}). "
            f"Chain order: {[p.name for p in self._chain]}"
        )

        return self

    def remove_provider(self, name: str) -> None:
        """
        Remove provider from chain.

        Args:
            name: Provider name
        """
        self._chain = [p for p in self._chain if p.name != name]

        # Remove from cache
        if name in self._instances:
            del self._instances[name]

        logger.info(f"Removed provider from chain: {name}")

    async def get_provider(
        self,
        check_health: bool = True
    ) -> Optional[Any]:
        """
        Get best available provider from chain.

        Args:
            check_health: Run health checks

        Returns:
            Provider instance or None if all failed

        Raises:
            RuntimeError: If no providers in chain
        """
        if not self._chain:
            raise RuntimeError(f"No providers configured for {self.provider_type.value}")

        # Try each provider in priority order
        for provider_config in self._chain:
            name = provider_config.name

            # Check health if requested
            if check_health and provider_config.health_check:
                try:
                    is_healthy = await provider_config.health_check()
                    if not is_healthy:
                        logger.warning(f"⚠️ Provider {name} failed health check, trying next")
                        continue
                except Exception as e:
                    logger.error(f"❌ Health check error for {name}: {e}")
                    continue

            # Get or create instance
            try:
                provider = await self._get_or_create_instance(provider_config)
                logger.debug(f"✅ Selected provider: {name}")
                return provider
            except Exception as e:
                logger.error(f"❌ Failed to instantiate {name}: {e}")
                continue

        logger.error(f"❌ No available providers for {self.provider_type.value}")
        return None

    async def _get_or_create_instance(self, config: ProviderConfig) -> Any:
        """
        Get cached instance or create new one.

        Args:
            config: Provider configuration

        Returns:
            Provider instance
        """
        name = config.name

        # Return cached instance
        if name in self._instances:
            return self._instances[name]

        # Create new instance
        provider_class = self.registry.get(self.provider_type, name)

        # Instantiate
        provider = provider_class(**config.config)

        # Initialize if needed
        if hasattr(provider, 'initialize') and callable(provider.initialize):
            await provider.initialize()

        # Cache
        self._instances[name] = provider

        logger.info(f"✅ Instantiated provider: {name}")
        return provider

    async def execute(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with automatic fallback.

        Args:
            operation: Function to execute (receives provider as first arg)
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Operation result

        Raises:
            RuntimeError: If all providers fail

        Example:
            result = await selector.execute(
                lambda provider: provider.generate("Hello"),
            )
        """
        self._stats["total_requests"] += 1

        if not self._chain:
            raise RuntimeError(f"No providers configured for {self.provider_type.value}")

        last_error = None
        fallback_used = False

        # Try each provider in chain
        for provider_config in self._chain:
            name = provider_config.name

            # Track usage
            if name not in self._stats["provider_usage"]:
                self._stats["provider_usage"][name] = {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0
                }

            try:
                # Get provider instance
                provider = await self._get_or_create_instance(provider_config)

                # Try operation with retries
                for attempt in range(self.max_retries):
                    self._stats["provider_usage"][name]["attempts"] += 1

                    try:
                        # Execute operation
                        if asyncio.iscoroutinefunction(operation):
                            result = await operation(provider, *args, **kwargs)
                        else:
                            result = operation(provider, *args, **kwargs)

                        # Success!
                        self._stats["successful_requests"] += 1
                        self._stats["provider_usage"][name]["successes"] += 1

                        if fallback_used:
                            self._stats["fallback_used"] += 1
                            logger.info(f"✅ Fallback successful with {name}")

                        return result

                    except Exception as e:
                        last_error = e
                        logger.warning(
                            f"⚠️ Attempt {attempt + 1}/{self.max_retries} failed "
                            f"for provider {name}: {e}"
                        )

                        if attempt < self.max_retries - 1:
                            # Retry with exponential backoff
                            await asyncio.sleep(2 ** attempt)

                # All retries failed for this provider
                self._stats["provider_usage"][name]["failures"] += 1
                logger.error(f"❌ All retries exhausted for {name}")

                if not self.auto_retry:
                    break

                fallback_used = True

            except Exception as e:
                logger.error(f"❌ Critical error with provider {name}: {e}")
                last_error = e
                continue

        # All providers failed
        self._stats["failed_requests"] += 1
        error_msg = (
            f"All providers failed for {self.provider_type.value}. "
            f"Last error: {last_error}"
        )
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Statistics dictionary
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "provider_usage": {}
        }
        logger.info("Statistics reset")

    def get_chain(self) -> List[str]:
        """
        Get current provider chain order.

        Returns:
            List of provider names in priority order
        """
        return [p.name for p in self._chain]

    async def cleanup(self) -> None:
        """Cleanup all provider instances"""
        for name, provider in self._instances.items():
            if hasattr(provider, 'cleanup') and callable(provider.cleanup):
                try:
                    await provider.cleanup()
                    logger.debug(f"Cleaned up provider: {name}")
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}")

        self._instances.clear()
        logger.info("All provider instances cleaned up")


# ============================================================================
# Helper Functions
# ============================================================================

def create_selector(
    provider_type: Union[str, ProviderType],
    providers: List[Dict[str, Any]],
    **kwargs
) -> ProviderSelector:
    """
    Create selector with multiple providers.

    Args:
        provider_type: Type of provider ("llm", "stt", "tts" or ProviderType)
        providers: List of provider configs
        **kwargs: Additional selector options

    Returns:
        Configured selector

    Example:
        selector = create_selector(
            "llm",
            [
                {"name": "groq", "priority": 100, "config": {...}},
                {"name": "openai", "priority": 50, "config": {...}},
            ]
        )
    """
    # Convert string to enum
    if isinstance(provider_type, str):
        provider_type = ProviderType(provider_type.lower())

    selector = ProviderSelector(provider_type, **kwargs)

    # Add providers
    for provider_config in providers:
        selector.add_provider(**provider_config)

    return selector
