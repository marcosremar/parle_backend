"""
Base class for TTS providers
All TTS providers must implement this interface

NOTE: This class follows Liskov Substitution Principle (LSP)
- All providers must implement ALL abstract methods
- Behavior must be consistent across implementations
- Single Responsibility: TTS only (text-to-speech synthesis)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncIterator
import logging

logger = logging.getLogger(__name__)


class BaseTTSProvider(ABC):
    """
    Abstract base class for Text-to-Speech providers.
    All TTS providers (EdgeTTS, Azure, Kokoro, etc.) must implement these methods.

    Follows SOLID principles:
    - Single Responsibility: Only text-to-speech synthesis
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
                     Common keys: voice, language, speed, pitch
        """
        self.config = kwargs
        self.default_voice = kwargs.get("voice")
        self.default_language = kwargs.get("language", "pt-BR")
        self.default_speed = kwargs.get("speed", 1.0)
        logger.info(f"Initializing {self.__class__.__name__} provider")

    # ========================================================================
    # Abstract Methods (MUST be implemented)
    # ========================================================================

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        Convert text to speech audio.

        This is the core TTS functionality. All providers MUST implement this.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (provider-specific) - uses default if None
            language: Language code (e.g., "pt-BR", "en-US") - uses default if None
            speed: Speaking speed (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
            **kwargs: Provider-specific parameters (pitch, volume, emotion, etc.)

        Returns:
            Audio bytes (format specified by get_audio_format())

        Raises:
            Exception: If synthesis fails
        """
        pass

    @abstractmethod
    def get_audio_format(self) -> str:
        """
        Get the audio format produced by this provider.

        Returns:
            Audio format (e.g., "mp3", "wav", "ogg", "opus")
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the TTS provider.

        Returns:
            Dictionary with provider information:
            - name: Provider name
            - audio_format: Default audio format
            - default_voice: Default voice ID
            - default_language: Default language
            - supports_streaming: Whether streaming is supported
            - max_text_length: Maximum text length (characters)
        """
        pass

    # ========================================================================
    # Optional Methods (have default implementations)
    # ========================================================================

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text with streaming audio output (optional).

        Default implementation falls back to non-streaming synthesize().
        Override for providers that support streaming.

        Args:
            text: Text to synthesize
            voice: Voice identifier
            language: Language code
            speed: Speaking speed
            **kwargs: Provider-specific parameters

        Yields:
            Audio chunks as they are synthesized

        Raises:
            NotImplementedError: If provider doesn't support streaming
        """
        # Default: fall back to non-streaming
        audio = await self.synthesize(text, voice, language, speed, **kwargs)
        yield audio

    async def get_available_voices(
        self,
        language: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get list of available voices (async).

        Args:
            language: Filter by language (optional)

        Returns:
            List of voice dictionaries with:
            - id: Voice identifier (required)
            - name: Voice identifier (alias for id)
            - language: Language code (required)
            - gender: Voice gender (male/female/neutral)
            - display_name: Human-readable name
            - locale: Full locale (e.g., pt-BR)
            - quality: Voice quality (standard/premium/neural)
        """
        return self.get_available_voices_sync(language)

    def get_available_voices_sync(
        self,
        language: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get list of available voices (sync).

        Override this OR get_available_voices() (async).
        Default returns empty list.

        Args:
            language: Filter by language (optional)

        Returns:
            List of voice dictionaries
        """
        return []

    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming synthesis.

        Returns:
            True if streaming is supported
        """
        return False

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List of language codes (BCP 47 format, e.g., ["pt-BR", "en-US"])
        """
        return [
            "pt-BR", "pt-PT",
            "en-US", "en-GB", "en-AU",
            "es-ES", "es-MX",
            "fr-FR", "fr-CA",
            "de-DE", "it-IT",
            "ja-JP", "ko-KR", "zh-CN"
        ]

    def get_max_text_length(self) -> int:
        """
        Get maximum text length in characters.

        Returns:
            Maximum characters (default: 5000)
        """
        return 5000

    def get_supported_sample_rates(self) -> List[int]:
        """
        Get list of supported audio sample rates.

        Returns:
            List of sample rates in Hz (e.g., [8000, 16000, 24000, 48000])
        """
        return [8000, 16000, 24000, 48000]

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
        return f"{self.__class__.__name__}(voice={self.default_voice}, language={self.default_language})"