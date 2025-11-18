"""
Base class for Speech-to-Text (STT) providers

NOTE: This class follows Liskov Substitution Principle (LSP)
- All providers must implement ALL abstract methods
- Behavior must be consistent across implementations
- Single Responsibility: STT only (no LLM functionality)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List, AsyncIterator
import logging

logger = logging.getLogger(__name__)


class BaseSTTProvider(ABC):
    """
    Abstract base class for Speech-to-Text providers.
    All STT providers (Groq Whisper, OpenAI Whisper, etc.) must implement these methods.

    Follows SOLID principles:
    - Single Responsibility: Only audio transcription
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
    async def transcribe(
        self,
        audio: Union[bytes, str],
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio to text.

        This is the core STT functionality. All providers MUST implement this.

        Args:
            audio: Audio data (bytes) or file path (str)
            language: Language code (e.g., 'pt', 'en') - None for auto-detection
            **kwargs: Provider-specific parameters (e.g., model, prompt, temperature)

        Returns:
            Transcribed text

        Raises:
            Exception: If transcription fails
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information.

        Returns:
            Dictionary with provider details:
            - name: Provider name
            - model: Model name/version
            - max_audio_size: Maximum audio file size (bytes)
            - supported_formats: List of audio formats
            - supports_streaming: Whether streaming is supported
        """
        pass

    # ========================================================================
    # Optional Methods (have default implementations)
    # ========================================================================

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        language: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Transcribe audio stream with real-time output (optional).

        Default implementation raises NotImplementedError.
        Override for providers that support streaming.

        Args:
            audio_stream: Async iterator of audio chunks
            language: Language code
            **kwargs: Provider-specific parameters

        Yields:
            Transcribed text chunks as they are processed

        Raises:
            NotImplementedError: If provider doesn't support streaming
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming transcription"
        )

    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming transcription.

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

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported audio formats.

        Returns:
            List of formats (e.g., ["wav", "mp3", "ogg", "flac"])
        """
        return ["wav", "mp3", "ogg", "flac", "opus", "webm"]

    def get_max_audio_size(self) -> int:
        """
        Get maximum audio file size in bytes.

        Returns:
            Maximum size in bytes (default: 25MB)
        """
        return 25 * 1024 * 1024  # 25MB

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information (alias for get_provider_info).

        Returns:
            Dictionary with model/provider details
        """
        return self.get_provider_info()

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
        return f"{self.__class__.__name__}(config={self.config})"