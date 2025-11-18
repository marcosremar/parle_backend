"""
Provider Interfaces (Dependency Inversion Principle)

This module defines interfaces for all AI and data providers in the system.
Following the Dependency Inversion Principle, high-level modules depend on
these abstractions, not on concrete implementations.

Interface Design:
- Use Protocol for structural typing (duck typing, runtime checking)
- Use ABC for contracts requiring explicit inheritance
- All interfaces are runtime_checkable for isinstance() support
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Protocol, runtime_checkable


# ============================================================================
# AI Provider Interfaces (Protocol-based for flexibility)
# ============================================================================


@runtime_checkable
class ILLMProvider(Protocol):
    """
    Interface for Language Model providers

    Providers implementing this interface can be used interchangeably
    for text generation tasks (Groq, OpenAI, Ultravox, LiteLLM, etc.)

    This is a Protocol (structural typing) to allow existing classes
    to be used without modification.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text response from prompt

        Args:
            prompt: User input prompt
            system_prompt: System context/instructions
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            Generated text response
        """
        ...

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "pt",
        **kwargs
    ) -> str:
        """
        Transcribe audio to text (if provider supports it)

        Args:
            audio_data: Audio bytes to transcribe
            language: Language code (e.g., "pt", "en")
            **kwargs: Provider-specific parameters

        Returns:
            Transcribed text

        Raises:
            NotImplementedError: If provider doesn't support transcription
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model

        Returns:
            Dictionary with model information (name, provider, capabilities, etc.)
        """
        ...

    def is_available(self) -> bool:
        """
        Check if the provider is available and configured

        Returns:
            True if provider is ready to use
        """
        ...

    def supports_transcription(self) -> bool:
        """
        Check if this provider supports audio transcription

        Returns:
            True if transcription is supported
        """
        ...


@runtime_checkable
class ISTTProvider(Protocol):
    """
    Interface for Speech-to-Text providers

    Providers: Groq Whisper, OpenAI Whisper, local Whisper, etc.
    """

    async def transcribe(
        self,
        audio: Union[bytes, str],
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio to text

        Args:
            audio: Audio data (bytes) or file path
            language: Language code (e.g., 'pt', 'en')
            **kwargs: Provider-specific parameters

        Returns:
            Transcribed text
        """
        ...

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages

        Returns:
            List of language codes
        """
        ...

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information

        Returns:
            Dictionary with provider details (name, model, capabilities)
        """
        ...

    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming transcription

        Returns:
            True if streaming is supported
        """
        ...


@runtime_checkable
class ITTSProvider(Protocol):
    """
    Interface for Text-to-Speech providers

    Providers: EdgeTTS, Azure TTS, Kokoro, ElevenLabs, etc.
    """

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "pt-BR",
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        Convert text to speech audio

        Args:
            text: Text to convert to speech
            voice: Voice identifier (provider-specific)
            language: Language code (e.g., "pt-BR", "en-US")
            speed: Speaking speed (1.0 = normal)
            **kwargs: Provider-specific parameters

        Returns:
            Audio bytes (format depends on provider)
        """
        ...

    def get_available_voices(
        self,
        language: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get list of available voices

        Args:
            language: Filter by language (optional)

        Returns:
            List of voice dictionaries with:
            - name: Voice identifier
            - language: Language code
            - gender: Voice gender
            - display_name: Human-readable name
        """
        ...

    def get_audio_format(self) -> str:
        """
        Get the audio format produced by this provider

        Returns:
            Audio format (e.g., "mp3", "wav", "ogg")
        """
        ...

    def is_available(self) -> bool:
        """
        Check if the provider is available and configured

        Returns:
            True if provider is ready to use
        """
        ...

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes

        Returns:
            List of language codes (e.g., ["pt-BR", "en-US"])
        """
        ...


# ============================================================================
# Data Provider Interfaces (ABC-based for stricter contracts)
# ============================================================================


class IDatabaseProvider(ABC):
    """
    Interface for Database providers

    Abstract base class for database operations (vector store, SQL, NoSQL)
    Uses ABC for stricter contract enforcement.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish database connection

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health

        Returns:
            Dictionary with health status and metrics
        """
        pass

    @abstractmethod
    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute database query

        Args:
            query: Query string (SQL, vector search, etc.)
            params: Query parameters

        Returns:
            List of result dictionaries
        """
        pass

    @abstractmethod
    async def insert(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Insert data into database

        Args:
            table: Table/collection name
            data: Data to insert

        Returns:
            ID of inserted record (if applicable)
        """
        pass

    @abstractmethod
    async def update(
        self,
        table: str,
        record_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Update database record

        Args:
            table: Table/collection name
            record_id: Record identifier
            data: Updated data

        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def delete(
        self,
        table: str,
        record_id: str
    ) -> bool:
        """
        Delete database record

        Args:
            table: Table/collection name
            record_id: Record identifier

        Returns:
            True if deletion successful
        """
        pass


class IConversationStore(ABC):
    """
    Interface for Conversation Storage providers

    Abstract base class for storing and retrieving conversation history.
    Implementations: SQLite, PostgreSQL, MongoDB, Redis, etc.
    """

    @abstractmethod
    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Save a conversation message

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata (audio_duration, voice_used, etc.)

        Returns:
            Message ID if saved successfully
        """
        pass

    @abstractmethod
    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            offset: Number of messages to skip

        Returns:
            List of message dictionaries
        """
        pass

    @abstractmethod
    async def delete_conversation(
        self,
        session_id: str
    ) -> bool:
        """
        Delete entire conversation

        Args:
            session_id: Session identifier

        Returns:
            True if deletion successful
        """
        pass

    @abstractmethod
    async def get_statistics(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get conversation statistics

        Args:
            session_id: Session identifier (optional, None for global stats)

        Returns:
            Dictionary with statistics (message_count, unique_sessions, etc.)
        """
        pass

    @abstractmethod
    async def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search messages by content

        Args:
            query: Search query string
            session_id: Session to search within (optional)
            limit: Maximum results to return

        Returns:
            List of matching message dictionaries
        """
        pass


# ============================================================================
# Export all interfaces
# ============================================================================

__all__ = [
    # AI Providers
    "ILLMProvider",
    "ISTTProvider",
    "ITTSProvider",
    # Data Providers
    "IDatabaseProvider",
    "IConversationStore",
]
