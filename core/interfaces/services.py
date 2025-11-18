"""
Service Interfaces (Dependency Inversion Principle)

This module defines interfaces for all services in the Ultravox pipeline.
Following the Dependency Inversion Principle, components depend on these
abstractions rather than concrete service implementations.

Interface Design:
- Use Protocol for structural typing (most services)
- Use ABC for critical service contracts
- All Protocol interfaces are runtime_checkable
- Interfaces mirror the BaseService contract
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable
from fastapi import APIRouter


# ============================================================================
# Base Service Interface (Core Contract)
# ============================================================================


class IBaseService(ABC):
    """
    Interface for Base Service (Abstract Base Class)

    All services must implement this contract for lifecycle management.
    Uses ABC for strict enforcement of core service lifecycle methods.
    """

    @abstractmethod
    def _setup_router(self) -> None:
        """
        Setup FastAPI routes for this service

        Implementation should register endpoints using:
        @self.router.get/post/put/delete(...)
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize service resources

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check

        Returns:
            Dictionary with health status and metrics
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Cleanup resources on service shutdown
        """
        pass

    @abstractmethod
    def get_router(self) -> APIRouter:
        """
        Get the FastAPI router for this service

        Returns:
            APIRouter instance with all service endpoints
        """
        pass


# ============================================================================
# Core Service Interfaces (Protocol-based)
# ============================================================================


@runtime_checkable
class IOrchestratorService(Protocol):
    """
    Interface for Orchestrator Service

    Coordinates the complete conversation flow:
    STT → LLM → TTS with session management and error handling.
    """

    async def process_conversation(
        self,
        audio_data: Optional[bytes] = None,
        text_input: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        voice: Optional[str] = None,
        language: str = "pt-BR",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a complete conversation turn

        Args:
            audio_data: Input audio (for voice input)
            text_input: Input text (for text input)
            session_id: Session identifier
            user_id: User identifier
            voice: Voice identifier for TTS
            language: Language code
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
            - transcription: User's transcribed input
            - response_text: LLM's text response
            - response_audio: TTS synthesized audio (base64)
            - session_id: Session identifier
            - timing: Processing time breakdown
        """
        ...

    async def cancel_processing(self, session_id: str) -> bool:
        """
        Cancel ongoing processing (barge-in support)

        Args:
            session_id: Session to cancel

        Returns:
            True if cancellation successful
        """
        ...

    async def get_conversation_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation processing status

        Args:
            session_id: Session identifier

        Returns:
            Status dictionary (processing, completed, error, etc.)
        """
        ...


@runtime_checkable
class IAPIGatewayService(Protocol):
    """
    Interface for API Gateway Service

    Unified REST API entry point for the system.
    Routes requests to appropriate services.
    """

    async def validate_request(
        self,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate incoming request

        Args:
            request_data: Request payload

        Returns:
            Validation result (valid, errors, warnings)
        """
        ...

    async def route_request(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route request to appropriate service

        Args:
            endpoint: Target endpoint
            data: Request data

        Returns:
            Service response
        """
        ...


@runtime_checkable
class IWebSocketService(Protocol):
    """
    Interface for WebSocket Service

    Real-time bidirectional communication for conversations.
    """

    async def handle_connection(
        self,
        websocket: Any,
        session_id: Optional[str] = None
    ) -> None:
        """
        Handle WebSocket connection

        Args:
            websocket: WebSocket connection object
            session_id: Session identifier (optional)
        """
        ...

    async def broadcast_message(
        self,
        message: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> None:
        """
        Broadcast message to connected clients

        Args:
            message: Message to broadcast
            session_id: Target session (optional, None = all)
        """
        ...

    def get_active_connections(self) -> int:
        """
        Get number of active WebSocket connections

        Returns:
            Number of active connections
        """
        ...


@runtime_checkable
class IWebRTCService(Protocol):
    """
    Interface for WebRTC Service

    Ultra-low latency audio streaming for real-time conversations.
    """

    async def create_peer_connection(
        self,
        session_id: str,
        offer: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create WebRTC peer connection

        Args:
            session_id: Session identifier
            offer: WebRTC offer SDP

        Returns:
            WebRTC answer SDP
        """
        ...

    async def handle_ice_candidate(
        self,
        session_id: str,
        candidate: Dict[str, Any]
    ) -> None:
        """
        Handle ICE candidate

        Args:
            session_id: Session identifier
            candidate: ICE candidate data
        """
        ...

    async def close_connection(self, session_id: str) -> None:
        """
        Close WebRTC connection

        Args:
            session_id: Session to close
        """
        ...


# ============================================================================
# AI Service Interfaces (External Services)
# ============================================================================


@runtime_checkable
class ILLMService(Protocol):
    """
    Interface for LLM Service (local or external)

    Exposes LLM capabilities as a service.
    """

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text response

        Args:
            prompt: User prompt
            system_prompt: System context
            conversation_history: Previous messages
            temperature: Sampling temperature
            max_tokens: Max response length
            **kwargs: Model-specific parameters

        Returns:
            Generated text
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get LLM model information

        Returns:
            Model details (name, provider, capabilities)
        """
        ...


@runtime_checkable
class ISTTService(Protocol):
    """
    Interface for Speech-to-Text Service

    Exposes STT capabilities as a service.
    """

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text

        Args:
            audio_data: Audio bytes
            language: Language code
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary with:
            - text: Transcribed text
            - confidence: Confidence score
            - language: Detected language
            - duration_ms: Processing time
        """
        ...


@runtime_checkable
class ITTSService(Protocol):
    """
    Interface for Text-to-Speech Service

    Exposes TTS capabilities as a service.
    """

    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "pt-BR",
        speed: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            voice: Voice identifier
            language: Language code
            speed: Speaking speed
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary with:
            - audio_data: Audio bytes (base64 encoded)
            - format: Audio format (mp3, wav, etc.)
            - duration_ms: Audio duration
            - voice: Voice used
        """
        ...

    def get_available_voices(
        self,
        language: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get available voices

        Args:
            language: Filter by language

        Returns:
            List of voice dictionaries
        """
        ...


# ============================================================================
# Data Service Interfaces
# ============================================================================


@runtime_checkable
class IDatabaseService(Protocol):
    """
    Interface for Database Service

    Exposes database operations as a service.
    """

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute database query

        Args:
            query: Query string
            params: Query parameters

        Returns:
            Query results
        """
        ...

    async def store_embedding(
        self,
        document_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store document embedding

        Args:
            document_id: Document identifier
            embedding: Vector embedding
            metadata: Document metadata

        Returns:
            True if stored successfully
        """
        ...

    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search similar documents by embedding

        Args:
            query_embedding: Query vector
            limit: Max results
            threshold: Similarity threshold

        Returns:
            List of similar documents with scores
        """
        ...


@runtime_checkable
class IUserService(Protocol):
    """
    Interface for User Service

    Manages user accounts and authentication.
    """

    async def create_user(
        self,
        user_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create new user

        Args:
            user_data: User information

        Returns:
            User ID if created successfully
        """
        ...

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID

        Args:
            user_id: User identifier

        Returns:
            User data if found
        """
        ...

    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate user

        Args:
            username: Username
            password: Password

        Returns:
            User data with auth token if successful
        """
        ...


@runtime_checkable
class IScenariosService(Protocol):
    """
    Interface for Scenarios Service

    Manages conversation scenarios and personas.
    """

    async def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """
        Get scenario configuration

        Args:
            scenario_id: Scenario identifier

        Returns:
            Scenario configuration
        """
        ...

    async def list_scenarios(self) -> List[Dict[str, Any]]:
        """
        List all available scenarios

        Returns:
            List of scenario summaries
        """
        ...

    async def create_scenario(
        self,
        scenario_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create new scenario

        Args:
            scenario_data: Scenario configuration

        Returns:
            Scenario ID if created successfully
        """
        ...


# ============================================================================
# Export all interfaces
# ============================================================================

__all__ = [
    # Base
    "IBaseService",
    # Core Services
    "IOrchestratorService",
    "IAPIGatewayService",
    "IWebSocketService",
    "IWebRTCService",
    # AI Services
    "ILLMService",
    "ISTTService",
    "ITTSService",
    # Data Services
    "IDatabaseService",
    "IUserService",
    "IScenariosService",
]
