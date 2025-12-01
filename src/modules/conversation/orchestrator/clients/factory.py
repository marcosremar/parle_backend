"""
Service Clients Factory

Factory function to create all service clients.
"""

from __future__ import annotations

from .ai_clients import (
    ExternalLLMClient,
    ExternalSTTClient,
    ExternalTTSClient,
    SecondaryLLMClient,
)
from .base import BaseServiceClient
from .communication_clients import (
    NeuralCodecClient,
    RestPollingClient,
    ViberGatewayClient,
    WebRTCClient,
    WebRTCSignalingClient,
    WebSocketClient,
)
from .data_clients import (
    ConversationHistoryClient,
    ConversationStoreClient,
    DatabaseClient,
    FileStorageClient,
    ScenariosClient,
    SessionClient,
    UserClient,
)
from .gateway_clients import APIGatewayClient
from .its_clients import (
    DiagnosticModuleClient,
    LearningPathClient,
    PedagogicalPolicyClient,
    StudentModelClient,
)


def create_service_clients(config: dict[str, str] | None = None) -> dict[str, BaseServiceClient]:
    """
    Create all service clients.

    Module services (in-process, execution_mode: internal):
    - llm, stt, tts (renamed from external_*)
    - session, scenarios, orchestrator
    - file_storage, database, communication

    HTTP services (separate process, execution_mode: external or remote):
    - user, conversation_store
    - api_gateway, websocket, webrtc

    Args:
        config: Optional dictionary (preserved for compatibility, not used)

    Returns:
        Dictionary of service name -> client instance
    """
    clients: dict[str, BaseServiceClient] = {
        # AI Services (External/Remote) - MODULE services (in-process, lightweight API wrappers)
        "external_ultravox": SecondaryLLMClient(),  # Service name kept for backward compatibility
        "llm": ExternalLLMClient(),  # Renamed from external_llm
        "stt": ExternalSTTClient(),  # Renamed from external_stt
        "tts": ExternalTTSClient(),  # Renamed from external_tts
        # Data & State Services
        "session": SessionClient(),  # MODULE service (in-process CRUD)
        "scenarios": ScenariosClient(),  # MODULE service (in-process CRUD)
        "conversation_store": ConversationStoreClient(),  # HTTP service
        "conversation_history": ConversationHistoryClient(),  # HTTP service
        "user": UserClient(),  # HTTP service
        "database": DatabaseClient(),  # MODULE service (in-process)
        "file_storage": FileStorageClient(),  # MODULE service (in-process)
        # Communication Services
        "websocket": WebSocketClient(),  # HTTP service - Real-time communication
        "rest_polling": RestPollingClient(),  # HTTP service - REST polling
        "webrtc": WebRTCClient(),  # HTTP service - WebRTC communication
        "webrtc_signaling": WebRTCSignalingClient(),  # HTTP service - WebRTC signaling
        "neural_codec": NeuralCodecClient(),  # HTTP service - Neural audio codec
        # Gateway Services
        "api_gateway": APIGatewayClient(),  # HTTP service - API Gateway (reverse communication)
        "viber_gateway": ViberGatewayClient(),  # HTTP service - Viber integration
        # Intelligent Tutoring System Services
        "student_model": StudentModelClient(),  # MODULE service - Student knowledge tracking
        "pedagogical_policy": PedagogicalPolicyClient(),  # MODULE service - Pedagogical decisions
        "speech_grader": DiagnosticModuleClient(),  # MODULE service - Error analysis
        "learning_path": LearningPathClient(),  # MODULE service - Learning path navigation
    }

    return clients
