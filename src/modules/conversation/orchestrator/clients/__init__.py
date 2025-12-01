"""
Service Clients Module

HTTP clients for all downstream services, organized by category.
"""

# Import all client classes for backward compatibility
from .ai_clients import (
    ExternalLLMClient,
    ExternalSTTClient,
    ExternalTTSClient,
    ExternalUltravoxClient,  # Backward compatibility alias
    LLMClient,
    SecondaryLLMClient,
    STTClient,
    TTSClient,
)
from .base import BaseServiceClient, Priority, ServiceClientError
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
from .factory import create_service_clients
from .gateway_clients import APIGatewayClient
from .its_clients import (
    DiagnosticModuleClient,
    LearningPathClient,
    PedagogicalPolicyClient,
    StudentModelClient,
)

__all__ = [
    "BaseServiceClient",
    "ServiceClientError",
    "Priority",
    "create_service_clients",
    # AI Clients
    "LLMClient",
    "TTSClient",
    "STTClient",
    "SecondaryLLMClient",
    "ExternalUltravoxClient",  # Backward compatibility alias
    "ExternalLLMClient",
    "ExternalSTTClient",
    "ExternalTTSClient",
    # Data Clients
    "SessionClient",
    "ScenariosClient",
    "ConversationStoreClient",
    "ConversationHistoryClient",
    "UserClient",
    "DatabaseClient",
    "FileStorageClient",
    # Communication Clients
    "WebSocketClient",
    "RestPollingClient",
    "WebRTCClient",
    "WebRTCSignalingClient",
    "NeuralCodecClient",
    "ViberGatewayClient",
    # ITS Clients
    "StudentModelClient",
    "PedagogicalPolicyClient",
    "DiagnosticModuleClient",
    "LearningPathClient",
    # Gateway Clients
    "APIGatewayClient",
]
