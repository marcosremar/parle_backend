"""
Service Clients Module

HTTP clients for all downstream services, organized by category.
"""

from .base import BaseServiceClient, ServiceClientError, Priority
from .factory import create_service_clients

# Import all client classes for backward compatibility
from .ai_clients import (
    LLMClient,
    TTSClient,
    STTClient,
    ExternalUltravoxClient,
    ExternalLLMClient,
    ExternalSTTClient,
    ExternalTTSClient,
)
from .data_clients import (
    SessionClient,
    ScenariosClient,
    ConversationStoreClient,
    ConversationHistoryClient,
    UserClient,
    DatabaseClient,
    FileStorageClient,
)
from .communication_clients import (
    WebSocketClient,
    RestPollingClient,
    WebRTCClient,
    WebRTCSignalingClient,
    NeuralCodecClient,
    ViberGatewayClient,
)
from .its_clients import (
    StudentModelClient,
    PedagogicalPolicyClient,
    DiagnosticModuleClient,
    LearningPathClient,
)
from .gateway_clients import APIGatewayClient

__all__ = [
    "BaseServiceClient",
    "ServiceClientError",
    "Priority",
    "create_service_clients",
    # AI Clients
    "LLMClient",
    "TTSClient",
    "STTClient",
    "ExternalUltravoxClient",
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
