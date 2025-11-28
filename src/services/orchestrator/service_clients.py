"""
Service Clients - Backward Compatibility Wrapper

This file maintains backward compatibility by re-exporting all clients
from the new modular structure. The actual implementations are now in
src/services/orchestrator/clients/.
"""

from __future__ import annotations

# Re-export everything from the new modular structure
from .clients import (
    # Base classes
    BaseServiceClient,
    ServiceClientError,
    Priority,
    create_service_clients,
    # AI Clients
    LLMClient,
    TTSClient,
    STTClient,
    ExternalUltravoxClient,
    ExternalLLMClient,
    ExternalSTTClient,
    ExternalTTSClient,
    # Data Clients
    SessionClient,
    ScenariosClient,
    ConversationStoreClient,
    ConversationHistoryClient,
    UserClient,
    DatabaseClient,
    FileStorageClient,
    # Communication Clients
    WebSocketClient,
    RestPollingClient,
    WebRTCClient,
    WebRTCSignalingClient,
    NeuralCodecClient,
    ViberGatewayClient,
    # ITS Clients
    StudentModelClient,
    PedagogicalPolicyClient,
    DiagnosticModuleClient,
    LearningPathClient,
    # Gateway Clients
    APIGatewayClient,
)

__all__ = [
    "BaseServiceClient",
    "ServiceClientError",
    "Priority",
    "create_service_clients",
    "LLMClient",
    "TTSClient",
    "STTClient",
    "ExternalUltravoxClient",
    "ExternalLLMClient",
    "ExternalSTTClient",
    "ExternalTTSClient",
    "SessionClient",
    "ScenariosClient",
    "ConversationStoreClient",
    "ConversationHistoryClient",
    "UserClient",
    "DatabaseClient",
    "FileStorageClient",
    "WebSocketClient",
    "RestPollingClient",
    "WebRTCClient",
    "WebRTCSignalingClient",
    "NeuralCodecClient",
    "ViberGatewayClient",
    "StudentModelClient",
    "PedagogicalPolicyClient",
    "DiagnosticModuleClient",
    "LearningPathClient",
    "APIGatewayClient",
]
