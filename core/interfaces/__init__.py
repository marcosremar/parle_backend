"""
Core Interfaces Module (Dependency Inversion Principle)

This module provides all interfaces/abstractions for the Ultravox Pipeline system.
Following the Dependency Inversion Principle (SOLID), high-level modules depend
on these abstractions rather than concrete implementations.

Organization:
- providers.py: AI and data provider interfaces (LLM, STT, TTS, Database, etc.)
- managers.py: Infrastructure manager interfaces (GPU, Metrics, Session, Profile, etc.)
- services.py: Service interfaces (Orchestrator, APIGateway, WebSocket, etc.)

Usage:
    from src.core.interfaces import ILLMProvider, ISTTProvider, IGPUManager

    # Type hints with interfaces
    def process_text(llm: ILLMProvider, text: str) -> str:
        return llm.generate(text)

    # Runtime checking (Protocol interfaces)
    if isinstance(provider, ILLMProvider):
        result = provider.generate("Hello")

Design Patterns:
- Protocol: For structural typing (duck typing with runtime checking)
- ABC (Abstract Base Class): For strict contract enforcement
- All Protocol interfaces are @runtime_checkable for isinstance() support

Benefits:
1. Dependency Inversion: High-level code depends on abstractions
2. Testability: Easy to create mocks/fakes implementing interfaces
3. Flexibility: Swap implementations without changing dependent code
4. Documentation: Interfaces serve as API contracts
5. Type Safety: Better IDE support and static analysis
"""

# ============================================================================
# Provider Interfaces
# ============================================================================

from .providers import (
    # AI Providers
    ILLMProvider,
    ISTTProvider,
    ITTSProvider,
    # Data Providers
    IDatabaseProvider,
    IConversationStore,
)

# ============================================================================
# Manager Interfaces
# ============================================================================

from .managers import (
    # Session & Profile
    ISessionManager,
    IProfileManager,
    # Infrastructure
    IGPUManager,
    IMetricsCollector,
    ICommunicationManager,
    IBenchmarkManager,
)

# ============================================================================
# Service Interfaces
# ============================================================================

from .services import (
    # Base
    IBaseService,
    # Core Services
    IOrchestratorService,
    IAPIGatewayService,
    IWebSocketService,
    IWebRTCService,
    # AI Services
    ILLMService,
    ISTTService,
    ITTSService,
    # Data Services
    IDatabaseService,
    IUserService,
    IScenariosService,
)

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Provider Interfaces
    "ILLMProvider",
    "ISTTProvider",
    "ITTSProvider",
    "IDatabaseProvider",
    "IConversationStore",
    # Manager Interfaces
    "ISessionManager",
    "IProfileManager",
    "IGPUManager",
    "IMetricsCollector",
    "ICommunicationManager",
    "IBenchmarkManager",
    # Service Interfaces
    "IBaseService",
    "IOrchestratorService",
    "IAPIGatewayService",
    "IWebSocketService",
    "IWebRTCService",
    "ILLMService",
    "ISTTService",
    "ITTSService",
    "IDatabaseService",
    "IUserService",
    "IScenariosService",
]

# ============================================================================
# Version Information
# ============================================================================

__version__ = "1.0.0"
__author__ = "Ultravox Pipeline Team"
__description__ = "Core interfaces for dependency inversion (SOLID principles)"

# ============================================================================
# Interface Categories (for documentation)
# ============================================================================

PROVIDER_INTERFACES = [
    "ILLMProvider",
    "ISTTProvider",
    "ITTSProvider",
    "IDatabaseProvider",
    "IConversationStore",
]

MANAGER_INTERFACES = [
    "ISessionManager",
    "IProfileManager",
    "IGPUManager",
    "IMetricsCollector",
    "ICommunicationManager",
    "IBenchmarkManager",
]

SERVICE_INTERFACES = [
    "IBaseService",
    "IOrchestratorService",
    "IAPIGatewayService",
    "IWebSocketService",
    "IWebRTCService",
    "ILLMService",
    "ISTTService",
    "ITTSService",
    "IDatabaseService",
    "IUserService",
    "IScenariosService",
]

def get_interface_info() -> dict:
    """
    Get information about all available interfaces

    Returns:
        Dictionary with interface categories and counts
    """
    return {
        "total_interfaces": len(__all__),
        "providers": len(PROVIDER_INTERFACES),
        "managers": len(MANAGER_INTERFACES),
        "services": len(SERVICE_INTERFACES),
        "version": __version__,
        "categories": {
            "providers": PROVIDER_INTERFACES,
            "managers": MANAGER_INTERFACES,
            "services": SERVICE_INTERFACES,
        }
    }
