"""
Communication Package - Modular service communication components.

Part of Phase 2 refactoring - split monolithic Communication Manager
into focused, single-responsibility components following SOLID principles.

## Components (SOLID - Single Responsibility Principle)

### Core Interfaces (Interface Segregation Principle)
- **IProtocolSelector**: Protocol selection interface
- **IServiceResolver**: URL resolution interface
- **IResilienceManager**: Resilience patterns interface
- **IMetricsCollector**: Metrics tracking interface
- **IServiceRegistry**: Service registration interface
- **IServiceDiscoveryClient**: Service discovery interface
- **ICommunicationManager**: High-level facade interface

### Implementations
- **ProtocolSelector**: Intelligent protocol selection (ZeroMQ → gRPC → HTTP)
- **ServiceResolver**: URL resolution (internal vs external)
- **ResilienceManager**: Circuit breaker + retry patterns
- **MetricsCollector**: Performance metrics tracking
- **ServiceRegistry**: In-process service registration
- **ServiceDiscoveryClient**: Service discovery via Service Manager
- **CommunicationFacade**: Simple facade orchestrating all components

## Usage

```python
from src.core.communication import CommunicationFacade

# Create facade (auto-wires all components)
comm = CommunicationFacade()

# Call service (automatic protocol selection, resilience, metrics)
result = await comm.call_service("llm", "/generate", json_data={"prompt": "Hello"})

# Or use individual components
from src.core.communication import ProtocolSelector, MetricsCollector

selector = ProtocolSelector()
protocol = selector.select_protocol("llm", data_size=1024, priority=Priority.NORMAL)

metrics = MetricsCollector()
metrics.record_success("llm", protocol, latency_ms=45.2)
stats = metrics.get_metrics("llm")
```

## Design Principles Applied

1. **Single Responsibility**: Each component has ONE clear responsibility
2. **Open/Closed**: Easy to extend without modifying existing code
3. **Liskov Substitution**: Implementations can be swapped via interfaces
4. **Interface Segregation**: Multiple small interfaces vs one large interface
5. **Dependency Inversion**: Components depend on abstractions, not concretions
"""

# Interfaces (for type checking and contracts)
from .interfaces import (
    IProtocolSelector,
    IServiceResolver,
    IResilienceManager,
    IMetricsCollector,
    IServiceRegistry,
    IServiceDiscoveryClient,
    ICommunicationManager,
    CommunicationProtocol,
    Priority,
)

# Implementations
from .protocol_selector import ProtocolSelector
from .service_resolver import ServiceResolver
from .resilience_manager import ResilienceManager
from .metrics_collector import MetricsCollector, get_metrics_collector
from .service_registry import ServiceRegistry, get_service_registry
from .service_discovery_client import ServiceDiscoveryClient, get_service_discovery_client
from .facade import CommunicationFacade

__all__ = [
    # Interfaces
    "IProtocolSelector",
    "IServiceResolver",
    "IResilienceManager",
    "IMetricsCollector",
    "IServiceRegistry",
    "IServiceDiscoveryClient",
    "ICommunicationManager",
    "CommunicationProtocol",
    "Priority",
    # Implementations
    "ProtocolSelector",
    "ServiceResolver",
    "ResilienceManager",
    "MetricsCollector",
    "ServiceRegistry",
    "ServiceDiscoveryClient",
    "CommunicationFacade",
    # Singletons
    "get_metrics_collector",
    "get_service_registry",
    "get_service_discovery_client",
]
