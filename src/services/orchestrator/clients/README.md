# Service Clients

This directory contains HTTP clients for all downstream services, organized by category.

## Structure

```
clients/
├── __init__.py              # Re-exports all clients
├── base.py                  # BaseServiceClient with common functionality
├── factory.py               # create_service_clients() factory function
├── ai_clients.py            # LLM, TTS, STT clients
├── data_clients.py          # Session, Scenarios, ConversationStore, etc.
├── communication_clients.py # WebSocket, WebRTC, REST polling
├── its_clients.py           # StudentModel, PedagogicalPolicy, DiagnosticModule
└── gateway_clients.py       # APIGateway client
```

## Base Client

### BaseServiceClient

**File:** `base.py`

Base class with common functionality for all service clients.

**Features:**
- Retry logic with exponential backoff
- Circuit breaker support
- Health checks
- Timeout configuration
- Monolith mode support (direct module calls)

**Common Methods:**
- `_get(path, timeout)` - GET request with retry
- `_post(path, data, json_data, timeout)` - POST request with retry
- `_put(path, json_data, timeout)` - PUT request with retry
- `health_check()` - Check service health

## Client Categories

### AI Clients

**File:** `ai_clients.py`

Clients for AI services (LLM, TTS, STT).

**Clients:**
- `LLMClient` - Primary LLM service
- `TTSClient` - Text-to-Speech service
- `STTClient` - Speech-to-Text service
- `ExternalUltravoxClient` - External Ultravox (Groq STT + LLM)
- `ExternalLLMClient` - External LLM (Groq, OpenAI)
- `ExternalSTTClient` - External STT (Groq Whisper)
- `ExternalTTSClient` - External TTS (HuggingFace)

**Usage:**
```python
from src.services.orchestrator.clients import LLMClient, TTSClient

llm_client = LLMClient()
await llm_client.initialize(session)

result = await llm_client.process_audio(
    audio_data=audio_bytes,
    sample_rate=16000,
    system_prompt="You are helpful"
)

tts_client = TTSClient()
await tts_client.initialize(session)

audio = await tts_client.synthesize(
    text="Hello",
    voice_id="Rachel"
)
```

### Data Clients

**File:** `data_clients.py`

Clients for data and state services.

**Clients:**
- `SessionClient` - Session management
- `ScenariosClient` - Scenario configuration
- `ConversationStoreClient` - Conversation storage
- `ConversationHistoryClient` - Conversation history
- `UserClient` - User management
- `DatabaseClient` - Database operations
- `FileStorageClient` - File storage

**Usage:**
```python
from src.services.orchestrator.clients import SessionClient, ScenariosClient

session_client = SessionClient()
await session_client.initialize(session)

session_data = await session_client.get_session("session_123")
new_session = await session_client.create_session(
    conversation_id="conv_123",
    scenario_id="scenario_1"
)

scenarios_client = ScenariosClient()
await scenarios_client.initialize(session)

scenario = await scenarios_client.get_scenario("scenario_1")
```

### Communication Clients

**File:** `communication_clients.py`

Clients for real-time communication services.

**Clients:**
- `WebSocketClient` - WebSocket communication
- `RestPollingClient` - REST polling
- `WebRTCClient` - WebRTC communication
- `WebRTCSignalingClient` - WebRTC signaling
- `NeuralCodecClient` - Neural audio codec
- `ViberGatewayClient` - Viber integration

**Usage:**
```python
from src.services.orchestrator.clients import WebSocketClient

ws_client = WebSocketClient()
await ws_client.initialize(session)

await ws_client.send_notification(
    user_id="user_123",
    message={"type": "notification", "text": "Hello"}
)
```

### ITS Clients

**File:** `its_clients.py`

Clients for Intelligent Tutoring System services.

**Clients:**
- `StudentModelClient` - Student knowledge tracking
- `PedagogicalPolicyClient` - Pedagogical decisions
- `DiagnosticModuleClient` - Error analysis
- `LearningPathClient` - Learning path navigation

**Usage:**
```python
from src.services.orchestrator.clients import StudentModelClient

student_client = StudentModelClient()
await student_client.initialize(session)

await student_client.assess(
    user_id="user_123",
    skill_id="skill_1",
    correct=True,
    user_text="Hello",
    ai_text="Hi there!"
)

progress = await student_client.get_cefr_progress("user_123")
```

### Gateway Clients

**File:** `gateway_clients.py`

Clients for gateway services.

**Clients:**
- `APIGatewayClient` - API Gateway (reverse communication)

**Usage:**
```python
from src.services.orchestrator.clients import APIGatewayClient

gateway_client = APIGatewayClient()
await gateway_client.initialize(session)

await gateway_client.register_route(
    route="/api/custom",
    service="orchestrator",
    endpoint="/custom"
)
```

## Factory Function

### create_service_clients()

**File:** `factory.py`

Creates all service clients in one call.

**Usage:**
```python
from src.services.orchestrator.clients import create_service_clients

clients = create_service_clients()

# Clients dictionary contains all clients:
# - clients["llm"] → ExternalLLMClient
# - clients["tts"] → ExternalTTSClient
# - clients["stt"] → ExternalSTTClient
# - clients["session"] → SessionClient
# - clients["scenarios"] → ScenariosClient
# - ... and many more

# Initialize all clients with shared session
for name, client in clients.items():
    await client.initialize(http_session)
```

**Returns:**
Dictionary mapping service names to client instances.

## Adding New Clients

To add a new client:

1. Create the client class in the appropriate category file
2. Inherit from `BaseServiceClient`
3. Implement service-specific methods
4. Add to `factory.py` `create_service_clients()`
5. Export from `__init__.py`

**Example:**
```python
# clients/data_clients.py
from .base import BaseServiceClient

class MyNewClient(BaseServiceClient):
    """My new service client."""
    
    def __init__(self) -> None:
        super().__init__("my_service")
    
    async def do_something(self, param: str) -> Dict[str, Any]:
        """Do something with the service."""
        return await self._get(f"/api/something/{param}")
```

Then add to `factory.py`:
```python
def create_service_clients(...):
    clients = {
        # ... existing clients
        "my_service": MyNewClient(),
    }
    return clients
```

## Backward Compatibility

The old `service_clients.py` file is maintained as a backward compatibility wrapper that re-exports all clients from the new modular structure.

**Old import (still works):**
```python
from src.services.orchestrator.service_clients import LLMClient, create_service_clients
```

**New import (recommended):**
```python
from src.services.orchestrator.clients import LLMClient, create_service_clients
```

## Testing

Clients are tested indirectly through integration tests. For unit testing, use mocks:

```python
from unittest.mock import AsyncMock

mock_clients = {
    "llm": AsyncMock(),
    "tts": AsyncMock(),
    "session": AsyncMock(),
    # ...
}
```

## Design Principles

1. **Single Responsibility** - Each client handles one service
2. **Consistent Interface** - All clients inherit from BaseServiceClient
3. **Retry Logic** - Automatic retry with exponential backoff
4. **Circuit Breaker** - Automatic failover support
5. **Type Safety** - Full type hints
6. **Graceful Degradation** - Handle failures without crashing
