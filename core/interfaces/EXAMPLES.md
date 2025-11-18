# Interface Usage Examples

This document provides practical examples of how to use the interfaces defined in `src.core.interfaces`.

## Table of Contents
- [Why Interfaces?](#why-interfaces)
- [Provider Interfaces](#provider-interfaces)
- [Manager Interfaces](#manager-interfaces)
- [Service Interfaces](#service-interfaces)
- [Dependency Injection with Interfaces](#dependency-injection-with-interfaces)
- [Testing with Interfaces](#testing-with-interfaces)

---

## Why Interfaces?

Interfaces enable **Dependency Inversion** (the "D" in SOLID):
- **High-level modules** depend on abstractions (interfaces), not concrete implementations
- **Easy to swap** implementations without changing dependent code
- **Better testability** - create mocks/fakes that implement interfaces
- **Clear contracts** - interfaces document expected behavior

---

## Provider Interfaces

### Example 1: Using ILLMProvider

```python
from src.core.interfaces import ILLMProvider
from typing import Optional

class MyOrchestrator:
    """Orchestrator that depends on ILLMProvider interface"""

    def __init__(self, llm_provider: ILLMProvider):
        """
        Inject LLM provider via constructor (Dependency Injection)

        Args:
            llm_provider: Any provider implementing ILLMProvider interface
        """
        self.llm = llm_provider

    async def process_text(self, user_input: str) -> str:
        """
        Process user input using injected LLM provider

        Works with ANY ILLMProvider implementation:
        - GroqLLMProvider
        - OpenAIProvider
        - UltravoxProvider
        - MockLLMProvider (for tests)
        """
        # Get model info
        model_info = self.llm.get_model_info()
        print(f"Using model: {model_info['name']}")

        # Generate response
        response = await self.llm.generate(
            prompt=user_input,
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            max_tokens=500
        )

        return response

# Usage with different providers:

# Production: Use Groq
from src.services.external_llm.groq_provider import GroqLLMProvider
groq = GroqLLMProvider(api_key="...")
orchestrator = MyOrchestrator(groq)

# Development: Use mock
from tests.mocks import MockLLMProvider
mock = MockLLMProvider()
orchestrator = MyOrchestrator(mock)
```

### Example 2: Using ISTTProvider and ITTSProvider

```python
from src.core.interfaces import ISTTProvider, ITTSProvider

class ConversationPipeline:
    """Pipeline that depends on STT and TTS interfaces"""

    def __init__(
        self,
        stt_provider: ISTTProvider,
        tts_provider: ITTSProvider
    ):
        self.stt = stt_provider
        self.tts = tts_provider

    async def process_audio_conversation(
        self,
        audio_input: bytes,
        response_text: str
    ) -> dict:
        """
        Process audio conversation:
        1. Transcribe user audio (STT)
        2. Generate response audio (TTS)
        """
        # Transcribe user input
        transcription = await self.stt.transcribe(
            audio=audio_input,
            language="pt"
        )

        print(f"User said: {transcription}")

        # Synthesize response
        response_audio = await self.tts.synthesize(
            text=response_text,
            voice="pf_dora",
            language="pt-BR",
            speed=1.0
        )

        return {
            "transcription": transcription,
            "response_audio": response_audio,
            "audio_format": self.tts.get_audio_format()
        }
```

### Example 3: Using IConversationStore

```python
from src.core.interfaces import IConversationStore

class SessionHandler:
    """Handler that depends on IConversationStore interface"""

    def __init__(self, store: IConversationStore):
        self.store = store

    async def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str
    ) -> None:
        """Save a complete conversation turn"""
        # Save user message
        await self.store.save_message(
            session_id=session_id,
            role="user",
            content=user_message,
            metadata={"timestamp": "2025-01-01T10:00:00"}
        )

        # Save assistant response
        await self.store.save_message(
            session_id=session_id,
            role="assistant",
            content=assistant_response,
            metadata={"model": "gpt-4", "tokens": 150}
        )

    async def get_conversation_summary(
        self,
        session_id: str
    ) -> dict:
        """Get conversation summary"""
        # Get history
        history = await self.store.get_conversation_history(
            session_id=session_id,
            limit=10
        )

        # Get statistics
        stats = await self.store.get_statistics(session_id)

        return {
            "message_count": stats["message_count"],
            "recent_messages": history[-5:]
        }
```

---

## Manager Interfaces

### Example 4: Using ISessionManager

```python
from src.core.interfaces import ISessionManager

class UserSessionService:
    """Service that depends on ISessionManager interface"""

    def __init__(self, session_mgr: ISessionManager):
        self.sessions = session_mgr

    def start_new_conversation(
        self,
        user_id: str,
        device: str
    ) -> dict:
        """Start new conversation session"""
        # Create session with metadata
        session = self.sessions.create_session(
            user_id=user_id,
            metadata={
                "device": device,
                "started_at": "2025-01-01T10:00:00"
            }
        )

        return {
            "session_id": session.session_id,
            "user_id": session.user_id
        }

    def continue_conversation(
        self,
        session_id: str,
        user_input: str,
        bot_response: str
    ) -> bool:
        """Add interaction to existing session"""
        return self.sessions.add_interaction(
            session_id=session_id,
            user_message=user_input,
            assistant_response=bot_response,
            audio_duration_ms=1500,
            voice_used="pf_dora"
        )

    def get_context_for_llm(
        self,
        session_id: str
    ) -> list:
        """Get conversation context for LLM"""
        return self.sessions.get_session_context(
            session_id=session_id,
            max_messages=10
        )
```

### Example 5: Using IGPUManager

```python
from src.core.interfaces import IGPUManager

class ModelLoader:
    """Loader that depends on IGPUManager interface"""

    def __init__(self, gpu_mgr: IGPUManager):
        self.gpu = gpu_mgr

    async def load_model_safely(
        self,
        service_id: str,
        model_name: str,
        required_memory_mb: int
    ) -> dict:
        """Load model with GPU memory management"""
        # Check GPU availability
        if not self.gpu.is_available():
            return {
                "success": False,
                "error": "GPU not available"
            }

        # Get GPU info
        gpu_info = self.gpu.get_gpu_info()
        print(f"GPU: {gpu_info['name']}")

        # Check memory availability
        usage = self.gpu.get_memory_usage()
        if usage["free_mb"] < required_memory_mb:
            return {
                "success": False,
                "error": f"Insufficient memory. Required: {required_memory_mb}MB, Available: {usage['free_mb']}MB"
            }

        # Allocate memory
        allocated = await self.gpu.allocate_memory(
            service_id=service_id,
            required_mb=required_memory_mb
        )

        if not allocated:
            return {"success": False, "error": "Memory allocation failed"}

        # Load model (implementation here)
        # ...

        return {
            "success": True,
            "model": model_name,
            "memory_allocated_mb": required_memory_mb,
            "backend": self.gpu.get_recommended_backend(model_name)
        }
```

### Example 6: Using IMetricsCollector

```python
from src.core.interfaces import IMetricsCollector
import time

class RequestProcessor:
    """Processor that depends on IMetricsCollector interface"""

    def __init__(self, metrics: IMetricsCollector):
        self.metrics = metrics

    async def process_request(self, request_data: dict) -> dict:
        """Process request with metrics tracking"""
        start_time = time.time()

        try:
            # Increment request counter
            self.metrics.increment(
                "requests_total",
                value=1,
                tags={"endpoint": "/process", "method": "POST"}
            )

            # Update active requests gauge
            self.metrics.gauge("active_requests", 1.0)

            # Process request (implementation here)
            result = {"status": "success"}

            # Record processing time
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.timing(
                "request_duration",
                duration_ms,
                tags={"endpoint": "/process"}
            )

            # Record histogram
            self.metrics.histogram("response_size", len(str(result)))

            return result

        except Exception as e:
            # Track errors
            self.metrics.increment(
                "errors_total",
                value=1,
                tags={"endpoint": "/process", "error_type": type(e).__name__}
            )
            raise

        finally:
            # Update gauge
            self.metrics.gauge("active_requests", 0.0)
```

### Example 7: Using ICommunicationManager

```python
from src.core.interfaces import ICommunicationManager

class ServiceClient:
    """Client that depends on ICommunicationManager interface"""

    def __init__(self, comm: ICommunicationManager):
        self.comm = comm

    async def call_orchestrator(
        self,
        audio_data: bytes,
        session_id: str
    ) -> dict:
        """Call orchestrator service via communication manager"""
        # Check service health first
        healthy = await self.comm.health_check("orchestrator")
        if not healthy:
            raise Exception("Orchestrator service is not healthy")

        # Call service (protocol automatically selected)
        result = await self.comm.call_service(
            service_name="orchestrator",
            endpoint="/process",
            method="POST",
            data={
                "audio": audio_data,
                "session_id": session_id
            },
            timeout=30.0
        )

        return result
```

---

## Service Interfaces

### Example 8: Using IOrchestratorService

```python
from src.core.interfaces import IOrchestratorService

class ConversationAPI:
    """API that depends on IOrchestratorService interface"""

    def __init__(self, orchestrator: IOrchestratorService):
        self.orchestrator = orchestrator

    async def handle_voice_input(
        self,
        audio_data: bytes,
        session_id: str,
        user_id: str
    ) -> dict:
        """Handle voice input through orchestrator"""
        # Process conversation
        result = await self.orchestrator.process_conversation(
            audio_data=audio_data,
            session_id=session_id,
            user_id=user_id,
            voice="pf_dora",
            language="pt-BR"
        )

        return {
            "transcription": result["transcription"],
            "response": result["response_text"],
            "audio": result["response_audio"],
            "timing": result["timing"]
        }

    async def handle_barge_in(self, session_id: str) -> bool:
        """Handle user interruption (barge-in)"""
        return await self.orchestrator.cancel_processing(session_id)
```

---

## Dependency Injection with Interfaces

### Example 9: Full DI Setup

```python
from src.core.interfaces import (
    ILLMProvider,
    ISTTProvider,
    ITTSProvider,
    ISessionManager,
    IMetricsCollector
)

class ConversationService:
    """
    Service with full dependency injection

    All dependencies are interfaces - can be swapped easily
    """

    def __init__(
        self,
        llm: ILLMProvider,
        stt: ISTTProvider,
        tts: ITTSProvider,
        sessions: ISessionManager,
        metrics: IMetricsCollector
    ):
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.sessions = sessions
        self.metrics = metrics

    async def process_conversation_turn(
        self,
        audio_input: bytes,
        session_id: str
    ) -> dict:
        """Process complete conversation turn"""
        # Track request
        self.metrics.increment("conversations_total")

        # Transcribe
        text = await self.stt.transcribe(audio_input)

        # Get context
        context = self.sessions.get_session_context(session_id)

        # Generate response
        response = await self.llm.generate(
            prompt=text,
            system_prompt="You are helpful",
            temperature=0.7
        )

        # Synthesize
        audio = await self.tts.synthesize(response)

        # Save interaction
        self.sessions.add_interaction(
            session_id,
            text,
            response
        )

        return {
            "transcription": text,
            "response": response,
            "audio": audio
        }

# Setup with production providers:
from src.services.external_llm.groq_provider import GroqLLMProvider
from src.services.external_stt.groq_stt_provider import GroqSTTProvider
from src.services.external_tts.kokoro_provider import KokoroTTSProvider
from src.core.managers.session_manager import SessionManager
from src.core.metrics import MetricsCollector

service = ConversationService(
    llm=GroqLLMProvider(api_key="..."),
    stt=GroqSTTProvider(api_key="..."),
    tts=KokoroTTSProvider(api_key="..."),
    sessions=SessionManager(),
    metrics=MetricsCollector()
)

# Setup with mocks for testing:
from tests.mocks import (
    MockLLMProvider,
    MockSTTProvider,
    MockTTSProvider,
    MockSessionManager,
    MockMetricsCollector
)

test_service = ConversationService(
    llm=MockLLMProvider(),
    stt=MockSTTProvider(),
    tts=MockTTSProvider(),
    sessions=MockSessionManager(),
    metrics=MockMetricsCollector()
)
```

---

## Testing with Interfaces

### Example 10: Creating Mocks

```python
from src.core.interfaces import ILLMProvider
from typing import Dict, Any, Optional

class MockLLMProvider:
    """Mock LLM provider for testing"""

    def __init__(self, predefined_responses: dict = None):
        self.responses = predefined_responses or {}
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        self.call_count += 1
        # Return predefined response or echo
        return self.responses.get(prompt, f"Mock response to: {prompt}")

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "pt",
        **kwargs
    ) -> str:
        return "Mock transcription"

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "mock-model", "provider": "mock"}

    def is_available(self) -> bool:
        return True

    def supports_transcription(self) -> bool:
        return True

# Usage in tests:
import pytest

@pytest.mark.asyncio
async def test_orchestrator_with_mock():
    """Test orchestrator with mock LLM"""
    # Create mock with predefined responses
    mock_llm = MockLLMProvider({
        "Hello": "Hi there! How can I help you?"
    })

    # Create orchestrator with mock
    orchestrator = MyOrchestrator(llm_provider=mock_llm)

    # Test
    response = await orchestrator.process_text("Hello")

    # Assertions
    assert response == "Hi there! How can I help you?"
    assert mock_llm.call_count == 1
```

### Example 11: Runtime Type Checking

```python
from src.core.interfaces import ILLMProvider, ISTTProvider

def validate_provider(provider: object) -> str:
    """
    Runtime validation using isinstance()

    Protocol interfaces are @runtime_checkable
    """
    if isinstance(provider, ILLMProvider):
        return "Valid LLM Provider"
    elif isinstance(provider, ISTTProvider):
        return "Valid STT Provider"
    else:
        return "Unknown provider type"

# Usage:
from src.services.external_llm.groq_provider import GroqLLMProvider

groq = GroqLLMProvider(api_key="...")
result = validate_provider(groq)
print(result)  # "Valid LLM Provider"
```

---

## Best Practices

1. **Always depend on interfaces, not implementations**
   ```python
   # ✅ Good
   def __init__(self, llm: ILLMProvider):

   # ❌ Bad
   def __init__(self, llm: GroqLLMProvider):
   ```

2. **Use constructor injection (not property injection)**
   ```python
   # ✅ Good
   class Service:
       def __init__(self, provider: ILLMProvider):
           self.provider = provider

   # ❌ Bad
   class Service:
       provider: ILLMProvider  # Property without initialization
   ```

3. **Type hint with interfaces**
   ```python
   # ✅ Good
   async def process(llm: ILLMProvider, text: str) -> str:
       return await llm.generate(text)
   ```

4. **Create interface-compliant mocks for testing**
   ```python
   # ✅ Good - Mock implements interface
   class MockLLM:
       async def generate(self, prompt: str, **kwargs) -> str:
           return "mock"
       # ... other interface methods

   # ❌ Bad - Mock doesn't implement full interface
   class MockLLM:
       async def generate(self, prompt: str) -> str:
           return "mock"
       # Missing other methods!
   ```

5. **Use runtime checking when needed**
   ```python
   from src.core.interfaces import ILLMProvider

   if isinstance(provider, ILLMProvider):
       result = await provider.generate("Hello")
   ```

---

## Summary

Interfaces enable:
- ✅ **Dependency Inversion** - Depend on abstractions
- ✅ **Testability** - Easy mocking
- ✅ **Flexibility** - Swap implementations
- ✅ **Clear Contracts** - Documented behavior
- ✅ **Type Safety** - Better IDE support

Use interfaces everywhere for clean, maintainable, testable code!
