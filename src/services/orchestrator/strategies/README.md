# Strategy Pattern for LLM and TTS

This directory implements the Strategy Pattern to eliminate if/else chains for in-process vs HTTP processing modes.

## Overview

The Strategy Pattern allows the orchestrator to switch between different processing strategies (in-process vs HTTP) without complex conditional logic.

## Architecture

```
LLMStrategy (abstract)
    ├── InProcessLLMStrategy    # Direct module calls (ultra-low latency)
    └── HTTPLLMStrategy          # HTTP calls with failover

TTSStrategy (abstract)
    ├── InProcessTTSStrategy     # Direct module calls (ultra-low latency)
    └── HTTPTTSStrategy          # HTTP calls with fallback

Factories
    ├── LLMStrategyFactory       # Creates appropriate LLM strategy
    └── TTSStrategyFactory       # Creates appropriate TTS strategy
```

## LLM Strategies

### InProcessLLMStrategy

**File:** `llm_strategy.py`

Uses direct module calls for ultra-low latency processing.

**When to use:**
- `in_process_mode=True`
- `llm_instance` is available
- GPU is available
- Ultra-low latency is required

**Features:**
- Direct numpy array processing
- No HTTP overhead
- Automatic fallback to HTTP on failure

**Usage:**
```python
from src.services.orchestrator.strategies import InProcessLLMStrategy

strategy = InProcessLLMStrategy(llm_instance, stats_tracker)

text_response, llm_used, llm_result = await strategy.process_audio(
    audio_data=audio_bytes,
    sample_rate=16000,
    system_prompt="You are a helpful assistant",
    conversation_history=[],
    conversation_id="conv_123",
    force_external_llm=False
)
```

### HTTPLLMStrategy

**File:** `llm_strategy.py`

Uses HTTP calls with automatic failover between primary and fallback LLMs.

**When to use:**
- `in_process_mode=False` OR
- `llm_instance` is not available OR
- Fallback from in-process strategy

**Features:**
- Automatic failover (primary → fallback)
- Circuit breaker support
- Stats tracking

**Usage:**
```python
from src.services.orchestrator.strategies import HTTPLLMStrategy

strategy = HTTPLLMStrategy(fallback_manager, stats_tracker)

text_response, llm_used, llm_result = await strategy.process_audio(
    audio_data=audio_bytes,
    sample_rate=16000,
    system_prompt="You are a helpful assistant",
    conversation_history=[],
    conversation_id="conv_123",
    force_external_llm=False
)
```

## TTS Strategies

### InProcessTTSStrategy

**File:** `tts_strategy.py`

Uses direct module calls for ultra-low latency TTS.

**When to use:**
- `in_process_mode=True`
- `tts_instance` is available
- Ultra-low latency is required

**Features:**
- Direct synthesis
- No HTTP overhead
- Automatic fallback to HTTP on failure

**Usage:**
```python
from src.services.orchestrator.strategies import InProcessTTSStrategy

strategy = InProcessTTSStrategy(tts_instance)

audio_bytes = await strategy.synthesize(
    text="Hello, how are you?",
    voice_id="Rachel"
)
```

### HTTPTTSStrategy

**File:** `tts_strategy.py`

Uses HTTP calls with fallback to external TTS.

**When to use:**
- `in_process_mode=False` OR
- `tts_instance` is not available OR
- Fallback from in-process strategy

**Features:**
- Tries local TTS first
- Falls back to external TTS (HuggingFace) if local fails
- Returns None if both fail

**Usage:**
```python
from src.services.orchestrator.strategies import HTTPTTSStrategy

strategy = HTTPTTSStrategy(tts_client)

audio_bytes = await strategy.synthesize(
    text="Hello, how are you?",
    voice_id="Rachel"
)
```

## Factories

### LLMStrategyFactory

**File:** `llm_strategy.py`

Creates the appropriate LLM strategy based on configuration.

**Usage:**
```python
from src.services.orchestrator.strategies import LLMStrategyFactory

strategy = LLMStrategyFactory.create_strategy(
    in_process_mode=True,
    llm_instance=llm_instance,
    fallback_manager=fallback_manager,
    stats_tracker=stats_tracker
)
```

**Logic:**
- If `in_process_mode=True` AND `llm_instance` exists → `InProcessLLMStrategy`
- Otherwise → `HTTPLLMStrategy`

### TTSStrategyFactory

**File:** `tts_strategy.py`

Creates the appropriate TTS strategy based on configuration.

**Usage:**
```python
from src.services.orchestrator.strategies import TTSStrategyFactory

strategy = TTSStrategyFactory.create_strategy(
    in_process_mode=True,
    tts_instance=tts_instance,
    tts_client=tts_client
)
```

**Logic:**
- If `in_process_mode=True` AND `tts_instance` exists → `InProcessTTSStrategy`
- Otherwise → `HTTPTTSStrategy`

## Benefits

1. **Eliminates if/else chains** - No more complex conditional logic
2. **Easy to test** - Each strategy can be tested independently
3. **Easy to extend** - Add new strategies without modifying existing code
4. **Clear separation** - Processing logic separated from orchestration
5. **Type safe** - Full type hints and abstract base classes

## Adding New Strategies

To add a new strategy:

1. Create a new strategy class inheriting from `LLMStrategy` or `TTSStrategy`
2. Implement the abstract methods
3. Update the factory to include the new strategy
4. Add unit tests

**Example:**
```python
# strategies/custom_llm_strategy.py
from .llm_strategy import LLMStrategy

class CustomLLMStrategy(LLMStrategy):
    """Custom LLM processing strategy."""
    
    async def process_audio(self, ...) -> Tuple[str, str, Dict[str, Any]]:
        # Custom implementation
        pass
```

## Testing

Comprehensive unit tests in `tests/unit/test_strategies.py`:

- InProcessLLMStrategy tests
- HTTPLLMStrategy tests
- InProcessTTSStrategy tests
- HTTPTTSStrategy tests
- Factory tests

Run tests:
```bash
pytest src/services/orchestrator/tests/unit/test_strategies.py -v
```

## Integration

Strategies are used in `TurnProcessor`:

```python
# In TurnProcessor.__init__
self.llm_strategy = LLMStrategyFactory.create_strategy(...)
self.tts_strategy = TTSStrategyFactory.create_strategy(...)

# In TurnProcessor._generate_llm_response
text_response, llm_used, llm_result = await self.llm_strategy.process_audio(...)

# In TurnProcessor._synthesize_audio
audio_response = await self.tts_strategy.synthesize(...)
```

This eliminates all if/else chains for mode selection!
