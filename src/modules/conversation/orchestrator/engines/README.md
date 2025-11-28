# Orchestrator Engines

This directory contains modular engine classes that handle specific responsibilities in the orchestrator service.

## Architecture

The orchestrator has been refactored to use a modular engine architecture, where each engine handles a specific concern:

```
ConversationOrchestrator (orchestrator_engine.py)
    ├── TurnProcessor          # Processes complete conversation turns
    ├── ContextLoader          # Loads context in parallel
    ├── KnowledgeAnalyzer      # Analyzes and updates student knowledge
    ├── StatsTracker           # Tracks statistics
    └── HealthChecker          # Checks service health
```

## Engines

### TurnProcessor

**File:** `turn_processor.py`

Processes complete conversation turns end-to-end, coordinating STT, LLM, and TTS.

**Responsibilities:**
- Transcribe audio to text (STT)
- Analyze turn for errors and skills
- Update knowledge before response generation
- Compose pedagogical prompt
- Generate LLM response
- Synthesize audio (TTS)
- Save turn and session data

**Usage:**
```python
from src.services.orchestrator.engines import TurnProcessor

processor = TurnProcessor(
    clients=clients,
    fallback_manager=fallback_manager,
    context_loader=context_loader,
    knowledge_analyzer=knowledge_analyzer,
    stats_tracker=stats_tracker,
    in_process_mode=False
)

result = await processor.process_turn(
    audio_data=audio_bytes,
    session_id="session_123",
    sample_rate=16000
)
```

**Key Methods:**
- `process_turn()` - Main entry point for processing a turn
- `_transcribe_audio()` - Transcribes audio to text
- `_analyze_turn()` - Analyzes turn for errors and skills
- `_generate_llm_response()` - Generates LLM response using strategy pattern
- `_synthesize_audio()` - Synthesizes audio using strategy pattern

### ContextLoader

**File:** `context_loader.py`

Loads conversation context in parallel for better performance.

**Responsibilities:**
- Load session data
- Load scenario data (parallel)
- Load conversation history (parallel)
- Load student CEFR progress (parallel)
- Load target skill (parallel)

**Usage:**
```python
from src.services.orchestrator.engines import ContextLoader

loader = ContextLoader(clients)

scenario_data, history, student_data, target_skill, conv_id, scen_id, user_id = await loader.load_context(
    session_id="session_123"
)
```

**Key Features:**
- Parallel execution using `asyncio.gather()`
- Graceful degradation when services fail
- Returns None for missing data instead of raising exceptions

### KnowledgeAnalyzer

**File:** `knowledge_analyzer.py`

Analyzes student knowledge and updates skill mastery.

**Responsibilities:**
- Analyze turn for successes and errors
- Update student model with skill assessments
- Process high-confidence skills only
- Handle errors gracefully

**Usage:**
```python
from src.services.orchestrator.engines import KnowledgeAnalyzer

analyzer = KnowledgeAnalyzer(clients)

await analyzer.analyze_and_update_knowledge(
    user_id="user_123",
    target_skill={"skill_id": "skill_1"},
    user_text="Hello",
    ai_text="Hi there!"
)
```

**Key Features:**
- Background task (doesn't block main flow)
- Processes skills above confidence threshold
- Handles both successes and errors
- Graceful degradation when services unavailable

### StatsTracker

**File:** `stats_tracker.py`

Centralized statistics tracking for the orchestrator.

**Responsibilities:**
- Track total/successful/failed turns
- Track LLM usage (primary, fallback, in-process, HTTP)
- Track processing times
- Calculate success rates and averages

**Usage:**
```python
from src.services.orchestrator.engines import StatsTracker

tracker = StatsTracker()
tracker.increment_total_turns()
tracker.increment_successful_turns()
tracker.add_processing_time(1.5)

stats = tracker.get_stats()
avg_time_ms = tracker.get_average_processing_time_ms()
success_rate = tracker.get_success_rate()
```

**Key Methods:**
- `increment_*()` - Increment various counters
- `get_stats()` - Get all statistics (returns copy)
- `get_average_processing_time_ms()` - Calculate average
- `get_success_rate()` - Calculate success rate
- `reset()` - Reset all statistics

### HealthChecker

**File:** `health_checker.py`

Checks health of downstream services.

**Responsibilities:**
- Check health of all service clients
- Handle timeouts gracefully
- Handle exceptions gracefully
- Return health status dictionary

**Usage:**
```python
from src.services.orchestrator.engines import HealthChecker

checker = HealthChecker(clients)

health_status = await checker.check_all_services()
# Returns: {"stt": True, "llm": True, "tts": False, ...}
```

**Key Features:**
- Checks all services in sequence
- Logs health status for each service
- Returns summary of healthy/unhealthy services
- Handles exceptions without failing completely

## Adding New Engines

To add a new engine:

1. Create a new file in `engines/` directory
2. Implement the engine class with clear responsibilities
3. Add to `engines/__init__.py` exports
4. Inject into `ConversationOrchestrator` if needed
5. Add unit tests in `tests/unit/test_<engine_name>.py`

**Example:**
```python
# engines/my_engine.py
from __future__ import annotations
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MyEngine:
    """Description of what this engine does."""
    
    def __init__(self, clients: Dict[str, Any]) -> None:
        self.clients = clients
    
    async def do_something(self, param: str) -> Dict[str, Any]:
        """Do something useful."""
        # Implementation
        pass
```

## Testing

Each engine has comprehensive unit tests in `tests/unit/`:

- `test_stats_tracker.py` - 15 tests
- `test_health_checker.py` - 6 tests
- `test_knowledge_analyzer.py` - 5 tests
- `test_context_loader.py` - 4 tests
- `test_turn_processor.py` - (to be added)

Run tests:
```bash
pytest src/services/orchestrator/tests/unit/ -v
```

## Dependencies

Engines are designed to be:
- **Independent** - Each engine has a single responsibility
- **Testable** - Easy to mock and test in isolation
- **Injectable** - Dependencies passed via constructor
- **Async** - All I/O operations are async

## Design Principles

1. **Single Responsibility** - Each engine does one thing well
2. **Dependency Injection** - Dependencies passed in, not created
3. **Graceful Degradation** - Handle failures without crashing
4. **Parallel Execution** - Use `asyncio.gather()` when possible
5. **Type Safety** - Full type hints with `from __future__ import annotations`
