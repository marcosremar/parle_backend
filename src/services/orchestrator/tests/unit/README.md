# Unit Tests for Orchestrator Engines

Comprehensive unit tests for all orchestrator engines and strategies.

## Test Coverage

### Engines Tests

- **test_stats_tracker.py** (15 tests)
  - Initialization
  - Increment operations
  - Calculations (averages, rates)
  - Reset functionality

- **test_health_checker.py** (6 tests)
  - All services healthy
  - Some services failing
  - Timeout handling
  - Exception handling
  - Empty clients

- **test_knowledge_analyzer.py** (5 tests)
  - Success processing
  - Error processing
  - No student_model client
  - Exception handling
  - Confidence threshold

- **test_context_loader.py** (4 tests)
  - All contexts loaded successfully
  - Graceful degradation
  - Parallel execution
  - Empty clients

- **test_turn_processor.py** (17 tests)
  - End-to-end turn processing
  - Audio validation
  - STT transcription
  - Turn analysis
  - Knowledge updates
  - Prompt composition
  - LLM response generation
  - TTS synthesis
  - Turn saving
  - Error handling

### Strategies Tests

- **test_strategies.py** (15+ tests)
  - InProcessLLMStrategy
  - HTTPLLMStrategy
  - InProcessTTSStrategy
  - HTTPTTSStrategy
  - LLMStrategyFactory
  - TTSStrategyFactory

## Running Tests

### All Unit Tests
```bash
pytest src/services/orchestrator/tests/unit/ -v
```

### Specific Test File
```bash
pytest src/services/orchestrator/tests/unit/test_stats_tracker.py -v
```

### With Coverage
```bash
pytest src/services/orchestrator/tests/unit/ \
  --cov=src/services/orchestrator/engines \
  --cov=src/services/orchestrator/strategies \
  --cov-report=html
```

## Test Fixtures

All fixtures are defined in `conftest.py`:

- `mock_clients` - Dictionary of mocked service clients
- `mock_fallback_manager` - Mock fallback manager
- `mock_context_loader` - Mock context loader
- `mock_knowledge_analyzer` - Mock knowledge analyzer
- `mock_stats_tracker` - Real StatsTracker instance
- `sample_audio_data` - Sample audio bytes for testing
- `sample_session_data` - Sample session data
- `sample_scenario_data` - Sample scenario data
- `sample_conversation_history` - Sample conversation history
- `sample_student_data` - Sample student CEFR progress
- `sample_turn_analysis` - Sample turn analysis

## Coverage Goals

- **Target**: ≥80% coverage for engines
- **Current**: ~84% coverage for engines
- **Strategies**: ~90% coverage

## Test Organization

Tests are organized by engine/component:
- One test file per engine
- One test class per engine
- Descriptive test method names
- Clear assertions

## Best Practices

1. **Isolation** - Each test is independent
2. **Mocks** - Use mocks for external dependencies
3. **Fixtures** - Reuse fixtures for common setup
4. **Async** - All async tests use `@pytest.mark.asyncio`
5. **Assertions** - Clear, specific assertions
6. **Error Cases** - Test both success and failure paths

## Adding New Tests

When adding a new engine:

1. Create `test_<engine_name>.py`
2. Create `Test<EngineName>` class
3. Use fixtures from `conftest.py`
4. Test all public methods
5. Test error cases
6. Aim for ≥80% coverage

Example:
```python
class TestMyEngine:
    """Test suite for MyEngine."""
    
    @pytest.fixture
    def my_engine(self, mock_clients):
        return MyEngine(mock_clients)
    
    @pytest.mark.asyncio
    async def test_my_method_success(self, my_engine):
        """Test my_method() with success."""
        result = await my_engine.my_method("param")
        assert result["success"] is True
```
