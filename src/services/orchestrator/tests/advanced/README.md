# Orchestrator Advanced - Test Suite

Comprehensive advanced test suite for Orchestrator with 20 tests covering workflow orchestration and error recovery.

## Test Coverage: 20 Tests

- **Workflow Orchestration:** 10 tests
- **Error Recovery:** 10 tests

## Running Tests

```bash
# All advanced tests
pytest src/services/orchestrator/tests/advanced/ -v

# Specific category
pytest src/services/orchestrator/tests/advanced/test_workflow_orchestration.py -v
pytest src/services/orchestrator/tests/advanced/test_error_recovery.py -v
```

## Key Features Tested

### Workflow Orchestration
- Workflow definition
- Workflow execution
- Step completion tracking
- Input data handling
- Output data collection
- Workflow cancellation
- Running workflow tracking
- Load balancing
- Load rebalancing
- Performance optimization

### Error Recovery
- Workflow failure handling
- Compensation handler registration
- Compensation execution
- Retry with exponential backoff
- Maximum retry attempts
- Retry attempt reset
- Partial workflow compensation
- Bottleneck identification
- Load balancing fairness
- Execution time tracking

## Configuration

Default configuration values:

```python
{
    "max_concurrent_workflows": 100,
    "workflow_timeout": 300.0,
    "retry_attempts": 3,
    "retry_delay": 5.0,
    "enable_workflow_persistence": True,
    "enable_compensation": True
}
```

## Workflow Orchestration

### Defining Workflows

```python
steps = [
    {"name": "validate_input", "action": "validate"},
    {"name": "process_data", "action": "process"},
    {"name": "store_result", "action": "store"}
]

workflow = await workflow_engine.define_workflow("user_workflow", steps)
```

### Executing Workflows

```python
# Execute with input data
input_data = {"user_id": 123, "action": "transcribe"}
execution_id = await workflow_engine.execute_workflow("user_workflow", input_data)

# Check status
status = await workflow_engine.get_execution_status(execution_id)
```

### Workflow Execution Status

```python
{
    "id": "exec_123",
    "workflow_id": "user_workflow",
    "status": "running",  # or "completed", "failed", "cancelled"
    "started_at": datetime,
    "steps_completed": 2,
    "total_steps": 3,
    "input_data": {...},
    "output_data": {...}
}
```

### Workflow Steps

```python
{
    "name": "process_data",
    "action": "llm_completion",
    "timeout": 30.0,
    "retry": True,
    "compensation": "undo_process"
}
```

### Canceling Workflows

```python
# Cancel running workflow
success = await workflow_engine.cancel_workflow(execution_id)
```

## Error Recovery

### Compensation Pattern

```python
# Register compensation handlers
async def undo_database_write():
    await database.delete(record_id)

async def undo_api_call():
    await api.cancel_request(request_id)

error_recovery.register_compensation("database_write", undo_database_write)
error_recovery.register_compensation("api_call", undo_api_call)

# Execute compensation on failure
completed_steps = ["database_write", "api_call"]
await error_recovery.compensate("failed_step", completed_steps)
```

### Compensation Order

Compensation is executed in **reverse order** of step completion:

```
Steps executed: A → B → C → FAIL
Compensation:   C ← B ← A
```

### Retry with Exponential Backoff

```python
# Retry failed step
can_retry = await error_recovery.retry_step("api_call", max_attempts=3)

# Backoff delays: 1s, 2s, 4s, 8s, ...
```

### Retry Strategy

- **Base Delay:** 1 second
- **Multiplier:** 2x (exponential)
- **Max Attempts:** 3 (configurable)
- **Max Delay:** Capped at reasonable limit

### Reset After Success

```python
# Reset retry counter after successful execution
error_recovery.reset_attempts("api_call")
```

## Load Balancing

### Worker Assignment

```python
# Assign workflow to least loaded worker
worker = load_balancer.assign_workflow("workflow_123")

# Complete workflow to reduce load
load_balancer.complete_workflow("workflow_123")
```

### Load Distribution

```python
distribution = load_balancer.get_load_distribution()
# {
#     "worker_1": 10,
#     "worker_2": 8,
#     "worker_3": 12
# }
```

### Load Balancing Algorithm

- **Strategy:** Least loaded worker first
- **Metric:** Number of active workflows
- **Rebalancing:** Automatic on workflow completion

### Checking Balance

```python
# Check if load is balanced (within 30% threshold)
is_balanced = load_balancer.is_balanced(threshold=0.3)
```

## Performance Optimization

### Recording Execution Times

```python
performance_optimizer.record_execution("workflow_id", duration_ms=150.0)
```

### Identifying Bottlenecks

```python
step_times = {
    "validate": 10.0,
    "llm_call": 500.0,  # Bottleneck
    "store": 15.0
}

bottleneck = performance_optimizer.identify_bottleneck("workflow_id", step_times)
# "llm_call"
```

### Optimization Suggestions

```python
avg_time = performance_optimizer.get_avg_execution_time("workflow_id")
suggestions = performance_optimizer.suggest_optimization("workflow_id", avg_time)

# [
#     "Consider parallel execution of independent steps",
#     "Review step implementations for optimization",
#     "Consider caching intermediate results"
# ]
```

## Workflow Patterns

### Sequential Workflow

```python
steps = [
    {"name": "step1"},
    {"name": "step2"},
    {"name": "step3"}
]
# Executes: step1 → step2 → step3
```

### Conditional Workflow

```python
{
    "name": "conditional_step",
    "condition": "result.status == 'success'",
    "then": "next_step",
    "else": "error_handler"
}
```

### Parallel Workflow

```python
{
    "name": "parallel_processing",
    "parallel": [
        {"name": "task1"},
        {"name": "task2"},
        {"name": "task3"}
    ]
}
```

### Saga Pattern (with Compensation)

```python
# Transaction workflow
steps = [
    {"name": "reserve_inventory", "compensation": "release_inventory"},
    {"name": "charge_payment", "compensation": "refund_payment"},
    {"name": "ship_order", "compensation": "cancel_shipment"}
]

# If any step fails, compensation runs in reverse
```

## Best Practices

### Workflow Design
- Keep workflows focused and single-purpose
- Define clear input/output contracts
- Set appropriate timeouts for each step
- Use compensation for distributed transactions

### Error Handling
- Implement idempotent operations
- Use compensation for rollback scenarios
- Set reasonable retry limits
- Log failures for debugging

### Performance
- Identify and optimize bottlenecks
- Use parallel execution where possible
- Cache intermediate results
- Monitor execution times

### Load Balancing
- Distribute load evenly across workers
- Monitor worker health
- Handle worker failures gracefully
- Scale workers based on demand

## Common Scenarios

### Voice Transcription Workflow

```python
steps = [
    {"name": "validate_audio", "timeout": 5.0},
    {"name": "transcribe_stt", "timeout": 60.0, "retry": True},
    {"name": "process_llm", "timeout": 30.0, "retry": True},
    {"name": "synthesize_tts", "timeout": 30.0, "retry": True},
    {"name": "store_result", "timeout": 10.0, "compensation": "delete_result"}
]
```

### User Registration Workflow

```python
steps = [
    {"name": "validate_input", "timeout": 2.0},
    {"name": "create_user", "compensation": "delete_user"},
    {"name": "send_email", "retry": True},
    {"name": "create_session", "compensation": "destroy_session"}
]
```

### Data Pipeline Workflow

```python
steps = [
    {"name": "extract_data", "timeout": 60.0},
    {"name": "transform_data", "timeout": 120.0},
    {"name": "validate_data", "timeout": 30.0},
    {"name": "load_data", "compensation": "rollback_load"}
]
```

## Monitoring

### Key Metrics

- **Workflow Success Rate:** % of workflows completed successfully
- **Average Execution Time:** Mean workflow duration
- **P95 Latency:** 95th percentile execution time
- **Retry Rate:** % of steps that required retry
- **Compensation Rate:** % of workflows requiring compensation

### Alerts

- Workflow execution time > threshold
- Failure rate > threshold
- Worker load imbalance
- Compensation failures

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.asyncio` - Async tests (all tests are async)

---

**Last Updated:** 2025-10-19
