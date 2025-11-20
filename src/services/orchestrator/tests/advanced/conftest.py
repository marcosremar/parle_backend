"""Shared fixtures for Orchestrator Advanced tests."""

import pytest
from typing import Dict, List, Optional
from datetime import datetime
import asyncio


@pytest.fixture
def orchestrator_config():
    """Orchestrator configuration."""
    return {
        "max_concurrent_workflows": 100,
        "workflow_timeout": 300.0,
        "retry_attempts": 3,
        "retry_delay": 5.0,
        "enable_workflow_persistence": True,
        "enable_compensation": True
    }


@pytest.fixture
def workflow_engine():
    """Workflow orchestration engine."""

    class WorkflowEngine:
        def __init__(self):
            self.workflows = {}
            self.executions = {}

        async def define_workflow(self, workflow_id: str, steps: List[Dict]) -> Dict:
            """Define a workflow."""
            self.workflows[workflow_id] = {
                "id": workflow_id,
                "steps": steps,
                "created_at": datetime.now()
            }
            return self.workflows[workflow_id]

        async def execute_workflow(self, workflow_id: str, input_data: Dict = None) -> str:
            """Execute a workflow."""
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")

            execution_id = f"exec_{len(self.executions)}"
            workflow = self.workflows[workflow_id]

            self.executions[execution_id] = {
                "id": execution_id,
                "workflow_id": workflow_id,
                "status": "running",
                "started_at": datetime.now(),
                "steps_completed": 0,
                "total_steps": len(workflow["steps"]),
                "input_data": input_data or {},
                "output_data": {}
            }

            # Simulate workflow execution
            try:
                for i, step in enumerate(workflow["steps"]):
                    await self._execute_step(execution_id, step)
                    self.executions[execution_id]["steps_completed"] = i + 1

                self.executions[execution_id]["status"] = "completed"
                self.executions[execution_id]["completed_at"] = datetime.now()

            except Exception as e:
                self.executions[execution_id]["status"] = "failed"
                self.executions[execution_id]["error"] = str(e)

            return execution_id

        async def _execute_step(self, execution_id: str, step: Dict):
            """Execute a workflow step."""
            await asyncio.sleep(0.01)  # Simulate work

            if step.get("fail", False):
                raise Exception(f"Step {step['name']} failed")

            execution = self.executions[execution_id]
            execution["output_data"][step["name"]] = step.get("output", "success")

        async def get_execution_status(self, execution_id: str) -> Optional[Dict]:
            """Get workflow execution status."""
            return self.executions.get(execution_id)

        async def cancel_workflow(self, execution_id: str) -> bool:
            """Cancel running workflow."""
            if execution_id not in self.executions:
                return False

            execution = self.executions[execution_id]
            if execution["status"] == "running":
                execution["status"] = "cancelled"
                execution["cancelled_at"] = datetime.now()
                return True

            return False

        def get_running_workflows(self) -> List[Dict]:
            """Get all running workflows."""
            return [e for e in self.executions.values() if e["status"] == "running"]

    return WorkflowEngine()


@pytest.fixture
def error_recovery():
    """Error recovery and compensation."""

    class ErrorRecovery:
        def __init__(self):
            self.compensation_handlers = {}
            self.recovery_attempts = {}

        def register_compensation(self, step_name: str, handler):
            """Register compensation handler for step."""
            self.compensation_handlers[step_name] = handler

        async def compensate(self, failed_step: str, completed_steps: List[str]) -> bool:
            """Execute compensation for failed workflow."""
            # Compensate in reverse order
            for step in reversed(completed_steps):
                if step in self.compensation_handlers:
                    try:
                        await self.compensation_handlers[step]()
                    except Exception:
                        return False

            return True

        async def retry_step(self, step_name: str, max_attempts: int = 3) -> bool:
            """Retry failed step with exponential backoff."""
            if step_name not in self.recovery_attempts:
                self.recovery_attempts[step_name] = 0

            attempt = self.recovery_attempts[step_name]

            if attempt >= max_attempts:
                return False

            # Exponential backoff
            delay = 2 ** attempt
            await asyncio.sleep(delay * 0.01)  # Simulated delay

            self.recovery_attempts[step_name] += 1
            return True

        def reset_attempts(self, step_name: str):
            """Reset retry attempts counter."""
            if step_name in self.recovery_attempts:
                del self.recovery_attempts[step_name]

    return ErrorRecovery()


@pytest.fixture
def load_balancer():
    """Workflow load balancing."""

    class LoadBalancer:
        def __init__(self):
            self.workers = ["worker_1", "worker_2", "worker_3"]
            self.workload = {w: 0 for w in self.workers}
            self.assignments = {}

        def assign_workflow(self, workflow_id: str) -> str:
            """Assign workflow to least loaded worker."""
            # Find worker with minimum load
            worker = min(self.workload.items(), key=lambda x: x[1])[0]

            self.workload[worker] += 1
            self.assignments[workflow_id] = worker

            return worker

        def complete_workflow(self, workflow_id: str):
            """Mark workflow as completed and reduce worker load."""
            if workflow_id in self.assignments:
                worker = self.assignments[workflow_id]
                self.workload[worker] = max(0, self.workload[worker] - 1)
                del self.assignments[workflow_id]

        def get_worker_load(self, worker: str) -> int:
            """Get current load for worker."""
            return self.workload.get(worker, 0)

        def get_load_distribution(self) -> Dict[str, int]:
            """Get load distribution across workers."""
            return self.workload.copy()

        def is_balanced(self, threshold: float = 0.3) -> bool:
            """Check if load is balanced across workers."""
            loads = list(self.workload.values())
            if not loads:
                return True

            avg_load = sum(loads) / len(loads)
            if avg_load == 0:
                return True

            # Check if any worker deviates more than threshold from average
            for load in loads:
                deviation = abs(load - avg_load) / avg_load
                if deviation > threshold:
                    return False

            return True

    return LoadBalancer()


@pytest.fixture
def performance_optimizer():
    """Workflow performance optimization."""

    class PerformanceOptimizer:
        def __init__(self):
            self.execution_times = {}
            self.bottlenecks = []

        def record_execution(self, workflow_id: str, duration_ms: float):
            """Record workflow execution time."""
            if workflow_id not in self.execution_times:
                self.execution_times[workflow_id] = []

            self.execution_times[workflow_id].append(duration_ms)

        def get_avg_execution_time(self, workflow_id: str) -> float:
            """Get average execution time."""
            times = self.execution_times.get(workflow_id, [])
            return sum(times) / len(times) if times else 0

        def identify_bottleneck(self, workflow_id: str, step_times: Dict[str, float]) -> str:
            """Identify bottleneck step."""
            if not step_times:
                return None

            bottleneck = max(step_times.items(), key=lambda x: x[1])
            self.bottlenecks.append({
                "workflow": workflow_id,
                "step": bottleneck[0],
                "duration_ms": bottleneck[1]
            })

            return bottleneck[0]

        def suggest_optimization(self, workflow_id: str, avg_time: float) -> List[str]:
            """Suggest workflow optimizations."""
            suggestions = []

            if avg_time > 1000:
                suggestions.append("Consider parallel execution of independent steps")

            if avg_time > 500:
                suggestions.append("Review step implementations for optimization")

            if len(self.execution_times.get(workflow_id, [])) > 10:
                suggestions.append("Consider caching intermediate results")

            return suggestions

    return PerformanceOptimizer()
