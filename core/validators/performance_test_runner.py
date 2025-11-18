"""
Performance Test Runner
Framework for running performance tests across different scenarios
"""

import asyncio
import logging
import time
import statistics
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .metrics import get_metrics_collector
from .structured_logger import get_logger
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)
structured_logger = get_logger("PerformanceTestRunner")
metrics = get_metrics_collector()


class TestScenario(Enum):
    """Available test scenarios"""
    SINGLE_REQUEST = "single_request_baseline"
    SUSTAINED_LOAD = "sustained_load"
    BURST_LOAD = "burst_load"
    MIXED_PIPELINE = "mixed_pipeline"
    LONG_CONTEXT = "long_context"
    MEMORY_STRESS = "memory_stress"


@dataclass
class TestConfig:
    """Configuration for a test scenario"""
    name: str
    description: str
    requests: int = 100
    concurrency: int = 1
    duration_seconds: Optional[int] = None
    requests_per_second: Optional[int] = None
    sequence_length: Optional[int] = None
    batch_size: int = 1


@dataclass
class TestRequest:
    """Individual test request"""
    request_id: str
    module: str  # 'stt', 'llm', 'tts'
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a single test request"""
    request_id: str
    module: str
    success: bool
    latency_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Aggregated results for a test scenario"""
    scenario: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_ms: float
    latencies_ms: List[float]

    # Latency metrics
    avg_latency_ms: float
    median_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput metrics
    requests_per_second: float
    success_rate_percent: float
    error_rate_percent: float

    # Resource metrics (optional)
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    cpu_memory_used_mb: Optional[float] = None

    # Additional metrics
    first_token_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None

    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTestRunner:
    """
    Performance test runner for service benchmarking

    Features:
    - 6 predefined test scenarios
    - Concurrent request execution
    - Latency percentile calculation
    - GPU/CPU resource monitoring
    - Detailed metrics collection
    """

    def __init__(self, profile_id: str = "minimal_gpu"):
        """
        Initialize performance test runner

        Args:
            profile_id: GPU profile to test with
        """
        self.profile_id = profile_id
        self.test_results: List[ScenarioResult] = []

        # Predefined scenarios from gpu_profiles.yaml
        self.scenarios = {
            TestScenario.SINGLE_REQUEST: TestConfig(
                name="single_request_baseline",
                description="Single request latency (P50, P95, P99)",
                requests=100,
                concurrency=1
            ),
            TestScenario.SUSTAINED_LOAD: TestConfig(
                name="sustained_load",
                description="Sustained load over time",
                duration_seconds=120,
                requests_per_second=10
            ),
            TestScenario.BURST_LOAD: TestConfig(
                name="burst_load",
                description="Burst of concurrent requests",
                requests=50,
                concurrency=50
            ),
            TestScenario.MIXED_PIPELINE: TestConfig(
                name="mixed_pipeline",
                description="Full STT→LLM→TTS pipeline",
                requests=50,
                concurrency=4
            ),
            TestScenario.LONG_CONTEXT: TestConfig(
                name="long_context",
                description="Maximum sequence length",
                requests=20,
                sequence_length=2048
            ),
            TestScenario.MEMORY_STRESS: TestConfig(
                name="memory_stress",
                description="Memory leak detection",
                requests=1000,
                concurrency=8,
                duration_seconds=600
            )
        }

        structured_logger.info(
            f"🧪 PerformanceTestRunner initialized for profile '{profile_id}'",
            metadata={'scenarios': len(self.scenarios)}
        )

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile from list of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        return sorted_values[min(index, len(sorted_values) - 1)]

    async def _execute_request(
        self,
        request: TestRequest,
        executor: Callable
    ) -> TestResult:
        """Execute a single test request"""
        start_time = time.time()

        try:
            # Execute request via provided executor function
            response = await executor(request)

            latency_ms = (time.time() - start_time) * 1000

            return TestResult(
                request_id=request.request_id,
                module=request.module,
                success=True,
                latency_ms=latency_ms,
                metadata={
                    'response': response,
                    **request.metadata
                }
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            return TestResult(
                request_id=request.request_id,
                module=request.module,
                success=False,
                latency_ms=latency_ms,
                error=str(e),
                metadata=request.metadata
            )

    async def _execute_concurrent_batch(
        self,
        requests: List[TestRequest],
        executor: Callable,
        concurrency: int
    ) -> List[TestResult]:
        """Execute requests with controlled concurrency"""
        results = []
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_execute(req):
            async with semaphore:
                return await self._execute_request(req, executor)

        tasks = [limited_execute(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(TestResult(
                    request_id=requests[i].request_id,
                    module=requests[i].module,
                    success=False,
                    latency_ms=0,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        return final_results

    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics"""
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)  # MB
                memory_reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)  # MB

                # Try to get GPU utilization (requires nvidia-ml-py3)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    pynvml.nvmlShutdown()

                    return {
                        'gpu_memory_used_mb': info.used / (1024 * 1024),
                        'gpu_memory_peak_mb': memory_reserved,
                        'gpu_utilization_percent': util.gpu
                    }
                except Exception as e:
                    return {
                        'gpu_memory_used_mb': memory_allocated,
                        'gpu_memory_peak_mb': memory_reserved,
                        'gpu_utilization_percent': None
                    }
        except Exception as e:
            pass

        return {}

    def _get_cpu_metrics(self) -> Dict[str, float]:
        """Get current CPU metrics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'cpu_memory_used_mb': memory_info.rss / (1024 * 1024)
            }
        except Exception as e:
            return {}

    def _aggregate_results(
        self,
        scenario: TestConfig,
        results: List[TestResult],
        duration_ms: float
    ) -> ScenarioResult:
        """Aggregate test results into scenario result"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        latencies = [r.latency_ms for r in successful] if successful else [0]

        # Calculate metrics
        avg_latency = statistics.mean(latencies) if latencies else 0
        median_latency = statistics.median(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0

        # Get resource metrics
        gpu_metrics = self._get_gpu_metrics()
        cpu_metrics = self._get_cpu_metrics()

        return ScenarioResult(
            scenario=scenario.name,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_duration_ms=duration_ms,
            latencies_ms=latencies,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p50_latency_ms=self._calculate_percentile(latencies, 50),
            p95_latency_ms=self._calculate_percentile(latencies, 95),
            p99_latency_ms=self._calculate_percentile(latencies, 99),
            requests_per_second=len(results) / (duration_ms / 1000) if duration_ms > 0 else 0,
            success_rate_percent=(len(successful) / len(results) * 100) if results else 0,
            error_rate_percent=(len(failed) / len(results) * 100) if results else 0,
            gpu_memory_used_mb=gpu_metrics.get('gpu_memory_used_mb'),
            gpu_memory_peak_mb=gpu_metrics.get('gpu_memory_peak_mb'),
            gpu_utilization_percent=gpu_metrics.get('gpu_utilization_percent'),
            cpu_memory_used_mb=cpu_metrics.get('cpu_memory_used_mb'),
            errors=[r.error for r in failed if r.error],
            metadata={
                'scenario_config': {
                    'name': scenario.name,
                    'description': scenario.description,
                    'requests': scenario.requests,
                    'concurrency': scenario.concurrency,
                    'duration_seconds': scenario.duration_seconds,
                    'sequence_length': scenario.sequence_length
                }
            }
        )

    async def run_scenario(
        self,
        scenario: TestScenario,
        request_generator: Callable[[int, TestConfig], List[TestRequest]],
        executor: Callable
    ) -> ScenarioResult:
        """
        Run a specific test scenario

        Args:
            scenario: Test scenario to run
            request_generator: Function to generate test requests
            executor: Function to execute requests

        Returns:
            ScenarioResult with aggregated metrics
        """
        config = self.scenarios[scenario]

        structured_logger.info(
            f"🧪 Running scenario: {config.name}",
            metadata={'description': config.description}
        )

        start_time = time.time()

        # Generate test requests
        if config.duration_seconds and config.requests_per_second:
            # Time-based scenario
            total_requests = config.duration_seconds * config.requests_per_second
            requests = request_generator(total_requests, config)

            # Execute with rate limiting
            results = []
            request_interval = 1.0 / config.requests_per_second

            for i, req in enumerate(requests):
                iter_start = time.time()

                result = await self._execute_request(req, executor)
                results.append(result)

                # Rate limiting
                elapsed = time.time() - iter_start
                if elapsed < request_interval:
                    await asyncio.sleep(request_interval - elapsed)

                # Check duration
                if time.time() - start_time >= config.duration_seconds:
                    break
        else:
            # Request count-based scenario
            requests = request_generator(config.requests, config)
            results = await self._execute_concurrent_batch(
                requests,
                executor,
                config.concurrency
            )

        duration_ms = (time.time() - start_time) * 1000

        # Aggregate results
        scenario_result = self._aggregate_results(config, results, duration_ms)
        self.test_results.append(scenario_result)

        # Log summary
        structured_logger.info(
            f"✅ Scenario '{config.name}' complete",
            metadata={
                'duration_ms': duration_ms,
                'total_requests': scenario_result.total_requests,
                'success_rate': f"{scenario_result.success_rate_percent:.1f}%",
                'avg_latency_ms': scenario_result.avg_latency_ms,
                'p95_latency_ms': scenario_result.p95_latency_ms,
                'p99_latency_ms': scenario_result.p99_latency_ms
            }
        )

        return scenario_result

    async def run_all_scenarios(
        self,
        request_generator: Callable[[int, TestConfig], List[TestRequest]],
        executor: Callable,
        scenarios: Optional[List[TestScenario]] = None
    ) -> List[ScenarioResult]:
        """
        Run all or selected test scenarios

        Args:
            request_generator: Function to generate test requests
            executor: Function to execute requests
            scenarios: Optional list of scenarios to run (runs all if None)

        Returns:
            List of ScenarioResult for each scenario
        """
        scenarios_to_run = scenarios or list(TestScenario)

        structured_logger.info(
            f"🚀 Running {len(scenarios_to_run)} test scenarios",
            metadata={'profile': self.profile_id}
        )

        results = []
        for scenario in scenarios_to_run:
            try:
                result = await self.run_scenario(scenario, request_generator, executor)
                results.append(result)

                # Pause between scenarios
                await asyncio.sleep(2)

            except Exception as e:
                structured_logger.error(
                    f"❌ Scenario '{scenario.value}' failed",
                    exception=e
                )

        return results

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all test results"""
        if not self.test_results:
            return {'error': 'No test results available'}

        return {
            'profile_id': self.profile_id,
            'total_scenarios': len(self.test_results),
            'timestamp': datetime.now().isoformat(),
            'scenarios': [
                {
                    'name': r.scenario,
                    'total_requests': r.total_requests,
                    'success_rate': f"{r.success_rate_percent:.1f}%",
                    'avg_latency_ms': round(r.avg_latency_ms, 2),
                    'p50_latency_ms': round(r.p50_latency_ms, 2),
                    'p95_latency_ms': round(r.p95_latency_ms, 2),
                    'p99_latency_ms': round(r.p99_latency_ms, 2),
                    'requests_per_second': round(r.requests_per_second, 2),
                    'gpu_memory_mb': round(r.gpu_memory_used_mb, 2) if r.gpu_memory_used_mb else None,
                    'gpu_utilization': f"{r.gpu_utilization_percent:.1f}%" if r.gpu_utilization_percent else None
                }
                for r in self.test_results
            ],
            'overall': {
                'total_requests': sum(r.total_requests for r in self.test_results),
                'avg_success_rate': statistics.mean([r.success_rate_percent for r in self.test_results]),
                'avg_latency_ms': statistics.mean([r.avg_latency_ms for r in self.test_results]),
                'avg_p95_latency_ms': statistics.mean([r.p95_latency_ms for r in self.test_results])
            }
        }


# Example request generators for different modules

def stt_request_generator(count: int, config: TestConfig) -> List[TestRequest]:
    """Generate STT test requests"""
    requests = []

    for i in range(count):
        # Generate synthetic audio
        duration = 3.0  # 3 seconds
        sample_rate = 16000
        audio = np.random.randn(int(sample_rate * duration)) * 0.01
        audio = audio.astype(np.float32)

        requests.append(TestRequest(
            request_id=f"stt_{i}",
            module="stt",
            payload={
                'audio': audio,
                'sample_rate': sample_rate,
                'language': 'pt'
            },
            metadata={
                'iteration': i,
                'scenario': config.name
            }
        ))

    return requests


def llm_request_generator(count: int, config: TestConfig) -> List[TestRequest]:
    """Generate LLM test requests"""
    requests = []

    seq_len = config.sequence_length or 128

    for i in range(count):
        # Generate prompt with target sequence length
        base_prompt = "Responda de forma concisa: "
        padding = "contexto " * max(0, (seq_len - len(base_prompt)) // 10)
        prompt = base_prompt + padding

        requests.append(TestRequest(
            request_id=f"llm_{i}",
            module="llm",
            payload={
                'prompt': prompt,
                'max_tokens': 50,
                'temperature': 0.7
            },
            metadata={
                'iteration': i,
                'scenario': config.name,
                'sequence_length': seq_len
            }
        ))

    return requests


def tts_request_generator(count: int, config: TestConfig) -> List[TestRequest]:
    """Generate TTS test requests"""
    requests = []

    texts = [
        "Olá, como posso ajudar?",
        "Sistema de síntese de voz inicializado",
        "Este é um teste de conversão de texto para fala"
    ]

    for i in range(count):
        text = texts[i % len(texts)]

        requests.append(TestRequest(
            request_id=f"tts_{i}",
            module="tts",
            payload={
                'text': text,
                'voice_id': 'af_sky',
                'speed': 1.0
            },
            metadata={
                'iteration': i,
                'scenario': config.name
            }
        ))

    return requests
