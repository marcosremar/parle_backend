"""
Profile Optimizer
Background testing engine for finding optimal GPU configurations
"""

import asyncio
import logging
import time
import json
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .gpu_profile_manager import get_gpu_profile_manager
from .performance_test_runner import (
    PerformanceTestRunner,
    TestScenario,
    ScenarioResult,
    TestRequest,
    TestConfig
)
from .structured_logger import get_logger

logger = logging.getLogger(__name__)
structured_logger = get_logger("ProfileOptimizer")


@dataclass
class OptimizationConfig:
    """Configuration for optimization run"""
    profile_id: str
    test_matrix: Dict[str, Any]
    scenarios: List[str]
    iterations: int = 50
    exploration_factor: float = 0.2
    database_path: str = "data/optimization_results.db"


@dataclass
class OptimizationResult:
    """Result of a single optimization iteration"""
    run_id: str
    profile_id: str
    iteration: int
    timestamp: str

    # Test parameters
    gpu_memory_utilization_llm: float
    gpu_memory_utilization_stt: Optional[float]
    batch_size: int
    sequence_length: int
    concurrency: int

    # Performance metrics
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    success_rate: float

    # Resource metrics
    gpu_memory_used_mb: Optional[float]
    gpu_memory_peak_mb: Optional[float]
    gpu_utilization_percent: Optional[float]

    # Quality score (0-100)
    quality_score: float

    # Additional data
    scenario_results: str  # JSON
    metadata: str  # JSON


class OptimizationDatabase:
    """SQLite database for storing optimization results"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    run_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_iterations INTEGER,
                    best_quality_score REAL,
                    status TEXT,
                    config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,

                    gpu_memory_utilization_llm REAL,
                    gpu_memory_utilization_stt REAL,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    concurrency INTEGER,

                    avg_latency_ms REAL,
                    p95_latency_ms REAL,
                    p99_latency_ms REAL,
                    throughput_rps REAL,
                    success_rate REAL,

                    gpu_memory_used_mb REAL,
                    gpu_memory_peak_mb REAL,
                    gpu_utilization_percent REAL,

                    quality_score REAL,

                    scenario_results TEXT,
                    metadata TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (run_id) REFERENCES optimization_runs(run_id)
                )
            """)

            # Indexes for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_id
                ON optimization_results(run_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_score
                ON optimization_results(quality_score DESC)
            """)

            conn.commit()

    def create_run(self, run_id: str, profile_id: str, config: Dict[str, Any]) -> None:
        """Create new optimization run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimization_runs
                (run_id, profile_id, start_time, status, config, total_iterations)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                profile_id,
                datetime.now().isoformat(),
                'running',
                json.dumps(config),
                config.get('iterations', 0)
            ))
            conn.commit()

    def save_result(self, result: OptimizationResult) -> None:
        """Save optimization result"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimization_results (
                    run_id, profile_id, iteration, timestamp,
                    gpu_memory_utilization_llm, gpu_memory_utilization_stt,
                    batch_size, sequence_length, concurrency,
                    avg_latency_ms, p95_latency_ms, p99_latency_ms,
                    throughput_rps, success_rate,
                    gpu_memory_used_mb, gpu_memory_peak_mb, gpu_utilization_percent,
                    quality_score, scenario_results, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.run_id, result.profile_id, result.iteration, result.timestamp,
                result.gpu_memory_utilization_llm, result.gpu_memory_utilization_stt,
                result.batch_size, result.sequence_length, result.concurrency,
                result.avg_latency_ms, result.p95_latency_ms, result.p99_latency_ms,
                result.throughput_rps, result.success_rate,
                result.gpu_memory_used_mb, result.gpu_memory_peak_mb,
                result.gpu_utilization_percent, result.quality_score,
                result.scenario_results, result.metadata
            ))
            conn.commit()

    def complete_run(self, run_id: str, best_score: float) -> None:
        """Mark run as complete"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE optimization_runs
                SET end_time = ?, status = ?, best_quality_score = ?
                WHERE run_id = ?
            """, (datetime.now().isoformat(), 'completed', best_score, run_id))
            conn.commit()

    def get_best_results(self, run_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top results for a run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM optimization_results
                WHERE run_id = ?
                ORDER BY quality_score DESC
                LIMIT ?
            """, (run_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary of optimization run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get run info
            run_cursor = conn.execute("""
                SELECT * FROM optimization_runs WHERE run_id = ?
            """, (run_id,))
            run_info = dict(run_cursor.fetchone())

            # Get best result
            best_cursor = conn.execute("""
                SELECT * FROM optimization_results
                WHERE run_id = ?
                ORDER BY quality_score DESC
                LIMIT 1
            """, (run_id,))
            best_result = dict(best_cursor.fetchone()) if best_cursor.rowcount > 0 else None

            # Get stats
            stats_cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_iterations,
                    AVG(quality_score) as avg_score,
                    MAX(quality_score) as max_score,
                    MIN(quality_score) as min_score,
                    AVG(avg_latency_ms) as avg_latency,
                    AVG(throughput_rps) as avg_throughput
                FROM optimization_results
                WHERE run_id = ?
            """, (run_id,))
            stats = dict(stats_cursor.fetchone())

            return {
                'run_info': run_info,
                'best_result': best_result,
                'statistics': stats
            }


class ProfileOptimizer:
    """
    Profile optimizer for finding optimal GPU configurations

    Uses performance testing to evaluate different parameter combinations
    and find the best configuration for each profile.
    """

    def __init__(self, profile_id: str):
        self.profile_id = profile_id
        self.profile_manager = get_gpu_profile_manager()

        # Get optimization config from profile
        self.optimization_config = self.profile_manager.get_optimization_config()

        # Initialize database
        db_path = self.optimization_config.get('results', {}).get(
            'database_path',
            'data/optimization_results.db'
        )
        self.db = OptimizationDatabase(db_path)

        # Test runner
        self.test_runner = PerformanceTestRunner(profile_id)

        # Current run
        self.run_id: Optional[str] = None
        self.iteration_count = 0
        self.best_score = 0.0

        structured_logger.info(
            f"🔬 ProfileOptimizer initialized for profile '{profile_id}'",
            metadata={'db_path': db_path}
        )

    def _calculate_quality_score(
        self,
        scenario_results: List[ScenarioResult],
        target_latency_ms: float = 500,
        target_throughput_rps: float = 10
    ) -> float:
        """
        Calculate quality score (0-100) based on performance metrics

        Higher score = better performance

        Factors:
        - Latency (lower is better)
        - Throughput (higher is better)
        - Success rate (higher is better)
        - Resource efficiency (lower GPU memory is better)
        """
        if not scenario_results:
            return 0.0

        # Aggregate metrics across scenarios
        avg_latency = sum(r.avg_latency_ms for r in scenario_results) / len(scenario_results)
        avg_p95_latency = sum(r.p95_latency_ms for r in scenario_results) / len(scenario_results)
        avg_throughput = sum(r.requests_per_second for r in scenario_results) / len(scenario_results)
        avg_success_rate = sum(r.success_rate_percent for r in scenario_results) / len(scenario_results)

        # GPU memory (average from results that have it)
        gpu_memory_results = [r.gpu_memory_used_mb for r in scenario_results if r.gpu_memory_used_mb]
        avg_gpu_memory = sum(gpu_memory_results) / len(gpu_memory_results) if gpu_memory_results else 0

        # Component scores (0-100 each)

        # 1. Latency score (target: 500ms avg, 1000ms p95)
        latency_score = max(0, 100 - (avg_latency / target_latency_ms * 50))
        p95_score = max(0, 100 - (avg_p95_latency / (target_latency_ms * 2) * 50))

        # 2. Throughput score (target: 10 rps)
        throughput_score = min(100, (avg_throughput / target_throughput_rps) * 100)

        # 3. Success rate score
        success_score = avg_success_rate

        # 4. Resource efficiency score (lower GPU memory = better, max 20GB)
        resource_score = max(0, 100 - (avg_gpu_memory / 20000 * 100)) if avg_gpu_memory > 0 else 50

        # Weighted average
        quality_score = (
            latency_score * 0.30 +      # 30% weight on avg latency
            p95_score * 0.25 +           # 25% weight on p95 latency
            throughput_score * 0.20 +    # 20% weight on throughput
            success_score * 0.20 +       # 20% weight on success rate
            resource_score * 0.05        # 5% weight on resource efficiency
        )

        return round(quality_score, 2)

    async def run_optimization_iteration(
        self,
        test_params: Dict[str, Any],
        executor: Any
    ) -> OptimizationResult:
        """Run single optimization iteration with given parameters"""

        structured_logger.info(
            f"🔬 Running optimization iteration {self.iteration_count + 1}",
            metadata=test_params
        )

        # Run performance tests with these parameters
        from .performance_test_runner import (
            llm_request_generator,
            stt_request_generator,
            tts_request_generator,
            TestScenario,
            TestConfig
        )
        from .service_executor import get_service_executor

        # Get or create executor
        if executor is None:
            executor = get_service_executor()

        scenario_results = []

        # Run a subset of scenarios for optimization (faster iterations)
        quick_scenarios = [
            TestScenario.SINGLE_REQUEST,
            TestScenario.MIXED_PIPELINE
        ]

        for scenario in quick_scenarios:
            try:
                result = await self.test_runner.run_scenario(
                    scenario,
                    llm_request_generator,  # Use LLM as primary test
                    executor
                )
                scenario_results.append(result)
            except Exception as e:
                structured_logger.error(
                    f"❌ Scenario {scenario.value} failed in optimization",
                    exception=e
                )

        # Calculate quality score
        quality_score = self._calculate_quality_score(scenario_results)

        # Aggregate metrics
        avg_latency = sum(r.avg_latency_ms for r in scenario_results) / len(scenario_results) if scenario_results else 0
        avg_p95 = sum(r.p95_latency_ms for r in scenario_results) / len(scenario_results) if scenario_results else 0
        avg_p99 = sum(r.p99_latency_ms for r in scenario_results) / len(scenario_results) if scenario_results else 0
        avg_throughput = sum(r.requests_per_second for r in scenario_results) / len(scenario_results) if scenario_results else 0
        avg_success = sum(r.success_rate_percent for r in scenario_results) / len(scenario_results) if scenario_results else 0

        # GPU metrics
        gpu_memory = scenario_results[0].gpu_memory_used_mb if scenario_results else None
        gpu_peak = scenario_results[0].gpu_memory_peak_mb if scenario_results else None
        gpu_util = scenario_results[0].gpu_utilization_percent if scenario_results else None

        # Create result
        result = OptimizationResult(
            run_id=self.run_id,
            profile_id=self.profile_id,
            iteration=self.iteration_count + 1,
            timestamp=datetime.now().isoformat(),
            gpu_memory_utilization_llm=test_params.get('gpu_memory_utilization_llm', 0.75),
            gpu_memory_utilization_stt=test_params.get('gpu_memory_utilization_stt'),
            batch_size=test_params.get('batch_size', 1),
            sequence_length=test_params.get('sequence_length', 128),
            concurrency=test_params.get('concurrency', 1),
            avg_latency_ms=avg_latency,
            p95_latency_ms=avg_p95,
            p99_latency_ms=avg_p99,
            throughput_rps=avg_throughput,
            success_rate=avg_success,
            gpu_memory_used_mb=gpu_memory,
            gpu_memory_peak_mb=gpu_peak,
            gpu_utilization_percent=gpu_util,
            quality_score=quality_score,
            scenario_results=json.dumps([asdict(r) for r in scenario_results]),
            metadata=json.dumps(test_params)
        )

        # Save to database
        self.db.save_result(result)

        # Update best score
        if quality_score > self.best_score:
            self.best_score = quality_score
            structured_logger.info(
                f"🎯 New best score: {quality_score:.2f}",
                metadata={'params': test_params}
            )

        self.iteration_count += 1

        return result

    def _generate_test_params(self, iteration: int) -> Dict[str, Any]:
        """Generate test parameters for iteration"""
        test_matrix = self.optimization_config.get('test_matrix', {})

        # Get parameter ranges
        llm_utils = test_matrix.get('gpu_memory_utilization', {}).get('llm', [0.75])
        stt_utils = test_matrix.get('gpu_memory_utilization', {}).get('stt', [0.30])
        batch_sizes = test_matrix.get('batch_sizes', [1, 2, 4, 8])
        seq_lengths = test_matrix.get('sequence_lengths', [128, 256, 512])
        concurrency_levels = test_matrix.get('concurrent_requests', [1, 2, 4])

        # Simple grid search (in production, use Bayesian optimization)
        params = {
            'gpu_memory_utilization_llm': llm_utils[iteration % len(llm_utils)],
            'gpu_memory_utilization_stt': stt_utils[iteration % len(stt_utils)],
            'batch_size': batch_sizes[iteration % len(batch_sizes)],
            'sequence_length': seq_lengths[iteration % len(seq_lengths)],
            'concurrency': concurrency_levels[iteration % len(concurrency_levels)]
        }

        return params

    async def run_optimization(
        self,
        max_iterations: Optional[int] = None,
        executor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run full optimization process

        Args:
            max_iterations: Maximum iterations to run (uses config if None)
            executor: Service executor for testing

        Returns:
            Summary with best configuration found
        """
        # Create run ID
        self.run_id = f"opt_{self.profile_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.iteration_count = 0
        self.best_score = 0.0

        # Get iterations from config
        if max_iterations is None:
            max_iterations = self.optimization_config.get('algorithm', {}).get('iterations', 50)

        # Create run in database
        self.db.create_run(
            self.run_id,
            self.profile_id,
            {
                'max_iterations': max_iterations,
                'optimization_config': self.optimization_config
            }
        )

        structured_logger.info(
            f"🚀 Starting optimization run '{self.run_id}'",
            metadata={
                'profile': self.profile_id,
                'iterations': max_iterations
            }
        )

        start_time = time.time()

        # Run optimization iterations
        for i in range(max_iterations):
            try:
                # Generate test parameters
                test_params = self._generate_test_params(i)

                # Run iteration
                result = await self.run_optimization_iteration(test_params, executor)

                structured_logger.info(
                    f"✓ Iteration {i+1}/{max_iterations} complete",
                    metadata={
                        'quality_score': result.quality_score,
                        'avg_latency_ms': result.avg_latency_ms
                    }
                )

                # Pause between iterations
                await asyncio.sleep(1)

            except Exception as e:
                structured_logger.error(
                    f"❌ Iteration {i+1} failed",
                    exception=e
                )

        # Complete run
        self.db.complete_run(self.run_id, self.best_score)

        duration_s = time.time() - start_time

        # Get summary
        summary = self.db.get_run_summary(self.run_id)
        best_results = self.db.get_best_results(self.run_id, limit=5)

        structured_logger.info(
            f"✅ Optimization complete",
            metadata={
                'duration_s': duration_s,
                'iterations': max_iterations,
                'best_score': self.best_score
            }
        )

        return {
            'run_id': self.run_id,
            'profile_id': self.profile_id,
            'duration_seconds': duration_s,
            'total_iterations': max_iterations,
            'best_score': self.best_score,
            'summary': summary,
            'top_5_results': best_results
        }

    def get_best_configuration(self, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get best configuration from optimization run"""
        if run_id is None:
            run_id = self.run_id

        if not run_id:
            return None

        best_results = self.db.get_best_results(run_id, limit=1)

        if not best_results:
            return None

        best = best_results[0]

        return {
            'quality_score': best['quality_score'],
            'parameters': {
                'gpu_memory_utilization_llm': best['gpu_memory_utilization_llm'],
                'gpu_memory_utilization_stt': best['gpu_memory_utilization_stt'],
                'batch_size': best['batch_size'],
                'sequence_length': best['sequence_length'],
                'concurrency': best['concurrency']
            },
            'metrics': {
                'avg_latency_ms': best['avg_latency_ms'],
                'p95_latency_ms': best['p95_latency_ms'],
                'p99_latency_ms': best['p99_latency_ms'],
                'throughput_rps': best['throughput_rps'],
                'success_rate': best['success_rate'],
                'gpu_memory_mb': best['gpu_memory_used_mb']
            }
        }
