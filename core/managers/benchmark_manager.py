"""
Benchmark Manager
Centralized audio pipeline benchmarking for GPU profile comparison
"""

import os
import asyncio
import logging
import time
import base64
import json
import statistics
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)


class BenchmarkStatus(Enum):
    """Benchmark execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AudioPipelineMetrics:
    """Metrics for a single audio pipeline execution"""
    iteration: int
    llm_latency_ms: float
    tts_latency_ms: float
    pipeline_latency_ms: float
    transcript: str = ""
    response_text: str = ""
    response_audio_size_bytes: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class ProfileBenchmarkResult:
    """Benchmark results for a single GPU profile"""
    profile_id: str
    vllm_gpu_utilization: float
    iterations: int
    successful_iterations: int

    # LLM metrics
    llm_avg_ms: float
    llm_p50_ms: float
    llm_p95_ms: float
    llm_p99_ms: float
    llm_min_ms: float
    llm_max_ms: float

    # TTS metrics
    tts_avg_ms: float
    tts_p50_ms: float
    tts_p95_ms: float
    tts_p99_ms: float
    tts_min_ms: float
    tts_max_ms: float

    # Pipeline metrics
    pipeline_avg_ms: float
    pipeline_p50_ms: float
    pipeline_p95_ms: float
    pipeline_p99_ms: float
    pipeline_min_ms: float
    pipeline_max_ms: float

    # GPU metrics
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None

    # Metadata
    warmup_time_seconds: float = 0.0
    test_duration_seconds: float = 0.0
    success_rate_percent: float = 100.0

    # Raw data
    raw_metrics: List[AudioPipelineMetrics] = field(default_factory=list)


@dataclass
class BenchmarkRun:
    """Complete benchmark run with multiple profiles"""
    run_id: str
    status: BenchmarkStatus
    profiles_tested: List[str]
    audio_file: str
    iterations_per_profile: int
    warmup_time_seconds: int
    start_time: datetime
    end_time: Optional[datetime] = None
    results: Dict[str, ProfileBenchmarkResult] = field(default_factory=dict)
    winner: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    error: Optional[str] = None


class BenchmarkManager:
    """
    Manages audio pipeline benchmarking across GPU profiles

    Features:
    - Audio-to-audio pipeline testing (LLM + TTS)
    - Multi-profile comparison
    - Automatic service restart and warmup
    - Statistical analysis (avg, P50, P95, P99)
    - Report generation (JSON, CSV, HTML, Markdown)
    """

    def __init__(self, service_manager_url: str = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")):
        """
        Initialize Benchmark Manager

        Args:
            service_manager_url: Service Manager API URL
        """
        self.service_manager_url = service_manager_url
        self.llm_url = "http://localhost:8100"
        self.tts_url = "http://localhost:8101"
        self.runs: Dict[str, BenchmarkRun] = {}

        logger.info(f"🔧 Benchmark Manager initialized")
        logger.info(f"   Service Manager: {service_manager_url}")
        logger.info(f"   LLM Service: {self.llm_url}")
        logger.info(f"   TTS Service: {self.tts_url}")

    async def run_audio_pipeline_benchmark(
        self,
        profiles: List[str],
        audio_file: str,
        iterations: int = 15,
        warmup_time: int = 45,
        auto_restart: bool = True
    ) -> str:
        """
        Run complete audio pipeline benchmark across multiple profiles

        Args:
            profiles: List of GPU profile IDs to test
            audio_file: Path to audio file for testing
            iterations: Number of iterations per profile
            warmup_time: Warmup time in seconds after service restart
            auto_restart: Automatically restart services with new profiles

        Returns:
            run_id: Benchmark run identifier
        """
        run_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Create benchmark run
        run = BenchmarkRun(
            run_id=run_id,
            status=BenchmarkStatus.RUNNING,
            profiles_tested=profiles,
            audio_file=audio_file,
            iterations_per_profile=iterations,
            warmup_time_seconds=warmup_time,
            start_time=datetime.now()
        )

        self.runs[run_id] = run

        logger.info(f"🚀 Starting benchmark run: {run_id}")
        logger.info(f"   Profiles: {profiles}")
        logger.info(f"   Audio file: {audio_file}")
        logger.info(f"   Iterations: {iterations}")
        logger.info(f"   Warmup: {warmup_time}s")

        try:
            # Load audio file
            audio_base64 = self._load_audio_file(audio_file)

            # Test each profile
            for profile_id in profiles:
                logger.info(f"📊 Testing profile: {profile_id}")

                # Restart services with new profile (if auto_restart enabled)
                if auto_restart:
                    restart_success = await self._restart_with_profile(profile_id)
                    if not restart_success:
                        logger.error(f"❌ Failed to restart with profile: {profile_id}")
                        run.results[profile_id] = self._create_failed_result(profile_id, "Service restart failed")
                        continue

                    # Wait for warmup
                    logger.info(f"⏳ Warming up for {warmup_time}s...")
                    await asyncio.sleep(warmup_time)

                # Run benchmark iterations
                result = await self._benchmark_profile(
                    profile_id=profile_id,
                    audio_base64=audio_base64,
                    iterations=iterations,
                    warmup_time=warmup_time
                )

                run.results[profile_id] = result
                logger.info(f"✅ Completed profile: {profile_id}")
                logger.info(f"   LLM avg: {result.llm_avg_ms:.2f}ms")
                logger.info(f"   TTS avg: {result.tts_avg_ms:.2f}ms")
                logger.info(f"   Pipeline avg: {result.pipeline_avg_ms:.2f}ms")

            # Analyze results and determine winner
            run.winner = self._determine_winner(run.results)
            run.recommendations = self._generate_recommendations(run.results)

            run.status = BenchmarkStatus.COMPLETED
            run.end_time = datetime.now()

            logger.info(f"🏆 Benchmark complete! Winner: {run.winner}")

        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            run.status = BenchmarkStatus.FAILED
            run.error = str(e)
            run.end_time = datetime.now()

        return run_id

    async def _benchmark_profile(
        self,
        profile_id: str,
        audio_base64: str,
        iterations: int,
        warmup_time: int
    ) -> ProfileBenchmarkResult:
        """
        Benchmark a single GPU profile

        Args:
            profile_id: GPU profile ID
            audio_base64: Base64-encoded audio data
            iterations: Number of test iterations
            warmup_time: Warmup time used (for reporting)

        Returns:
            ProfileBenchmarkResult
        """
        import aiohttp

        test_start = time.time()
        metrics_list: List[AudioPipelineMetrics] = []

        async with aiohttp.ClientSession() as session:
            for i in range(1, iterations + 1):
                try:
                    # Measure pipeline latency
                    pipeline_start = time.time()

                    # Step 1: Audio → Ultravox LLM
                    llm_start = time.time()
                    llm_response = await self._call_llm_audio(session, audio_base64)
                    llm_end = time.time()
                    llm_latency_ms = (llm_end - llm_start) * 1000

                    # Extract text response
                    response_text = self._extract_llm_text(llm_response)
                    transcript = self._extract_transcript(llm_response)

                    # Step 2: Text → Kokoro TTS
                    tts_start = time.time()
                    tts_response = await self._call_tts(session, response_text)
                    tts_end = time.time()
                    tts_latency_ms = (tts_end - tts_start) * 1000

                    # Extract audio size
                    audio_size = self._extract_audio_size(tts_response)

                    pipeline_end = time.time()
                    pipeline_latency_ms = (pipeline_end - pipeline_start) * 1000

                    # Store metrics
                    metrics = AudioPipelineMetrics(
                        iteration=i,
                        llm_latency_ms=llm_latency_ms,
                        tts_latency_ms=tts_latency_ms,
                        pipeline_latency_ms=pipeline_latency_ms,
                        transcript=transcript,
                        response_text=response_text,
                        response_audio_size_bytes=audio_size,
                        success=True
                    )

                    metrics_list.append(metrics)

                    logger.debug(f"   Iteration {i}/{iterations}: Pipeline={pipeline_latency_ms:.0f}ms (LLM={llm_latency_ms:.0f}ms + TTS={tts_latency_ms:.0f}ms)")

                except Exception as e:
                    logger.warning(f"⚠️  Iteration {i} failed: {e}")
                    metrics = AudioPipelineMetrics(
                        iteration=i,
                        llm_latency_ms=0,
                        tts_latency_ms=0,
                        pipeline_latency_ms=0,
                        success=False,
                        error=str(e)
                    )
                    metrics_list.append(metrics)

        test_end = time.time()
        test_duration = test_end - test_start

        # Calculate statistics
        return self._calculate_statistics(
            profile_id=profile_id,
            metrics_list=metrics_list,
            warmup_time=warmup_time,
            test_duration=test_duration
        )

    async def _call_llm_audio(self, session: 'aiohttp.ClientSession', audio_base64: str) -> Dict[str, Any]:
        """Call LLM service with audio input"""
        payload = {
            "audio_data": audio_base64,
            "sample_rate": 16000,
            "max_tokens": 100
        }

        async with session.post(
            f"{self.llm_url}/generate/audio",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            return await response.json()

    async def _call_tts(self, session: 'aiohttp.ClientSession', text: str) -> Dict[str, Any]:
        """Call TTS service"""
        payload = {
            "text": text,
            "voice": "af_sky",
            "speed": 1.0
        }

        async with session.post(
            f"{self.tts_url}/synthesize",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            return await response.json()

    async def _restart_with_profile(self, profile_id: str) -> bool:
        """Restart services with new GPU profile"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                # Call service manager profile switch endpoint
                async with session.post(
                    f"{self.service_manager_url}/profiles/switch",
                    json={"profile_id": profile_id, "auto_restart": True},
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Successfully switched to profile: {profile_id}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Profile switch failed: {error_text}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error switching profile: {e}")
            return False

    def _load_audio_file(self, audio_file: str) -> str:
        """Load audio file and convert to base64"""
        audio_path = Path(audio_file)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        logger.info(f"📁 Loaded audio file: {audio_file} ({len(audio_bytes)} bytes)")

        return audio_base64

    def _extract_llm_text(self, llm_response: Dict[str, Any]) -> str:
        """Extract text from LLM response (handles nested structure)"""
        text_field = llm_response.get('text', '')

        if isinstance(text_field, dict):
            return text_field.get('text', '')

        return text_field

    def _extract_transcript(self, llm_response: Dict[str, Any]) -> str:
        """Extract transcript from LLM response"""
        # Ultravox might include transcript in metadata
        text_field = llm_response.get('text', '')

        if isinstance(text_field, dict):
            metadata = text_field.get('metadata', {})
            return metadata.get('transcript', '')

        return ""

    def _extract_audio_size(self, tts_response: Dict[str, Any]) -> int:
        """Extract audio size from TTS response"""
        audio_data = tts_response.get('audio', '')

        if isinstance(audio_data, str):
            # Base64 encoded
            return len(audio_data) * 3 // 4  # Approximate decoded size

        return 0

    def _calculate_statistics(
        self,
        profile_id: str,
        metrics_list: List[AudioPipelineMetrics],
        warmup_time: float,
        test_duration: float
    ) -> ProfileBenchmarkResult:
        """Calculate benchmark statistics from metrics"""
        # Filter successful iterations
        successful = [m for m in metrics_list if m.success]

        if not successful:
            return self._create_failed_result(profile_id, "All iterations failed")

        # Extract latencies
        llm_latencies = [m.llm_latency_ms for m in successful]
        tts_latencies = [m.tts_latency_ms for m in successful]
        pipeline_latencies = [m.pipeline_latency_ms for m in successful]

        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            return statistics.quantiles(data, n=100)[int(p) - 1] if len(data) > 1 else data[0]

        # Get GPU utilization from profile ID (e.g., "speed_85" -> 0.85)
        vllm_util = 0.0
        if 'speed_' in profile_id:
            try:
                gpu_percent = int(profile_id.split('_')[-1])
                vllm_util = gpu_percent / 100.0
            except Exception as e:
                pass

        return ProfileBenchmarkResult(
            profile_id=profile_id,
            vllm_gpu_utilization=vllm_util,
            iterations=len(metrics_list),
            successful_iterations=len(successful),

            # LLM metrics
            llm_avg_ms=statistics.mean(llm_latencies),
            llm_p50_ms=percentile(llm_latencies, 50),
            llm_p95_ms=percentile(llm_latencies, 95),
            llm_p99_ms=percentile(llm_latencies, 99),
            llm_min_ms=min(llm_latencies),
            llm_max_ms=max(llm_latencies),

            # TTS metrics
            tts_avg_ms=statistics.mean(tts_latencies),
            tts_p50_ms=percentile(tts_latencies, 50),
            tts_p95_ms=percentile(tts_latencies, 95),
            tts_p99_ms=percentile(tts_latencies, 99),
            tts_min_ms=min(tts_latencies),
            tts_max_ms=max(tts_latencies),

            # Pipeline metrics
            pipeline_avg_ms=statistics.mean(pipeline_latencies),
            pipeline_p50_ms=percentile(pipeline_latencies, 50),
            pipeline_p95_ms=percentile(pipeline_latencies, 95),
            pipeline_p99_ms=percentile(pipeline_latencies, 99),
            pipeline_min_ms=min(pipeline_latencies),
            pipeline_max_ms=max(pipeline_latencies),

            # Metadata
            warmup_time_seconds=warmup_time,
            test_duration_seconds=test_duration,
            success_rate_percent=(len(successful) / len(metrics_list)) * 100,

            # Raw data
            raw_metrics=metrics_list
        )

    def _create_failed_result(self, profile_id: str, error: str) -> ProfileBenchmarkResult:
        """Create a failed benchmark result"""
        return ProfileBenchmarkResult(
            profile_id=profile_id,
            vllm_gpu_utilization=0.0,
            iterations=0,
            successful_iterations=0,
            llm_avg_ms=0, llm_p50_ms=0, llm_p95_ms=0, llm_p99_ms=0, llm_min_ms=0, llm_max_ms=0,
            tts_avg_ms=0, tts_p50_ms=0, tts_p95_ms=0, tts_p99_ms=0, tts_min_ms=0, tts_max_ms=0,
            pipeline_avg_ms=0, pipeline_p50_ms=0, pipeline_p95_ms=0, pipeline_p99_ms=0, pipeline_min_ms=0, pipeline_max_ms=0,
            success_rate_percent=0.0
        )

    def _determine_winner(self, results: Dict[str, ProfileBenchmarkResult]) -> Optional[str]:
        """Determine the best performing profile"""
        if not results:
            return None

        # Filter successful results
        successful = {k: v for k, v in results.items() if v.success_rate_percent > 80}

        if not successful:
            return None

        # Winner = lowest average pipeline latency
        winner = min(successful.items(), key=lambda x: x[1].pipeline_avg_ms)

        return winner[0]

    def _generate_recommendations(self, results: Dict[str, ProfileBenchmarkResult]) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []

        if not results:
            return ["No results to analyze"]

        # Filter successful results
        successful = {k: v for k, v in results.items() if v.success_rate_percent > 80}

        if not successful:
            return ["All profiles failed - check service logs"]

        # Find best profile
        best_profile = min(successful.items(), key=lambda x: x[1].pipeline_avg_ms)
        best_id, best_result = best_profile

        recommendations.append(
            f"Use {best_id} for production ({best_result.pipeline_avg_ms:.2f}ms average pipeline latency)"
        )

        # Compare with other profiles
        for profile_id, result in successful.items():
            if profile_id != best_id:
                speedup = ((result.pipeline_avg_ms - best_result.pipeline_avg_ms) / result.pipeline_avg_ms) * 100
                if speedup > 5:
                    recommendations.append(
                        f"{best_id} is {speedup:.1f}% faster than {profile_id}"
                    )

        # Check for failures
        failed = {k: v for k, v in results.items() if v.success_rate_percent < 80}
        for profile_id in failed:
            recommendations.append(f"Avoid {profile_id} - high failure rate")

        return recommendations

    def get_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """Get benchmark run by ID"""
        return self.runs.get(run_id)

    def export_results_json(self, run_id: str, output_file: str) -> Any:
        """Export results to JSON"""
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        # Convert to dict
        data = asdict(run)

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"📄 Exported JSON results: {output_file}")

    def export_results_csv(self, run_id: str, output_file: str) -> Any:
        """Export results to CSV"""
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Profile', 'GPU%', 'Iterations', 'Success%',
                'LLM Avg (ms)', 'LLM P95 (ms)',
                'TTS Avg (ms)', 'TTS P95 (ms)',
                'Pipeline Avg (ms)', 'Pipeline P95 (ms)'
            ])

            # Data rows
            for profile_id, result in run.results.items():
                writer.writerow([
                    profile_id,
                    f"{result.vllm_gpu_utilization * 100:.0f}%",
                    result.iterations,
                    f"{result.success_rate_percent:.1f}%",
                    f"{result.llm_avg_ms:.2f}",
                    f"{result.llm_p95_ms:.2f}",
                    f"{result.tts_avg_ms:.2f}",
                    f"{result.tts_p95_ms:.2f}",
                    f"{result.pipeline_avg_ms:.2f}",
                    f"{result.pipeline_p95_ms:.2f}"
                ])

        logger.info(f"📊 Exported CSV results: {output_file}")


# Singleton instance
_benchmark_manager: Optional[BenchmarkManager] = None


def get_benchmark_manager() -> BenchmarkManager:
    """Get global benchmark manager instance"""
    global _benchmark_manager

    if _benchmark_manager is None:
        _benchmark_manager = BenchmarkManager()

    return _benchmark_manager
