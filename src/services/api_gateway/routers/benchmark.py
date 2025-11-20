"""
Benchmark API Router
REST endpoints for audio pipeline benchmarking
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import ValidationError, BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

from src.core.benchmark_manager import get_benchmark_manager, BenchmarkStatus
from loguru import logger
from src.core.exceptions import UltravoxError, wrap_exception

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AudioPipelineBenchmarkRequest(BaseModel):
    """Request to start audio pipeline benchmark"""
    profiles: List[str]
    audio_file: Optional[str] = None  # If None, uses default test audio
    iterations: int = 15
    warmup_time: int = 45
    auto_restart: bool = True


class BenchmarkStartResponse(BaseModel):
    """Response when benchmark is started"""
    run_id: str
    status: str
    profiles: List[str]
    iterations: int
    estimated_duration_seconds: int
    message: str


class BenchmarkStatusResponse(BaseModel):
    """Current status of a benchmark run"""
    run_id: str
    status: str
    profiles_tested: List[str]
    start_time: str
    end_time: Optional[str] = None
    current_profile: Optional[str] = None
    progress_percent: float
    error: Optional[str] = None


class ProfileResultSummary(BaseModel):
    """Summary of results for a single profile"""
    profile_id: str
    vllm_gpu_utilization: float
    iterations: int
    successful_iterations: int
    success_rate_percent: float

    # Average latencies
    llm_avg_ms: float
    tts_avg_ms: float
    pipeline_avg_ms: float

    # P95 latencies
    llm_p95_ms: float
    tts_p95_ms: float
    pipeline_p95_ms: float


class BenchmarkResultsResponse(BaseModel):
    """Complete benchmark results"""
    run_id: str
    status: str
    profiles_tested: List[str]
    start_time: str
    end_time: Optional[str] = None
    test_duration_seconds: Optional[float] = None

    # Results per profile
    results: Dict[str, ProfileResultSummary]

    # Analysis
    winner: Optional[str] = None
    recommendations: List[str]

    # Metadata
    audio_file: str
    iterations_per_profile: int
    warmup_time_seconds: int


class BenchmarkListResponse(BaseModel):
    """List of benchmark runs"""
    runs: List[Dict[str, Any]]
    total: int


# ============================================================================
# Benchmark Endpoints
# ============================================================================

@router.post("/audio-pipeline", response_model=BenchmarkStartResponse)
async def start_audio_pipeline_benchmark(
    request: AudioPipelineBenchmarkRequest,
    background_tasks: BackgroundTasks
):
    """
    Start audio pipeline benchmark across multiple GPU profiles

    This will:
    1. Restart LLM service with each profile (if auto_restart=True)
    2. Wait for warmup period
    3. Run N iterations of audio→LLM→TTS pipeline
    4. Collect latency metrics (avg, P50, P95, P99)
    5. Compare profiles and determine winner

    Args:
        request: Benchmark configuration

    Returns:
        Benchmark run ID and estimated duration
    """
    try:
        logger.info("🚀 Starting audio pipeline benchmark", metadata={
            'profiles': request.profiles,
            'iterations': request.iterations,
            'warmup_time': request.warmup_time
        })

        # Validate profiles
        if not request.profiles:
            raise HTTPException(status_code=400, detail="At least one profile must be specified")

        if request.iterations < 1 or request.iterations > 100:
            raise HTTPException(status_code=400, detail="Iterations must be between 1 and 100")

        # Use default audio file if not provided
        audio_file = request.audio_file
        if not audio_file:
            # Get default audio file from project structure
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            default_audio = project_root / "data" / "audio" / "ultravox_informal_00000.wav"

            if default_audio.exists():
                audio_file = str(default_audio)
                logger.info(f"Using default audio file: {audio_file}")
            else:
                logger.warning(f"Default audio file not found at {default_audio}")
                audio_file = None

        # Estimate duration
        # Each profile: restart (110s) + warmup (45s) + iterations (15 * 2s) = ~185s
        estimated_duration = len(request.profiles) * (110 + request.warmup_time + (request.iterations * 2))

        # Start benchmark in background
        manager = get_benchmark_manager()

        # Run benchmark asynchronously
        run_id_future = asyncio.create_task(
            manager.run_audio_pipeline_benchmark(
                profiles=request.profiles,
                audio_file=audio_file,
                iterations=request.iterations,
                warmup_time=request.warmup_time,
                auto_restart=request.auto_restart
            )
        )

        # Wait a moment to get run_id
        await asyncio.sleep(0.1)
        run_id = await run_id_future

        logger.info(f"✅ Benchmark started: {run_id}", metadata={
            'estimated_duration': estimated_duration
        })

        return BenchmarkStartResponse(
            run_id=run_id,
            status="running",
            profiles=request.profiles,
            iterations=request.iterations,
            estimated_duration_seconds=estimated_duration,
            message=f"Benchmark started successfully. Estimated duration: {estimated_duration // 60}min {estimated_duration % 60}s"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Failed to start benchmark", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{run_id}", response_model=BenchmarkStatusResponse)
async def get_benchmark_status(run_id: str):
    """
    Get current status of a benchmark run

    Args:
        run_id: Benchmark run identifier

    Returns:
        Current status and progress
    """
    try:
        manager = get_benchmark_manager()
        run = manager.get_run(run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Benchmark run not found: {run_id}")

        # Calculate progress
        total_profiles = len(run.profiles_tested)
        completed_profiles = len(run.results)
        progress = (completed_profiles / total_profiles * 100) if total_profiles > 0 else 0

        # Determine current profile
        current_profile = None
        if run.status == BenchmarkStatus.RUNNING and completed_profiles < total_profiles:
            current_profile = run.profiles_tested[completed_profiles]

        return BenchmarkStatusResponse(
            run_id=run.run_id,
            status=run.status.value,
            profiles_tested=run.profiles_tested,
            start_time=run.start_time.isoformat(),
            end_time=run.end_time.isoformat() if run.end_time else None,
            current_profile=current_profile,
            progress_percent=progress,
            error=run.error
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get benchmark status", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{run_id}", response_model=BenchmarkResultsResponse)
async def get_benchmark_results(run_id: str):
    """
    Get complete benchmark results

    Args:
        run_id: Benchmark run identifier

    Returns:
        Complete results with all profiles and analysis
    """
    try:
        manager = get_benchmark_manager()
        run = manager.get_run(run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Benchmark run not found: {run_id}")

        # Convert results to summary format
        results_summary = {}
        for profile_id, result in run.results.items():
            results_summary[profile_id] = ProfileResultSummary(
                profile_id=result.profile_id,
                vllm_gpu_utilization=result.vllm_gpu_utilization,
                iterations=result.iterations,
                successful_iterations=result.successful_iterations,
                success_rate_percent=result.success_rate_percent,
                llm_avg_ms=result.llm_avg_ms,
                tts_avg_ms=result.tts_avg_ms,
                pipeline_avg_ms=result.pipeline_avg_ms,
                llm_p95_ms=result.llm_p95_ms,
                tts_p95_ms=result.tts_p95_ms,
                pipeline_p95_ms=result.pipeline_p95_ms
            )

        # Calculate test duration
        test_duration = None
        if run.end_time:
            test_duration = (run.end_time - run.start_time).total_seconds()

        return BenchmarkResultsResponse(
            run_id=run.run_id,
            status=run.status.value,
            profiles_tested=run.profiles_tested,
            start_time=run.start_time.isoformat(),
            end_time=run.end_time.isoformat() if run.end_time else None,
            test_duration_seconds=test_duration,
            results=results_summary,
            winner=run.winner,
            recommendations=run.recommendations,
            audio_file=run.audio_file,
            iterations_per_profile=run.iterations_per_profile,
            warmup_time_seconds=run.warmup_time_seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get benchmark results", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=BenchmarkListResponse)
async def list_benchmark_runs():
    """
    List all benchmark runs

    Returns:
        List of all benchmark runs with summary info
    """
    try:
        manager = get_benchmark_manager()

        runs_list = []
        for run_id, run in manager.runs.items():
            runs_list.append({
                'run_id': run.run_id,
                'status': run.status.value,
                'profiles_tested': run.profiles_tested,
                'start_time': run.start_time.isoformat(),
                'end_time': run.end_time.isoformat() if run.end_time else None,
                'winner': run.winner
            })

        # Sort by start time (newest first)
        runs_list.sort(key=lambda x: x['start_time'], reverse=True)

        return BenchmarkListResponse(
            runs=runs_list,
            total=len(runs_list)
        )

    except Exception as e:
        logger.error("❌ Failed to list benchmark runs", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/{run_id}/json")
async def export_results_json(run_id: str, output_file: str = None):
    """
    Export benchmark results to JSON

    Args:
        run_id: Benchmark run identifier
        output_file: Output file path (default: ~/.cache/ultravox-pipeline/benchmark_results.json)

    Returns:
        Export confirmation
    """
    if output_file is None:
        from pathlib import Path
        output_file = str(Path.home() / ".cache" / "ultravox-pipeline" / "benchmark_results.json")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    try:
        manager = get_benchmark_manager()
        manager.export_results_json(run_id, output_file)

        return {
            "success": True,
            "output_file": output_file,
            "message": f"Results exported to {output_file}"
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("❌ Failed to export results", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/{run_id}/csv")
async def export_results_csv(run_id: str, output_file: str = None):
    """
    Export benchmark results to CSV

    Args:
        run_id: Benchmark run identifier
        output_file: Output file path (default: ~/.cache/ultravox-pipeline/benchmark_results.csv)

    Returns:
        Export confirmation
    """
    if output_file is None:
        from pathlib import Path
        output_file = str(Path.home() / ".cache" / "ultravox-pipeline" / "benchmark_results.csv")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    try:
        manager = get_benchmark_manager()
        manager.export_results_csv(run_id, output_file)

        return {
            "success": True,
            "output_file": output_file,
            "message": f"Results exported to {output_file}"
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("❌ Failed to export results", exception=e)
        raise HTTPException(status_code=500, detail=str(e))
