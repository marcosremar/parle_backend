"""
GPU Profile Management Router
REST API endpoints for GPU profile operations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import ValidationError, BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import os
import time

from src.core.managers.gpu_profile_manager import get_gpu_profile_manager
from src.core.managers.profile_warmup_manager import get_profile_warmup_manager
from src.core.managers.profile_optimizer import ProfileOptimizer
from src.core.auto_optimizer import AutoOptimizer
from loguru import logger
from src.core.exceptions import UltravoxError, wrap_exception

router = APIRouter(prefix="/profiles", tags=["profiles"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ProfileActivateRequest(BaseModel):
    """Request model for profile activate"""
    profile_id: str
    backup: bool = True


class ProfileActivateResponse(BaseModel):
    """Response model for profile activate"""
    success: bool
    profile_id: str
    previous_profile: str
    message: str


class ProfileListResponse(BaseModel):
    """Response model for profile list"""
    profiles: List[Dict[str, Any]]
    active_profile: str


class ProfileSummaryResponse(BaseModel):
    """Response model for profile summary"""
    profile_id: str
    name: str
    description: str
    use_case: str
    active: bool
    services: Dict[str, Any]
    resources: Dict[str, Any]
    performance_targets: Dict[str, Any]
    warmup: Dict[str, Any]


class OptimizationStartRequest(BaseModel):
    """Request model for optimization start"""
    profile_id: Optional[str] = None
    max_iterations: int = 50
    use_bayesian: bool = True
    exploration_factor: float = 0.2


class OptimizationStartResponse(BaseModel):
    """Response model for optimization start"""
    run_id: str
    profile_id: str
    status: str
    message: str


class OptimizationStatusResponse(BaseModel):
    """Response model for optimization status"""
    run_id: str
    status: str
    current_iteration: int
    total_iterations: int
    best_score: float
    elapsed_seconds: float


class OptimizationResultsResponse(BaseModel):
    """Response model for optimization results"""
    run_id: str
    profile_id: str
    best_configuration: Dict[str, Any]
    summary: Dict[str, Any]
    top_results: List[Dict[str, Any]]


class BenchmarkRequest(BaseModel):
    """Request model for benchmark"""
    test_iterations: int = 20
    test_scenarios: List[str] = ["single_request", "sustained_load"]
    sequence_length: int = 512
    concurrent_requests: int = 4
    profiles_to_test: Optional[List[str]] = None  # None = test all
    test_vllm_utilizations: Optional[List[float]] = None  # Test different vLLM GPU utilizations (e.g., [0.7, 0.75, 0.8])


class ProfileBenchmarkResult(BaseModel):
    """Pydantic model for profile benchmark result"""
    profile_id: str
    vllm_gpu_utilization: Optional[float] = None  # vLLM GPU memory utilization tested
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tokens_per_second: Optional[float] = None
    requests_per_second: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_memory_available_mb: Optional[float] = None
    gpu_memory_allocated_profile_mb: Optional[int] = None  # From profile config
    gpu_utilization_percent: Optional[float] = None
    first_token_ms: Optional[float] = None
    success_rate_percent: float
    total_requests: int


class BenchmarkComparisonResponse(BaseModel):
    """Response model for benchmark comparison"""
    benchmark_id: str
    timestamp: str
    profiles_tested: List[str]
    results: Dict[str, ProfileBenchmarkResult]
    winner: Dict[str, str]  # category -> profile_id
    recommendations: List[str]
    restored_profile: str


# ============================================================================
# Profile Management Endpoints
# ============================================================================

@router.get("/list", response_model=ProfileListResponse)
async def list_profiles() -> ProfileListResponse:
    """
    List all available GPU profiles

    Returns:
        List of profiles with details and active status
    """
    try:
        manager = get_gpu_profile_manager()
        profiles = manager.list_profiles()

        logger.info("📋 Listed all profiles", metadata={'count': len(profiles)})

        return ProfileListResponse(
            profiles=profiles,
            active_profile=manager.active_profile
        )

    except Exception as e:
        logger.error("❌ Failed to list profiles", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current", response_model=ProfileSummaryResponse)
async def get_current_profile() -> ProfileSummaryResponse:
    """
    Get currently active profile details

    Returns:
        Detailed summary of active profile
    """
    try:
        manager = get_gpu_profile_manager()
        summary = manager.get_profile_summary()

        if not summary:
            raise HTTPException(status_code=404, detail="No active profile found")

        logger.info(
            f"📊 Retrieved current profile: {summary['profile_id']}",
            metadata={'profile': summary['profile_id']}
        )

        return ProfileSummaryResponse(**summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Failed to get current profile", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{profile_id}", response_model=ProfileSummaryResponse)
async def get_profile(profile_id: str) -> ProfileSummaryResponse:
    """
    Get specific profile details

    Args:
        profile_id: Profile identifier

    Returns:
        Detailed profile summary
    """
    try:
        manager = get_gpu_profile_manager()
        summary = manager.get_profile_summary(profile_id)

        if not summary:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_id}' not found"
            )

        logger.info(
            f"📊 Retrieved profile: {profile_id}",
            metadata={'profile': profile_id}
        )

        return ProfileSummaryResponse(**summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get profile '{profile_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/activate", response_model=ProfileActivateResponse)
async def activate_profile(request: ProfileActivateRequest) -> ProfileActivateResponse:
    """
    Activate a GPU profile

    This will:
    1. Validate profile compatibility with hardware
    2. Backup current configuration (if requested)
    3. Switch to new profile
    4. Update configuration file

    Note: Services must be restarted for changes to take effect

    Args:
        request: Profile activation request

    Returns:
        Activation result
    """
    try:
        manager = get_gpu_profile_manager()

        # Get current profile
        previous_profile = manager.active_profile

        # Validate profile
        is_valid, error_msg = manager.validate_profile(request.profile_id)

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Profile validation failed: {error_msg}"
            )

        # Activate profile
        success = manager.activate_profile(
            request.profile_id,
            backup=request.backup
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to activate profile"
            )

        logger.info(
            f"✅ Activated profile: {request.profile_id}",
            metadata={
                'new_profile': request.profile_id,
                'previous_profile': previous_profile
            }
        )

        return ProfileActivateResponse(
            success=True,
            profile_id=request.profile_id,
            previous_profile=previous_profile,
            message=f"Profile '{request.profile_id}' activated successfully. "
                   f"Services must be restarted for changes to take effect."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to activate profile '{request.profile_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch")
async def switch_profile(request: ProfileActivateRequest) -> Dict[str, Any]:
    """
    Switch to a new GPU profile with automatic service restart

    This endpoint is designed for benchmarking and automated testing.
    It will:
    1. Validate the new profile
    2. Activate the new profile (updates configuration files)
    3. Restart LLM service (it will reload the new configuration)
    4. Wait for LLM to become healthy

    Args:
        request: Profile activation request with profile_id

    Returns:
        Success status with timing information
    """
    try:
        import aiohttp
        import asyncio
        import time

        start_time = time.time()
        manager = get_gpu_profile_manager()

        # Get current profile
        previous_profile = manager.active_profile

        # Validate profile
        is_valid, error_msg = manager.validate_profile(request.profile_id)

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Profile validation failed: {error_msg}"
            )

        # Activate profile (this updates the config files)
        success = manager.activate_profile(
            request.profile_id,
            backup=False  # No backup needed for benchmarking
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to activate profile"
            )

        logger.info(f"✅ Activated profile: {request.profile_id}")

        # Restart LLM service via Service Manager
        # The service will reload the updated configuration
        service_manager_url = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")

        # Restart LLM service (stop + start)
        async with aiohttp.ClientSession() as session:
            # POST /services/llm/restart returns a tracking_id
            async with session.post(f"{service_manager_url}/services/llm/restart") as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to restart LLM service: {error_text}"
                    )
                restart_data = await resp.json()
                tracking_id = restart_data.get('tracking_id')

        logger.info(f"🔄 LLM restart initiated (tracking_id: {tracking_id})")

        # Wait for restart to complete
        max_wait = 120  # 2 minutes
        waited = 0
        restart_completed = False

        while waited < max_wait:
            await asyncio.sleep(5)
            waited += 5

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{service_manager_url}/services/llm/restart/status/{tracking_id}") as resp:
                        if resp.status == 200:
                            status_data = await resp.json()
                            if status_data.get('completed'):
                                restart_completed = True
                                if not status_data.get('start_result', {}).get('success'):
                                    raise HTTPException(
                                        status_code=500,
                                        detail=f"LLM service restart failed: {status_data}"
                                    )
                                break
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Error checking restart status: {e}")

        if not restart_completed:
            raise HTTPException(
                status_code=500,
                detail=f"LLM service restart did not complete after {max_wait}s"
            )

        total_time = time.time() - start_time

        logger.info(
            f"✅ Profile switched successfully: {previous_profile} -> {request.profile_id}",
            metadata={
                'switch_time_seconds': total_time,
                'new_profile': request.profile_id,
                'previous_profile': previous_profile
            }
        )

        return {
            "success": True,
            "profile_id": request.profile_id,
            "previous_profile": previous_profile,
            "switch_time_seconds": round(total_time, 2),
            "message": f"Successfully switched to profile '{request.profile_id}' and restarted LLM service"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to switch profile", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/{profile_id}")
async def validate_profile(profile_id: str) -> Dict[str, Any]:
    """
    Validate if a profile can be activated

    Checks:
    - Profile exists
    - GPU availability (if required)
    - GPU memory sufficiency

    Args:
        profile_id: Profile to validate

    Returns:
        Validation result with details
    """
    try:
        manager = get_gpu_profile_manager()

        is_valid, error_msg = manager.validate_profile(profile_id)

        logger.info(
            f"🔍 Validated profile '{profile_id}': {'✅' if is_valid else '❌'}",
            metadata={
                'profile': profile_id,
                'valid': is_valid,
                'error': error_msg
            }
        )

        return {
            'profile_id': profile_id,
            'is_valid': is_valid,
            'error_message': error_msg,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Failed to validate profile '{profile_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Optimization Endpoints
# ============================================================================

# Global optimization state
_optimization_runs: Dict[str, Dict[str, Any]] = {}


@router.post("/optimize/start", response_model=OptimizationStartResponse)
async def start_optimization(
    request: OptimizationStartRequest,
    background_tasks: BackgroundTasks
):
    """
    Start optimization for a profile

    Runs performance tests with different parameter combinations to find
    the best configuration.

    Args:
        request: Optimization configuration
        background_tasks: FastAPI background task manager

    Returns:
        Optimization run details
    """
    try:
        # Get profile
        profile_id = request.profile_id or get_gpu_profile_manager().active_profile

        # Create optimizer
        if request.use_bayesian:
            optimizer = AutoOptimizer(
                profile_id=profile_id,
                exploration_factor=request.exploration_factor
            )
            run_id = f"auto_{profile_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            optimizer = ProfileOptimizer(profile_id=profile_id)
            run_id = f"opt_{profile_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store run state
        _optimization_runs[run_id] = {
            'optimizer': optimizer,
            'profile_id': profile_id,
            'status': 'running',
            'start_time': datetime.now(),
            'current_iteration': 0,
            'total_iterations': request.max_iterations,
            'best_score': 0.0
        }

        # Start optimization in background
        async def run_optimization() -> Any:
            try:
                if request.use_bayesian:
                    result = await optimizer.optimize(
                        max_iterations=request.max_iterations
                    )
                else:
                    result = await optimizer.run_optimization(
                        max_iterations=request.max_iterations
                    )

                _optimization_runs[run_id]['status'] = 'completed'
                _optimization_runs[run_id]['result'] = result

                logger.info(
                    f"✅ Optimization run '{run_id}' completed",
                    metadata={'best_score': result.get('best_score', 0)}
                )

            except Exception as e:
                _optimization_runs[run_id]['status'] = 'failed'
                _optimization_runs[run_id]['error'] = str(e)

                logger.error(f"❌ Optimization run '{run_id}' failed", exception=e)

        background_tasks.add_task(run_optimization)

        logger.info(
            f"🚀 Started optimization run '{run_id}'",
            metadata={
                'profile': profile_id,
                'iterations': request.max_iterations,
                'bayesian': request.use_bayesian
            }
        )

        return OptimizationStartResponse(
            run_id=run_id,
            profile_id=profile_id,
            status='running',
            message=f"Optimization started for profile '{profile_id}'"
        )

    except Exception as e:
        logger.error("❌ Failed to start optimization", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/status/{run_id}", response_model=OptimizationStatusResponse)
async def get_optimization_status(run_id: str) -> OptimizationStatusResponse:
    """
    Get optimization run status

    Args:
        run_id: Optimization run identifier

    Returns:
        Current status and progress
    """
    try:
        if run_id not in _optimization_runs:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization run '{run_id}' not found"
            )

        run_state = _optimization_runs[run_id]
        optimizer = run_state['optimizer']

        # Calculate elapsed time
        elapsed_seconds = (datetime.now() - run_state['start_time']).total_seconds()

        # Get current iteration from optimizer
        current_iteration = 0
        best_score = 0.0

        if hasattr(optimizer, 'state'):
            current_iteration = optimizer.state.iteration
            best_score = optimizer.state.best_score
        elif hasattr(optimizer, 'iteration_count'):
            current_iteration = optimizer.iteration_count
            best_score = optimizer.best_score

        return OptimizationStatusResponse(
            run_id=run_id,
            status=run_state['status'],
            current_iteration=current_iteration,
            total_iterations=run_state['total_iterations'],
            best_score=best_score,
            elapsed_seconds=elapsed_seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get optimization status for '{run_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/results/{run_id}", response_model=OptimizationResultsResponse)
async def get_optimization_results(run_id: str) -> OptimizationResultsResponse:
    """
    Get optimization results

    Args:
        run_id: Optimization run identifier

    Returns:
        Best configuration and detailed results
    """
    try:
        if run_id not in _optimization_runs:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization run '{run_id}' not found"
            )

        run_state = _optimization_runs[run_id]

        if run_state['status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Optimization run '{run_id}' is not completed (status: {run_state['status']})"
            )

        result = run_state.get('result', {})
        optimizer = run_state['optimizer']

        # Get best configuration
        best_config = optimizer.get_best_configuration(run_id) or {}

        # Get top results
        top_results = []
        if hasattr(optimizer, 'db'):
            top_results = optimizer.db.get_best_results(run_id, limit=10)

        logger.info(
            f"📊 Retrieved optimization results for '{run_id}'",
            metadata={'best_score': result.get('best_score', 0)}
        )

        return OptimizationResultsResponse(
            run_id=run_id,
            profile_id=run_state['profile_id'],
            best_configuration=best_config,
            summary=result,
            top_results=top_results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get optimization results for '{run_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/apply/{run_id}")
async def apply_optimization_results(run_id: str) -> Dict[str, Any]:
    """
    Apply optimization results to profile

    Updates the profile configuration with the best parameters found

    Args:
        run_id: Optimization run identifier

    Returns:
        Application result
    """
    try:
        if run_id not in _optimization_runs:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization run '{run_id}' not found"
            )

        run_state = _optimization_runs[run_id]
        optimizer = run_state['optimizer']

        # Apply best configuration
        if hasattr(optimizer, 'export_best_config_to_profile'):
            success = optimizer.export_best_config_to_profile()

            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to apply best configuration"
                )

            logger.info(
                f"✅ Applied optimization results from '{run_id}' to profile",
                metadata={'profile': run_state['profile_id']}
            )

            return {
                'success': True,
                'run_id': run_id,
                'profile_id': run_state['profile_id'],
                'message': 'Best configuration applied to profile. Services must be restarted.',
                'timestamp': datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Optimizer does not support config export"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to apply optimization results from '{run_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/list")
async def list_optimization_runs() -> Dict[str, Any]:
    """
    List all optimization runs

    Returns:
        List of optimization runs with status
    """
    try:
        runs = []

        for run_id, run_state in _optimization_runs.items():
            runs.append({
                'run_id': run_id,
                'profile_id': run_state['profile_id'],
                'status': run_state['status'],
                'start_time': run_state['start_time'].isoformat(),
                'total_iterations': run_state['total_iterations'],
                'best_score': run_state.get('best_score', 0.0)
            })

        logger.info(f"📋 Listed optimization runs", metadata={'count': len(runs)})

        return {
            'runs': runs,
            'total': len(runs)
        }

    except Exception as e:
        logger.error("❌ Failed to list optimization runs", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Benchmark Endpoints
# ============================================================================

@router.post("/benchmark/compare", response_model=BenchmarkComparisonResponse)
async def benchmark_profiles(request: BenchmarkRequest) -> BenchmarkComparisonResponse:
    """
    Run performance benchmark on all GPU profiles and compare results

    This endpoint will:
    1. Save current active profile
    2. Test each profile (full_gpu, hybrid, minimal_gpu)
    3. Collect LLM performance metrics for each
    4. Generate comparison report with recommendations
    5. Restore original profile

    Args:
        request: Benchmark configuration

    Returns:
        Comparison results with winner and recommendations
    """
    try:
        from src.core.service_executor import get_service_executor
        from src.core.performance_test_runner import PerformanceTestRunner, TestScenario
        import statistics

        manager = get_gpu_profile_manager()
        executor = get_service_executor()

        # Save current profile
        original_profile = manager.active_profile

        # Determine profiles to test
        profiles_to_test = request.profiles_to_test or ['full_gpu', 'hybrid', 'minimal_gpu']

        benchmark_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = {}

        logger.info(
            f"🏁 Starting profile benchmark '{benchmark_id}'",
            metadata={
                'profiles': profiles_to_test,
                'iterations': request.test_iterations
            }
        )

        # Test each profile
        for profile_id in profiles_to_test:
            # Validate profile exists
            is_valid, error_msg = manager.validate_profile(profile_id)
            if not is_valid:
                logger.warning(f"⚠️ Skipping invalid profile '{profile_id}': {error_msg}")
                continue

            # Get vLLM utilizations to test for this profile
            profile = manager.get_profile(profile_id)
            vllm_utilizations = request.test_vllm_utilizations

            # If not specified, use profile's default
            if not vllm_utilizations:
                llm_config = profile.services.get('llm')
                if llm_config and hasattr(llm_config, 'vllm_gpu_memory_utilization') and llm_config.vllm_gpu_memory_utilization:
                    vllm_utilizations = [llm_config.vllm_gpu_memory_utilization]
                else:
                    vllm_utilizations = [0.75]  # Default fallback

            # Test each vLLM utilization value
            for vllm_util in vllm_utilizations:
                test_key = f"{profile_id}_vllm{vllm_util:.2f}"
                logger.info(f"📊 Testing profile: {profile_id} with vLLM GPU utilization: {vllm_util:.2f}")

                # Activate profile
                manager.activate_profile(profile_id, backup=False)

                # Wait for services to stabilize (2 seconds)
                await asyncio.sleep(2)

                # Initialize test runner
                test_runner = PerformanceTestRunner(profile_id=profile_id)

                # Run LLM performance tests
                latencies = []
                first_token_latencies = []
                tokens_per_sec_list = []
                successes = 0

                for i in range(request.test_iterations):
                    try:
                        # Create LLM request via orchestrator
                        import aiohttp

                        start_time = time.time()

                        async with aiohttp.ClientSession() as session:
                            llm_generate_url = os.getenv('LLM_GENERATE_URL', 'http://localhost:8100/generate')
                            async with session.post(
                                llm_generate_url,
                                json={
                                    'prompt': f"Explain machine learning in {request.sequence_length // 4} words.",
                                    'max_tokens': request.sequence_length,
                                    'temperature': 0.7
                                },
                                timeout=aiohttp.ClientTimeout(total=30)
                            ) as resp:
                                if resp.status == 200:
                                    result = await resp.json()
                                    latency_ms = (time.time() - start_time) * 1000
                                    latencies.append(latency_ms)
                                    successes += 1

                                    # Extract metrics
                                    if 'first_token_latency_ms' in result:
                                        first_token_latencies.append(result['first_token_latency_ms'])
                                    if 'tokens_per_second' in result:
                                        tokens_per_sec_list.append(result['tokens_per_second'])
                                else:
                                    logger.warning(f"⚠️ Request failed with status {resp.status}")

                    except Exception as e:
                        logger.warning(f"⚠️ Test iteration {i+1} failed for {test_key}: {e}")
                        continue

                # Collect GPU metrics
                gpu_memory_used_mb = None
                gpu_memory_total_mb = None
                gpu_memory_available_mb = None
                gpu_utilization = None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used_mb = mem_info.used / (1024 * 1024)
                    gpu_memory_total_mb = mem_info.total / (1024 * 1024)
                    gpu_memory_available_mb = mem_info.free / (1024 * 1024)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = util.gpu
                    pynvml.nvmlShutdown()
                except Exception as e:
                    # GPU metrics unavailable (NVIDIA driver not installed, etc)
                    logger.debug(f"Failed to retrieve GPU metrics: {e}")

                # Get allocated GPU memory from profile config
                gpu_memory_allocated = 0
                if profile:
                    for service_name, service_config in profile.services.items():
                        if hasattr(service_config, 'gpu_memory_mb') and service_config.gpu_memory_mb:
                            gpu_memory_allocated += service_config.gpu_memory_mb

                # Calculate metrics
                if latencies:
                    sorted_latencies = sorted(latencies)
                    results[test_key] = ProfileBenchmarkResult(
                        profile_id=profile_id,
                        vllm_gpu_utilization=vllm_util,
                        avg_latency_ms=statistics.mean(latencies),
                        p50_latency_ms=sorted_latencies[len(sorted_latencies) // 2],
                        p95_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)],
                        p99_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)],
                        tokens_per_second=statistics.mean(tokens_per_sec_list) if tokens_per_sec_list else None,
                        requests_per_second=successes / (sum(latencies) / 1000) if latencies else 0,
                        gpu_memory_used_mb=gpu_memory_used_mb,
                        gpu_memory_total_mb=gpu_memory_total_mb,
                        gpu_memory_available_mb=gpu_memory_available_mb,
                        gpu_memory_allocated_profile_mb=gpu_memory_allocated,
                        gpu_utilization_percent=gpu_utilization,
                        first_token_ms=statistics.mean(first_token_latencies) if first_token_latencies else None,
                        success_rate_percent=(successes / request.test_iterations) * 100,
                        total_requests=request.test_iterations
                    )

                logger.info(f"✅ {test_key} benchmark complete")
            else:
                logger.warning(f"⚠️ No successful results for {profile_id}")

        # Restore original profile
        manager.activate_profile(original_profile, backup=False)

        # Determine winners
        winners = {}
        if results:
            # Best latency
            best_latency_profile = min(results.items(), key=lambda x: x[1].avg_latency_ms)
            winners['latency'] = best_latency_profile[0]

            # Best throughput
            best_throughput_profile = max(results.items(), key=lambda x: x[1].requests_per_second)
            winners['throughput'] = best_throughput_profile[0]

            # Best GPU efficiency (lowest memory usage)
            gpu_results = {k: v for k, v in results.items() if v.gpu_memory_used_mb is not None}
            if gpu_results:
                best_gpu_profile = min(gpu_results.items(), key=lambda x: x[1].gpu_memory_used_mb)
                winners['gpu_efficiency'] = best_gpu_profile[0]

            # Best GPU availability (most free memory based on profile allocation)
            available_results = {k: v for k, v in results.items() if v.gpu_memory_available_mb is not None}
            if available_results:
                best_available_profile = max(available_results.items(), key=lambda x: x[1].gpu_memory_available_mb)
                winners['most_gpu_available'] = best_available_profile[0]

        # Generate recommendations
        recommendations = []
        if results:
            for test_key, result in results.items():
                rec = f"{test_key}: "
                highlights = []

                if winners.get('latency') == test_key:
                    highlights.append(f"⚡ Best latency ({result.avg_latency_ms:.1f}ms avg)")
                if winners.get('throughput') == test_key:
                    highlights.append(f"🚀 Best throughput ({result.requests_per_second:.1f} req/s)")
                if winners.get('gpu_efficiency') == test_key:
                    highlights.append(f"💾 Lowest GPU usage ({result.gpu_memory_used_mb:.0f}MB used)")
                if winners.get('most_gpu_available') == test_key:
                    highlights.append(f"🎯 Most GPU available ({result.gpu_memory_available_mb:.0f}MB free)")

                if not highlights:
                    vllm_info = f"vLLM={result.vllm_gpu_utilization:.2f}" if result.vllm_gpu_utilization else ""
                    gpu_info = f"GPU: {result.gpu_memory_allocated_profile_mb}MB allocated, {result.gpu_memory_available_mb:.0f}MB free" if result.gpu_memory_available_mb else ""
                    highlights.append(f"Latency: {result.avg_latency_ms:.1f}ms, {vllm_info}, {gpu_info}")

                rec += ", ".join(highlights)
                recommendations.append(rec)

        logger.info(
            f"🏆 Benchmark '{benchmark_id}' complete",
            metadata={
                'winners': winners,
                'profiles_tested': list(results.keys())
            }
        )

        return BenchmarkComparisonResponse(
            benchmark_id=benchmark_id,
            timestamp=datetime.now().isoformat(),
            profiles_tested=list(results.keys()),
            results=results,
            winner=winners,
            recommendations=recommendations,
            restored_profile=original_profile
        )

    except Exception as e:
        logger.error("❌ Benchmark failed", exception=e)

        # Try to restore original profile
        try:
            manager = get_gpu_profile_manager()
            if 'original_profile' in locals():
                manager.activate_profile(original_profile, backup=False)
        except Exception as recovery_err:
            # Profile restoration failed, but original error will be raised
            logger.warning(f"Failed to restore original profile during error recovery: {recovery_err}")

        raise HTTPException(status_code=500, detail=str(e))
