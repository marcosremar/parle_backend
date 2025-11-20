"""
Test Scheduler Router
REST API endpoints for test scheduler management
"""

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError, BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.core.test_scheduler import get_test_scheduler
from loguru import logger
from src.core.exceptions import UltravoxError, wrap_exception

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SchedulerStartResponse(BaseModel):
    """Response model for scheduler start"""
    success: bool
    message: str
    scheduled_tests_count: int


class SchedulerStopResponse(BaseModel):
    """Response model for scheduler stop"""
    success: bool
    message: str


class AddScheduledTestRequest(BaseModel):
    """Request model for add scheduled test"""
    test_id: str
    profile_id: str
    cron_schedule: str
    max_iterations: int = 50
    use_bayesian: bool = True
    exploration_factor: float = 0.2
    low_traffic_hours: Optional[List[int]] = None


class AddScheduledTestResponse(BaseModel):
    """Response model for add scheduled test"""
    success: bool
    test_id: str
    next_run: Optional[str]
    message: str


class ScheduledTestResponse(BaseModel):
    """Response model for scheduled test"""
    test_id: str
    profile_id: str
    cron_schedule: str
    max_iterations: int
    use_bayesian: bool
    exploration_factor: float
    enabled: bool
    low_traffic_hours: Optional[List[int]]
    last_run: Optional[str]
    next_run: Optional[str]


class ScheduledTestsListResponse(BaseModel):
    """Response model for scheduled tests list"""
    tests: List[ScheduledTestResponse]
    total: int


class RunningTestResponse(BaseModel):
    """Response model for running test"""
    run_id: str
    test_id: str
    start_time: str
    status: str


class RunningTestsListResponse(BaseModel):
    """Response model for running tests list"""
    tests: List[RunningTestResponse]
    total: int


# ============================================================================
# Scheduler Control Endpoints
# ============================================================================

@router.post("/start", response_model=SchedulerStartResponse)
async def start_scheduler() -> SchedulerStartResponse:
    """
    Start the test scheduler

    Begins monitoring scheduled tests and executing them at configured times

    Returns:
        Scheduler start status
    """
    try:
        scheduler = get_test_scheduler()
        scheduler.start()

        logger.info("✅ Scheduler started via API")

        return SchedulerStartResponse(
            success=True,
            message="Scheduler started successfully",
            scheduled_tests_count=len(scheduler.scheduled_tests)
        )

    except Exception as e:
        logger.error("❌ Failed to start scheduler", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=SchedulerStopResponse)
async def stop_scheduler() -> SchedulerStopResponse:
    """
    Stop the test scheduler

    Stops monitoring and executing scheduled tests

    Returns:
        Scheduler stop status
    """
    try:
        scheduler = get_test_scheduler()
        await scheduler.stop()

        logger.info("🛑 Scheduler stopped via API")

        return SchedulerStopResponse(
            success=True,
            message="Scheduler stopped successfully"
        )

    except Exception as e:
        logger.error("❌ Failed to stop scheduler", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_scheduler_status() -> Dict[str, Any]:
    """
    Get scheduler status

    Returns:
        Current scheduler status and statistics
    """
    try:
        scheduler = get_test_scheduler()

        scheduled_tests = scheduler.get_scheduled_tests()
        running_tests = scheduler.get_running_tests()

        return {
            'is_running': scheduler.is_running,
            'scheduled_tests_count': len(scheduled_tests),
            'running_tests_count': len(running_tests),
            'scheduled_tests': scheduled_tests,
            'running_tests': running_tests,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("❌ Failed to get scheduler status", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Scheduled Test Management
# ============================================================================

@router.get("/tests", response_model=ScheduledTestsListResponse)
async def list_scheduled_tests() -> ScheduledTestsListResponse:
    """
    List all scheduled tests

    Returns:
        List of scheduled tests with details
    """
    try:
        scheduler = get_test_scheduler()
        tests = scheduler.get_scheduled_tests()

        test_responses = [
            ScheduledTestResponse(**test)
            for test in tests
        ]

        logger.info(f"📋 Listed {len(tests)} scheduled tests")

        return ScheduledTestsListResponse(
            tests=test_responses,
            total=len(test_responses)
        )

    except Exception as e:
        logger.error("❌ Failed to list scheduled tests", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests", response_model=AddScheduledTestResponse)
async def add_scheduled_test(request: AddScheduledTestRequest) -> AddScheduledTestResponse:
    """
    Add a new scheduled test

    Args:
        request: Scheduled test configuration

    Returns:
        Created test details
    """
    try:
        scheduler = get_test_scheduler()

        test = scheduler.add_scheduled_test(
            test_id=request.test_id,
            profile_id=request.profile_id,
            cron_schedule=request.cron_schedule,
            max_iterations=request.max_iterations,
            use_bayesian=request.use_bayesian,
            exploration_factor=request.exploration_factor,
            low_traffic_hours=request.low_traffic_hours
        )

        logger.info(
            f"📋 Added scheduled test: {request.test_id}",
            metadata={'profile': request.profile_id, 'cron': request.cron_schedule}
        )

        return AddScheduledTestResponse(
            success=True,
            test_id=test.test_id,
            next_run=test.next_run.isoformat() if test.next_run else None,
            message=f"Scheduled test '{request.test_id}' added successfully"
        )

    except Exception as e:
        logger.error(f"❌ Failed to add scheduled test '{request.test_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tests/{test_id}")
async def remove_scheduled_test(test_id: str) -> Dict[str, Any]:
    """
    Remove a scheduled test

    Args:
        test_id: Test identifier to remove

    Returns:
        Removal status
    """
    try:
        scheduler = get_test_scheduler()
        success = scheduler.remove_scheduled_test(test_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Scheduled test '{test_id}' not found"
            )

        logger.info(f"🗑️ Removed scheduled test: {test_id}")

        return {
            'success': True,
            'test_id': test_id,
            'message': f"Scheduled test '{test_id}' removed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to remove scheduled test '{test_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests/{test_id}/enable")
async def enable_scheduled_test(test_id: str) -> Dict[str, Any]:
    """
    Enable a scheduled test

    Args:
        test_id: Test identifier to enable

    Returns:
        Enable status
    """
    try:
        scheduler = get_test_scheduler()
        success = scheduler.enable_test(test_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Scheduled test '{test_id}' not found"
            )

        logger.info(f"✅ Enabled scheduled test: {test_id}")

        return {
            'success': True,
            'test_id': test_id,
            'enabled': True,
            'message': f"Scheduled test '{test_id}' enabled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to enable scheduled test '{test_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests/{test_id}/disable")
async def disable_scheduled_test(test_id: str) -> Dict[str, Any]:
    """
    Disable a scheduled test

    Args:
        test_id: Test identifier to disable

    Returns:
        Disable status
    """
    try:
        scheduler = get_test_scheduler()
        success = scheduler.disable_test(test_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Scheduled test '{test_id}' not found"
            )

        logger.info(f"⏸️ Disabled scheduled test: {test_id}")

        return {
            'success': True,
            'test_id': test_id,
            'enabled': False,
            'message': f"Scheduled test '{test_id}' disabled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to disable scheduled test '{test_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests/{test_id}/run-now")
async def run_test_now(test_id: str) -> Dict[str, Any]:
    """
    Run a scheduled test immediately (bypass schedule)

    Args:
        test_id: Test identifier to run

    Returns:
        Test execution result
    """
    try:
        scheduler = get_test_scheduler()

        logger.info(f"▶️ Manually triggering test: {test_id}")

        result = await scheduler.run_test_now(test_id)

        return {
            'success': True,
            'test_id': test_id,
            'message': f"Test '{test_id}' executed successfully",
            'result': result
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to run test '{test_id}'", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Running Tests
# ============================================================================

@router.get("/running", response_model=RunningTestsListResponse)
async def list_running_tests() -> RunningTestsListResponse:
    """
    List currently running tests

    Returns:
        List of running tests with status
    """
    try:
        scheduler = get_test_scheduler()
        tests = scheduler.get_running_tests()

        test_responses = [
            RunningTestResponse(**test)
            for test in tests
        ]

        logger.info(f"🏃 Listed {len(tests)} running tests")

        return RunningTestsListResponse(
            tests=test_responses,
            total=len(test_responses)
        )

    except Exception as e:
        logger.error("❌ Failed to list running tests", exception=e)
        raise HTTPException(status_code=500, detail=str(e))
