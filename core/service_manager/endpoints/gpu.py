"""
GPU Management Endpoints

Endpoints for GPU memory monitoring, cleanup, and process management.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from src.core.logging import setup_logging

# Setup logging
logger = setup_logging("service-manager-gpu", level="INFO")

# Create router
router = APIRouter(prefix="/gpu", tags=["GPU Management"])


@router.get("/status")
async def get_gpu_status():
    """Get current GPU memory status"""
    try:
        from src.core.gpu_memory_manager import get_gpu_manager

        gpu_manager = get_gpu_manager()
        gpu_info = gpu_manager.get_gpu_info()

        if not gpu_info:
            raise HTTPException(status_code=503, detail="GPU not available")

        return {
            "success": True,
            "gpu_info": {
                "total_mb": gpu_info.total_mb,
                "used_mb": gpu_info.used_mb,
                "free_mb": gpu_info.free_mb,
                "device_name": gpu_info.device_name,
                "device_index": gpu_info.device_index
            },
            "processes": gpu_manager.get_gpu_processes(),
            "reserved_memory": gpu_manager.reserved_memory,
            "total_reserved_mb": sum(gpu_manager.reserved_memory.values())
        }
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_gpu(level: str = "soft"):
    """
    Clean GPU memory

    Args:
        level: Cleanup level - "soft" (cache only), "hard" (kill processes), "nuclear" (full cleanup)

    Returns:
        Cleanup results
    """
    try:
        from src.core.gpu_memory_manager import get_gpu_manager

        gpu_manager = get_gpu_manager()
        gpu_before = gpu_manager.get_gpu_info()

        logger.info(f"🧹 GPU cleanup requested (level: {level})")

        success = gpu_manager.cleanup_gpu_memory(level=level)

        gpu_after = gpu_manager.get_gpu_info()

        freed_mb = (gpu_after.free_mb - gpu_before.free_mb) if gpu_before and gpu_after else 0

        return {
            "success": success,
            "level": level,
            "freed_mb": freed_mb,
            "gpu_before": {
                "used_mb": gpu_before.used_mb,
                "free_mb": gpu_before.free_mb
            } if gpu_before else None,
            "gpu_after": {
                "used_mb": gpu_after.used_mb,
                "free_mb": gpu_after.free_mb
            } if gpu_after else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error cleaning GPU: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup/full")
async def full_gpu_cleanup():
    """
    Perform full GPU cleanup (all strategies)

    This will:
    1. Clean CUDA cache
    2. Kill all GPU service processes
    3. Kill all stale GPU processes
    4. Kill zombie GPU processes
    5. Deep CUDA cleanup (vLLM-optimized)
    """
    try:
        from src.core.gpu_memory_manager import get_gpu_manager

        gpu_manager = get_gpu_manager()
        gpu_before = gpu_manager.get_gpu_info()

        logger.info("🧹 Full GPU cleanup initiated...")

        success = gpu_manager.full_gpu_reset(include_zombies=True)

        gpu_after = gpu_manager.get_gpu_info()

        freed_mb = (gpu_after.free_mb - gpu_before.free_mb) if gpu_before and gpu_after else 0

        return {
            "success": success,
            "cleanup_type": "full",
            "freed_mb": freed_mb,
            "gpu_before": {
                "used_mb": gpu_before.used_mb,
                "free_mb": gpu_before.free_mb
            } if gpu_before else None,
            "gpu_after": {
                "used_mb": gpu_after.used_mb,
                "free_mb": gpu_after.free_mb
            } if gpu_after else None,
            "cleanup_history_count": len(gpu_manager.cleanup_history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in full GPU cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kill/zombies")
async def kill_zombie_gpu_processes():
    """Kill zombie GPU processes using fuser"""
    try:
        from src.core.gpu_memory_manager import get_gpu_manager

        gpu_manager = get_gpu_manager()

        logger.info("🧹 Killing zombie GPU processes...")

        killed_count = gpu_manager.kill_zombie_gpu_processes()

        return {
            "success": True,
            "killed_count": killed_count,
            "message": f"Killed {killed_count} zombie GPU processes",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error killing zombie processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
