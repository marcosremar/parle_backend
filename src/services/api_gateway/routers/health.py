"""
Health and metrics endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import os
import psutil
import httpx
from datetime import datetime

from schemas.responses import HealthResponse

router = APIRouter(tags=["health"])


async def _get_gpu_status_from_service_manager() -> Dict[str, Any]:
    """
    Get GPU status from Service Manager GPU endpoint
    Returns GPU info or None if unavailable
    """
    try:
        service_manager_url = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{service_manager_url}/gpu/status")
            if response.status_code == 200:
                data = response.json()
                return data.get("gpu_info", {})
    except (httpx.RequestError, httpx.TimeoutException, ValueError):
        # Failed to retrieve GPU status from service manager
        pass
    return {}


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    Returns system status and service availability
    """
    try:
        # Get GPU status from Service Manager (proper architecture)
        gpu_info = await _get_gpu_status_from_service_manager()
        gpu_available = bool(gpu_info)

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        services = {
            "ultravox": True,  # TODO: Check actual Ultravox status
            "tts": True,       # TODO: Check TTS availability
            "stt": True,       # TODO: Check STT availability
            "gpu": gpu_available
        }

        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "gpu": gpu_info
        }

        return HealthResponse(
            status="healthy",
            services=services,
            metrics=metrics
        )

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Detailed metrics endpoint
    Returns performance and resource metrics
    """
    try:
        cpu_times = psutil.cpu_times()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu": {
                    "percent": psutil.cpu_percent(interval=0.1),
                    "count": psutil.cpu_count(),
                    "user_time": cpu_times.user,
                    "system_time": cpu_times.system
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent": memory.percent,
                    "used_gb": memory.used / (1024**3)
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent
                }
            }
        }

        # Add GPU metrics from Service Manager
        gpu_info = await _get_gpu_status_from_service_manager()
        if gpu_info:
            metrics["gpu"] = {
                "available": True,
                "device_name": gpu_info.get("device_name", "Unknown"),
                "total_mb": gpu_info.get("total_mb", 0),
                "used_mb": gpu_info.get("used_mb", 0),
                "free_mb": gpu_info.get("free_mb", 0)
            }
        else:
            metrics["gpu"] = {"available": False}

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))