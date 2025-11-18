"""
Info & System Endpoints

Service information and system resource monitoring.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import subprocess
import psutil
from src.core.core_logging import setup_logging
from src.core.service_manager.core import HealthStatus
from src.core.service_manager.models import ProcessStatus

# Setup logging
logger = setup_logging("endpoints-info", level="INFO")

# Create router
router = APIRouter(tags=["Info & System"])

# Manager instance will be set via dependency injection
_manager = None

def set_manager(manager):
    """Set the manager instance"""
    global _manager
    _manager = manager

def get_manager():
    """Get the manager instance"""
    return _manager


@router.get("/info")
async def service_info():
    """Service information"""
    return {
        "service": "service-manager",
        "version": "1.0.0",
        "endpoints": {
            "/services": "List all services",
            "/services/status": "Get detailed status of all services",
            "/services/validate-all": "Validate all service APIs",
            "/services/{service_id}": "Get specific service status",
            "/services/{service_id}/start": "Start a specific service",
            "/services/{service_id}/stop": "Stop a specific service",
            "/services/{service_id}/restart": "Restart a specific service",
            "/services/start-all": "Start all services",
            "/services/stop-all": "Stop all services",
            "/system": "Get system information",
            "/health": "Health check"
        }
    }


@router.get("/system")
async def get_system_info():
    """Obtém informações do sistema"""
    try:
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="Manager not initialized")

        # GPU info
        gpu_info = {}
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
                                   '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                gpu_info = {
                    'name': parts[0],
                    'memory_total_mb': int(parts[1].replace(' MiB', '')),
                    'memory_used_mb': int(parts[2].replace(' MiB', '')),
                    'memory_free_mb': int(parts[3].replace(' MiB', ''))
                }
        except Exception as e:
            logger.debug(f"Could not get GPU info (nvidia-smi): {e}")

        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "cores": psutil.cpu_count(),
                "usage_percent": psutil.cpu_percent(interval=1)
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "percent": psutil.virtual_memory().percent
            },
            "gpu": gpu_info,
            "services_status": manager.get_all_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
