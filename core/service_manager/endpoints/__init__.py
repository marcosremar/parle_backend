"""
Service Manager Endpoints
Modular endpoint organization for better maintainability
"""

from .services import router as services_router, set_manager
from .gpu import router as gpu_router
from .info import router as info_router
from .info import set_manager as set_info_manager
from .pipeline import router as pipeline_router
from .pipeline import set_manager as set_pipeline_manager
from .config import router as config_router
from .config import set_manager as set_config_manager
from .venv import create_venv_router

__all__ = [
    "services_router",
    "gpu_router",
    "info_router",
    "pipeline_router",
    "config_router",
    "create_venv_router",
    "set_manager",
    "set_info_manager",
    "set_pipeline_manager",
    "set_config_manager"
]
