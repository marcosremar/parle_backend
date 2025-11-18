"""
Pydantic Models for Service Manager Endpoints

Centralized model definitions to avoid circular imports.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ServiceRequest(BaseModel):
    """Request para iniciar/parar serviço"""
    force: bool = False
    timeout: int = 60


class VenvInstallRequest(BaseModel):
    """Request to install venv and dependencies"""
    requirements_file: Optional[str] = None
    install_deps: bool = True
    force: bool = False  # Force reinstall if venv already exists


class PackageInstallRequest(BaseModel):
    """Request to install specific packages"""
    packages: List[str]


class VenvStatusResponse(BaseModel):
    """Response with venv status"""
    service_id: str
    exists: bool
    venv_path: Optional[str] = None
    python_executable: Optional[str] = None
    installed_packages: Optional[List[str]] = None
    requirements_file: Optional[str] = None


class VenvListResponse(BaseModel):
    """Response listing all venvs"""
    venvs: List[Dict[str, Any]]
    total: int


class BulkOperationRequest(BaseModel):
    """Request for bulk service operations"""
    service_ids: Optional[List[str]] = None  # If None, applies to all services
    respect_dependencies: bool = True
    parallel: bool = False
    timeout_per_service: int = 60  # Increased from 30 to 60
    gpu_cleanup: bool = True  # Clean GPU memory before starting GPU services
    retry_on_failure: bool = True  # Retry failed services
    adjust_gpu_memory: bool = True  # Automatically adjust GPU memory utilization
    max_retries: int = 3  # Maximum retries per service


class BulkInstallRequest(BaseModel):
    """Request for bulk venv installation"""
    service_ids: Optional[List[str]] = None  # If None, install all services
    install_deps: bool = False
    force: bool = False


class ConfigValidationResponse(BaseModel):
    """Response with configuration validation results"""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
