"""
Service Manager Models - Pydantic models and Enums
"""

from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from enum import Enum


class ProcessStatus(Enum):
    """Status do processo do serviço"""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class HealthStatus(Enum):
    """Status de saúde do serviço"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ServiceConfig:
    """Configuração de um serviço"""

    def __init__(self, name: str, port: int, script: str, env: Dict = None):
        self.name = name
        self.port = port
        self.script = script
        self.env = env or {}


class ServiceRequest(BaseModel):
    """Requisição para operações em serviços"""
    service_id: str
    force: Optional[bool] = False
    timeout: Optional[int] = 30


class StartServiceRequest(BaseModel):
    """Request to start a service"""
    force: Optional[bool] = False


class VenvInstallRequest(BaseModel):
    """Requisição para instalação de venv"""
    service_id: str
    force: Optional[bool] = False


class PackageInstallRequest(BaseModel):
    """Requisição para instalação de pacote"""
    package: str


class VenvStatusResponse(BaseModel):
    """Resposta de status do venv"""
    service_id: str
    exists: bool
    path: Optional[str] = None
    requirements_file: Optional[str] = None
    packages: List[str] = []


class VenvListResponse(BaseModel):
    """Resposta de lista de venvs"""
    venvs: List[VenvStatusResponse]


class ConfigValidationResponse(BaseModel):
    """Response for configuration validation"""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    services: Dict[str, Any] = {}


class BulkOperationRequest(BaseModel):
    """Request for bulk operations on multiple services"""
    service_ids: List[str]
    force: Optional[bool] = False
    timeout: Optional[int] = 30


class BulkInstallRequest(BaseModel):
    """Request for bulk installation"""
    service_ids: List[str]
    force: Optional[bool] = False
    parallel: Optional[bool] = True
