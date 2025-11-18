#!/usr/bin/env python3
"""
Service Discovery Endpoints
REST API for service registration and discovery
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from src.core.service_manager.registry.service_registry import get_registry, ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/discovery", tags=["discovery"])


# ============= REQUEST/RESPONSE MODELS =============

class EndpointInfo(BaseModel):
    """Endpoint information"""
    path: str
    method: str = "GET"
    description: str = ""


class ServiceRegistrationRequest(BaseModel):
    """Service registration request"""
    name: str
    host: str
    port: int
    endpoints: Optional[List[EndpointInfo]] = None
    protocols: Optional[List[str]] = ["http"]
    metadata: Optional[Dict[str, Any]] = None
    version: str = "1.0.0"


class HeartbeatRequest(BaseModel):
    """Heartbeat request"""
    name: str
    status: Optional[ServiceStatus] = None


class ServiceInfo(BaseModel):
    """Service information response"""
    service_id: str
    name: str
    host: str
    port: int
    status: str
    base_url: str
    protocols: List[str]
    endpoints: List[Dict[str, str]]
    metadata: Dict[str, Any]
    registered_at: str
    last_heartbeat: str
    uptime_seconds: float
    version: str


class DiscoveryResponse(BaseModel):
    """Service discovery response"""
    name: str
    base_url: str
    host: str
    port: int
    status: str
    protocols: List[str]
    endpoints: List[Dict[str, str]]
    metadata: Dict[str, Any]


class RegistryStatsResponse(BaseModel):
    """Registry statistics response"""
    total_services: int
    healthy: int
    degraded: int
    down: int
    services: List[str]


# ============= ENDPOINTS =============

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_service(request: ServiceRegistrationRequest) -> Dict[str, Any]:
    """
    Register a new service with the registry

    Args:
        request: Service registration information

    Returns:
        Registration confirmation with service_id
    """
    registry = get_registry()

    # Convert endpoints
    endpoints_dict = None
    if request.endpoints:
        endpoints_dict = [
            {"path": ep.path, "method": ep.method, "description": ep.description}
            for ep in request.endpoints
        ]

    registration = await registry.register(
        name=request.name,
        host=request.host,
        port=request.port,
        endpoints=endpoints_dict,
        protocols=request.protocols,
        metadata=request.metadata,
        version=request.version
    )

    logger.info(f"✅ Service registered via API: {request.name}")

    return {
        "success": True,
        "service_id": registration.service_id,
        "name": registration.name,
        "message": f"Service {request.name} registered successfully"
    }


@router.post("/heartbeat")
async def heartbeat(request: HeartbeatRequest) -> Dict[str, Any]:
    """
    Update service heartbeat

    Args:
        request: Heartbeat information

    Returns:
        Heartbeat acknowledgment
    """
    registry = get_registry()

    success = await registry.heartbeat(request.name, request.status)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {request.name} not registered"
        )

    return {
        "success": True,
        "name": request.name,
        "timestamp": registry.get_service(request.name).last_heartbeat.isoformat()
    }


@router.delete("/deregister/{service_name}")
async def deregister_service(service_name: str) -> Dict[str, Any]:
    """
    Deregister a service

    Args:
        service_name: Name of service to deregister

    Returns:
        Deregistration confirmation
    """
    registry = get_registry()

    success = await registry.deregister(service_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_name} not found"
        )

    logger.info(f"✅ Service deregistered via API: {service_name}")

    return {
        "success": True,
        "name": service_name,
        "message": f"Service {service_name} deregistered successfully"
    }


@router.get("/services", response_model=List[str])
async def list_services() -> List[str]:
    """
    List all registered service names

    Returns:
        List of service names
    """
    registry = get_registry()
    services = registry.get_all_services()
    return list(services.keys())


@router.get("/services/detailed", response_model=List[ServiceInfo])
async def list_services_detailed() -> List[ServiceInfo]:
    """
    List all registered services with full details

    Returns:
        List of service information
    """
    registry = get_registry()
    services = registry.get_all_services()

    result = []
    for name, service in services.items():
        service_dict = service.to_dict()
        result.append(ServiceInfo(
            service_id=service_dict['service_id'],
            name=service_dict['name'],
            host=service_dict['host'],
            port=service_dict['port'],
            status=service_dict['status'],
            base_url=service.get_base_url(),
            protocols=service_dict['protocols'],
            endpoints=[
                {"path": ep['path'], "method": ep['method'], "description": ep['description']}
                for ep in service_dict['endpoints']
            ],
            metadata=service_dict['metadata'],
            registered_at=service_dict['registered_at'],
            last_heartbeat=service_dict['last_heartbeat'],
            uptime_seconds=service_dict['uptime_seconds'],
            version=service_dict['version']
        ))

    return result


@router.get("/service/{service_name}")
async def get_service(service_name: str) -> ServiceInfo:
    """
    Get detailed information about a specific service

    Args:
        service_name: Service name

    Returns:
        Service information
    """
    registry = get_registry()
    service = registry.get_service(service_name)

    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_name} not found"
        )

    service_dict = service.to_dict()

    return ServiceInfo(
        service_id=service_dict['service_id'],
        name=service_dict['name'],
        host=service_dict['host'],
        port=service_dict['port'],
        status=service_dict['status'],
        base_url=service.get_base_url(),
        protocols=service_dict['protocols'],
        endpoints=[
            {"path": ep['path'], "method": ep['method'], "description": ep['description']}
            for ep in service_dict['endpoints']
        ],
        metadata=service_dict['metadata'],
        registered_at=service_dict['registered_at'],
        last_heartbeat=service_dict['last_heartbeat'],
        uptime_seconds=service_dict['uptime_seconds'],
        version=service_dict['version']
    )


@router.get("/lookup/{service_name}", response_model=DiscoveryResponse)
async def lookup_service(service_name: str) -> DiscoveryResponse:
    """
    Lookup service endpoint information (for service discovery)

    Args:
        service_name: Service name

    Returns:
        Service discovery information
    """
    registry = get_registry()
    discovery_info = registry.discover(service_name)

    if not discovery_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_name} not found"
        )

    return DiscoveryResponse(**discovery_info)


@router.get("/health/all")
async def health_all() -> Dict[str, Any]:
    """
    Get health status of all services

    Returns:
        Health status for all services
    """
    registry = get_registry()
    services = registry.get_all_services()

    health_status = {}
    for name, service in services.items():
        is_healthy = service.is_healthy()
        health_status[name] = {
            "status": service.status.value,
            "healthy": is_healthy,
            "last_heartbeat": service.last_heartbeat.isoformat(),
            "base_url": service.get_base_url()
        }

    return {
        "total": len(services),
        "healthy": len([s for s in services.values() if s.is_healthy()]),
        "services": health_status
    }


@router.get("/stats", response_model=RegistryStatsResponse)
async def get_stats() -> RegistryStatsResponse:
    """
    Get registry statistics

    Returns:
        Registry statistics
    """
    registry = get_registry()
    stats = registry.get_stats()
    return RegistryStatsResponse(**stats)


@router.put("/service/{service_name}/status")
async def update_service_status(service_name: str, new_status: ServiceStatus) -> Dict[str, Any]:
    """
    Update service status

    Args:
        service_name: Service name
        new_status: New service status

    Returns:
        Status update confirmation
    """
    registry = get_registry()

    success = await registry.update_status(service_name, new_status)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_name} not found"
        )

    return {
        "success": True,
        "name": service_name,
        "status": new_status.value,
        "message": f"Service {service_name} status updated to {new_status.value}"
    }
