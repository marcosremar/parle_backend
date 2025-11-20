"""
Resilience Metrics Router
Exposes circuit breaker and retry policy statistics
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional

from src.core.resilience import (
    get_circuit_breaker_registry,
    get_retry_policy_registry
)

router = APIRouter(prefix="/resilience", tags=["Resilience"])


@router.get("/stats")
async def get_resilience_stats() -> Dict[str, Any]:
    """
    Get resilience statistics for all services

    Returns circuit breaker and retry policy stats
    """
    circuit_registry = get_circuit_breaker_registry()
    retry_registry = get_retry_policy_registry()

    return {
        "circuit_breakers": circuit_registry.get_all_stats(),
        "retry_policies": retry_registry.get_all_stats()
    }


@router.get("/circuit-breakers")
async def get_circuit_breakers() -> Dict[str, Any]:
    """
    Get all circuit breaker statistics
    """
    registry = get_circuit_breaker_registry()
    return registry.get_all_stats()


@router.get("/circuit-breakers/{service_name}")
async def get_circuit_breaker(service_name: str) -> Dict[str, Any]:
    """
    Get circuit breaker statistics for a specific service
    """
    registry = get_circuit_breaker_registry()
    circuit = registry.get(service_name)

    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit breaker for {service_name} not found")

    return circuit.get_stats()


@router.post("/circuit-breakers/{service_name}/reset")
async def reset_circuit_breaker(service_name: str) -> Dict[str, str]:
    """
    Manually reset circuit breaker to CLOSED state
    """
    registry = get_circuit_breaker_registry()
    circuit = registry.get(service_name)

    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit breaker for {service_name} not found")

    await circuit.reset()

    return {
        "message": f"Circuit breaker for {service_name} reset to CLOSED",
        "service": service_name
    }


@router.post("/circuit-breakers/reset-all")
async def reset_all_circuit_breakers() -> Dict[str, str]:
    """
    Reset all circuit breakers to CLOSED state
    """
    registry = get_circuit_breaker_registry()
    await registry.reset_all()

    return {"message": "All circuit breakers reset to CLOSED"}


@router.get("/retry-policies")
async def get_retry_policies() -> Dict[str, Any]:
    """
    Get all retry policy statistics
    """
    registry = get_retry_policy_registry()
    return registry.get_all_stats()


@router.get("/retry-policies/{service_name}")
async def get_retry_policy(service_name: str) -> Dict[str, Any]:
    """
    Get retry policy statistics for a specific service
    """
    registry = get_retry_policy_registry()
    policy = registry.get(service_name)

    if not policy:
        raise HTTPException(status_code=404, detail=f"Retry policy for {service_name} not found")

    return policy.get_stats()


@router.post("/retry-policies/reset-stats")
async def reset_all_retry_stats() -> Dict[str, str]:
    """
    Reset all retry policy statistics
    """
    registry = get_retry_policy_registry()
    registry.reset_all_stats()

    return {"message": "All retry policy stats reset"}


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "resilience-metrics"}
