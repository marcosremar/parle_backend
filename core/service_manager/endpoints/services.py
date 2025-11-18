"""
Service Manager - Services Endpoints
Extracted service-related endpoints for better code organization
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import ValidationError, BaseModel
from typing import Optional, Dict, List, Any
from pathlib import Path
from datetime import datetime
import asyncio
import uuid
import subprocess
import aiohttp
import requests

# Import models from centralized location (avoids circular import with main.py)
from src.core.service_manager.endpoints.models import (
    ServiceRequest,
    VenvInstallRequest,
    PackageInstallRequest,
    VenvStatusResponse,
    BulkOperationRequest,
    BulkInstallRequest
)

# Import from core modules
from src.core.service_manager.core import HealthStatus
from src.core.service_manager.models import ProcessStatus
from src.core.core_logging import setup_logging
from src.core.exceptions import UltravoxError, wrap_exception

# Setup logging
logger = setup_logging("service-manager-endpoints", level="INFO")

# Create router with prefix and tags
router = APIRouter(prefix="/services", tags=["services"])

# ============================================================================
# Manager Dependency Injection
# ============================================================================

_manager = None

def set_manager(manager) -> Any:
    """Set the global manager instance"""
    global _manager
    _manager = manager

def get_manager() -> Any:
    """Dependency to get the manager instance"""
    if _manager is None:
        raise HTTPException(status_code=500, detail="Service manager not initialized")
    return _manager

# ============================================================================
# Input Validation - Security
# ============================================================================

def validate_service_id(service_id: str) -> str:
    """
    Validate service_id input to prevent path traversal and injection attacks

    Args:
        service_id: Service identifier to validate

    Returns:
        Validated service_id

    Raises:
        HTTPException: If service_id contains malicious patterns
    """
    import re

    # Check for null or empty
    if not service_id or not service_id.strip():
        raise HTTPException(status_code=400, detail="service_id cannot be empty")

    # Check for path traversal patterns
    if ".." in service_id:
        raise HTTPException(status_code=400, detail="Invalid service_id: path traversal detected")

    # Check for absolute paths
    if service_id.startswith("/") or service_id.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid service_id: absolute paths not allowed")

    # Only allow alphanumeric, underscore, hyphen (standard service naming)
    if not re.match(r'^[a-zA-Z0-9_\-]+$', service_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid service_id: only alphanumeric characters, underscores, and hyphens allowed"
        )

    # Length check (reasonable limit)
    if len(service_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid service_id: too long (max 64 characters)")

    return service_id

# ============================================================================
# Service Status & Health Endpoints
# ============================================================================

@router.get("")
async def get_services(manager = Depends(get_manager)):
    """Lista status de todos os serviços"""
    return manager.get_all_status()

@router.get("/status")
async def get_services_status(check_health: bool = False, manager = Depends(get_manager)):
    """Status detalhado de todos os serviços - FAST by default

    Args:
        check_health: If True, performs health checks (slower). Default False for fast response.
    """
    # Use fast version by default, with optional health checks
    if check_health:
        status = await manager.get_all_status_with_health()
    else:
        status = manager.get_all_status()

    # Exclude service_manager from verification counts (self-check issue)
    status_without_manager = {k: v for k, v in status.items() if k != "service_manager"}

    # Count only healthy services (excluding service_manager)
    total = len(status_without_manager)
    healthy = sum(1 for s in status_without_manager.values() if s.get("healthy", False))
    unhealthy = total - healthy

    return {
        "summary": {
            "total_services": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "all_healthy": healthy == total
        },
        "services": status,  # Still include all services in the detailed list
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health")
async def get_services_health(manager = Depends(get_manager)):
    """Get detailed health status of all services with parallel health checks"""
    status = await manager.get_all_status_with_health()

    # Exclude service_manager from verification counts
    status_without_manager = {k: v for k, v in status.items() if k != "service_manager"}

    total = len(status_without_manager)
    healthy = sum(1 for s in status_without_manager.values() if s.get("healthy", False))
    unhealthy = total - healthy

    return {
        "summary": {
            "total_services": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "all_healthy": healthy == total
        },
        "services": status,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/internal/health")
async def get_internal_services_health(manager = Depends(get_manager)):
    """
    Get health status of internal (in-process) services

    Internal services don't have HTTP endpoints, so we check if they're loaded
    """
    results = {}

    # Check all internal services
    for service_id, service_instance in manager.internal_services.items():
        try:
            # Service is loaded and available
            is_healthy = service_instance is not None

            # Try to get additional info if service has health method
            health_info = {}
            if hasattr(service_instance, 'get_health'):
                health_info = await service_instance.get_health()
            elif hasattr(service_instance, 'health'):
                health_info = service_instance.health()

            results[service_id] = {
                "healthy": is_healthy,
                "loaded": True,
                "type": "internal",
                "instance": str(type(service_instance).__name__),
                "health_info": health_info if health_info else None
            }
        except Exception as e:
            results[service_id] = {
                "healthy": False,
                "loaded": False,
                "error": str(e),
                "type": "internal"
            }

    total = len(results)
    healthy = sum(1 for r in results.values() if r.get("healthy", False))

    return {
        "summary": {
            "total": total,
            "healthy": healthy,
            "unhealthy": total - healthy,
            "all_healthy": healthy == total
        },
        "services": results,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Bulk Operations Endpoints
# ============================================================================

@router.post("/bulk/start")
async def bulk_start_services(request: BulkOperationRequest, manager = Depends(get_manager)):
    """
    Start multiple services at once with GPU awareness and retry logic

    Args:
        request: Bulk operation configuration (if service_ids is None, starts all services)

    Returns:
        Results for each service including GPU allocation details
    """
    try:
        # Try to import GPU manager (may not be available if torch not installed)
        gpu_manager = None
        gpu_available = False
        try:
            from src.core.gpu_memory_manager import get_gpu_manager, ServiceGPURequirement
            gpu_manager = get_gpu_manager()
            gpu_available = True
            logger.info("✅ GPU manager available")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"⚠️  GPU manager not available: {e}")
            logger.info("   Continuing without GPU management...")

        # If service_ids is None, start all enabled auto-start services
        if request.service_ids is not None:
            service_ids = request.service_ids
        else:
            # Filter to only enabled auto-start services
            service_ids = [
                sid for sid, config in manager.execution_config.services.items()
                if config.auto_start and getattr(config, 'enabled', True)
            ]
            logger.info(f"📋 Filtered to {len(service_ids)} enabled auto-start services (from {len(manager.services)} total)")

        logger.info(f"🚀 Bulk starting {len(service_ids)} services (GPU cleanup: {request.gpu_cleanup})")
        results = []
        failed_count = 0

        # Check profile restrictions for GPU services
        profile_allows_gpu = True
        try:
            from tests.profile_manager import get_profile_manager
            pm = get_profile_manager()
            active_profile = pm.get_active_profile()
            restrictions = active_profile.get('restrictions', {})
            profile_allows_gpu = restrictions.get('allow_gpu_services', True)

            if not profile_allows_gpu:
                logger.warning(f"⚠️  Profile '{pm.get_active_profile_name()}' does not allow GPU services (allow_gpu_services: false)")
        except ImportError:
            pass  # Profile manager not available

        # Validate all service IDs first (check both external and internal services)
        for service_id in service_ids:
            # Check if service exists in either manager.services OR execution_config
            in_services = service_id in manager.services
            in_exec_config = manager.execution_config.get_service_info(service_id) is not None

            if not in_services and not in_exec_config:
                results.append({
                    "service_id": service_id,
                    "success": False,
                    "error": "Service not found in services registry or execution config"
                })
                failed_count += 1

        # Services to start includes both external (in manager.services) and internal-only services
        services_to_start = [
            sid for sid in service_ids
            if sid in manager.services or manager.execution_config.get_service_info(sid) is not None
        ]
        services_to_start = [sid for sid in services_to_start if sid not in [r["service_id"] for r in results if not r.get("success", True)]]

        # Separate GPU and non-GPU services
        gpu_services = []
        non_gpu_services = []
        blocked_gpu_services = []

        for service_id in services_to_start:
            service_config = manager.execution_config.services.get(service_id)
            is_gpu = service_config and hasattr(service_config, 'gpu_required') and service_config.gpu_required

            if is_gpu:
                # Block GPU services if profile doesn't allow them
                if not profile_allows_gpu:
                    blocked_gpu_services.append(service_id)
                    results.append({
                        "service_id": service_id,
                        "success": False,
                        "error": f"GPU service blocked by profile restrictions (allow_gpu_services: false)",
                        "blocked": True
                    })
                    failed_count += 1
                    logger.warning(f"🚫 {service_id}: Blocked - GPU services not allowed in current profile")
                else:
                    gpu_services.append(service_id)
            else:
                non_gpu_services.append(service_id)

        if blocked_gpu_services:
            logger.warning(f"🚫 Blocked {len(blocked_gpu_services)} GPU services due to profile restrictions: {blocked_gpu_services}")

        logger.info(f"   GPU services: {len(gpu_services)} - {gpu_services}")
        logger.info(f"   Non-GPU services: {len(non_gpu_services)}")

        # Clean GPU memory if requested
        if request.gpu_cleanup and len(gpu_services) > 0 and gpu_available and gpu_manager:
            logger.info("🧹 Cleaning GPU memory before starting services...")
            gpu_manager.cleanup_gpu_memory(level="soft")
            await asyncio.sleep(1)

        # Sort GPU services by startup_priority
        if request.respect_dependencies and gpu_services:
            gpu_services_sorted = []
            for service_id in manager.startup_order:
                if service_id in gpu_services:
                    gpu_services_sorted.append(service_id)
            # Add remaining GPU services
            for service_id in gpu_services:
                if service_id not in gpu_services_sorted:
                    gpu_services_sorted.append(service_id)
            gpu_services = gpu_services_sorted

        # Start GPU services sequentially with memory management
        for service_id in gpu_services:
            service_result = await _start_service_with_retry(
                manager, service_id, gpu_manager, request
            )
            results.append(service_result)

            if not service_result.get("success", False):
                failed_count += 1
            else:
                # Delay to let GPU memory stabilize
                await asyncio.sleep(2)

        # Sort non-GPU services if respecting dependencies
        if request.respect_dependencies and non_gpu_services:
            ordered_non_gpu = []
            for service_id in manager.startup_order:
                if service_id in non_gpu_services:
                    ordered_non_gpu.append(service_id)
            for service_id in non_gpu_services:
                if service_id not in ordered_non_gpu:
                    ordered_non_gpu.append(service_id)
            non_gpu_services = ordered_non_gpu

        # Start non-GPU services (can be parallel)
        for service_id in non_gpu_services:
            service_result = await _start_service_with_retry(
                manager, service_id, None, request
            )
            results.append(service_result)

            if not service_result.get("success", False):
                failed_count += 1
            elif not request.parallel:
                await asyncio.sleep(0.5)

        # Get final GPU status
        gpu_status = None
        if gpu_available and gpu_manager:
            gpu_status = gpu_manager.get_status_report()

        return {
            "success": failed_count == 0,
            "total": len(service_ids),
            "started": len(results) - failed_count,
            "failed": failed_count,
            "results": results,
            "gpu_status": gpu_status
        }

    except Exception as e:
        logger.error(f"❌ Error in bulk start: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def _start_service_with_retry(
    manager: 'ServiceManager',
    service_id: str,
    gpu_manager: Optional['GPUMemoryManager'],
    request: BulkOperationRequest
) -> Dict:
    """
    Start a service with retry logic and GPU memory management

    Args:
        manager: ServiceManager instance
        service_id: Service to start
        gpu_manager: GPU manager (None for non-GPU services or if GPU not available)
        request: Bulk operation request with retry settings

    Returns:
        Result dictionary with service status
    """
    # Try to import ServiceGPURequirement if available
    ServiceGPURequirement = None
    try:
        from src.core.gpu_memory_manager import ServiceGPURequirement as _ServiceGPURequirement
        ServiceGPURequirement = _ServiceGPURequirement
    except (ImportError, ModuleNotFoundError):
        pass  # GPU not available, ServiceGPURequirement will remain None

    max_attempts = request.max_retries if request.retry_on_failure else 1
    is_internal = manager._is_internal_service(service_id)

    # Get GPU requirements if this is a GPU service
    service_config = manager.execution_config.services.get(service_id)
    is_gpu_service = service_config and hasattr(service_config, 'gpu_required') and service_config.gpu_required

    gpu_requirement = None
    if is_gpu_service and gpu_manager and service_config and ServiceGPURequirement:
        gpu_requirement = ServiceGPURequirement(
            service_id=service_id,
            memory_mb=getattr(service_config, 'gpu_memory_mb', 1000),
            utilization=getattr(service_config, 'gpu_memory_utilization', 0.75),
            utilization_min=getattr(service_config, 'gpu_memory_utilization_min', None),
            is_gpu_service=True
        )

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"🎯 Starting {service_id} (attempt {attempt}/{max_attempts})")

            # Initialize GPU environment overrides (default: None for non-GPU services)
            gpu_env = None
            suggested_utilization = None

            # GPU pre-flight check
            if gpu_requirement and gpu_manager and request.adjust_gpu_memory:
                can_allocate, suggested_utilization = gpu_manager.can_allocate(gpu_requirement)

                if not can_allocate:
                    if attempt < max_attempts:
                        logger.warning(f"   ⚠️  Insufficient GPU memory, cleaning and retrying...")
                        gpu_manager.cleanup_gpu_memory(level="hard")
                        await asyncio.sleep(2)
                        continue
                    else:
                        return {
                            "service_id": service_id,
                            "success": False,
                            "error": "Insufficient GPU memory after retries",
                            "attempts": attempt,
                            "type": "internal" if is_internal else "external"
                        }

                # Prepare GPU environment overrides if adjusted
                if suggested_utilization and suggested_utilization != gpu_requirement.utilization:
                    logger.info(f"   📊 Adjusting GPU utilization: {suggested_utilization:.2f}")
                    gpu_env = {'VLLM_GPU_MEMORY_UTILIZATION': str(suggested_utilization)}

                # Reserve memory
                expected_mb = int(suggested_utilization * 23700) if suggested_utilization else gpu_requirement.memory_mb
                gpu_manager.reserve_memory(service_id, expected_mb)

            # Start the service
            if is_internal:
                success = await manager._start_local_service(service_id, manager.app)
                result = {
                    "service_id": service_id,
                    "success": success,
                    "type": "internal",
                    "attempts": attempt
                }
            else:
                # Pass GPU environment overrides to start_service
                result = manager.start_service(service_id, force=False, gpu_env_overrides=gpu_env)
                result["attempts"] = attempt

            # Add GPU info if applicable
            if gpu_requirement and suggested_utilization:
                result["gpu_utilization"] = suggested_utilization
                result["gpu_memory_reserved_mb"] = expected_mb

            if result.get("success"):
                logger.info(f"   ✅ {service_id} started successfully")
                return result
            else:
                # Release GPU memory on failure
                if gpu_manager and gpu_requirement:
                    gpu_manager.release_memory(service_id)

                if attempt < max_attempts:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"   ⚠️  Retry in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"   ❌ {service_id} failed after {attempt} attempts")
                    return result

        except Exception as e:
            # Release GPU memory on error
            if gpu_manager and gpu_requirement:
                gpu_manager.release_memory(service_id)

            if attempt < max_attempts:
                wait_time = 2 ** attempt
                logger.warning(f"   ⚠️  Error: {e}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"   ❌ {service_id} error after {attempt} attempts: {e}")
                return {
                    "service_id": service_id,
                    "success": False,
                    "error": str(e),
                    "attempts": attempt,
                    "type": "internal" if is_internal else "external"
                }

    # Should not reach here
    return {
        "service_id": service_id,
        "success": False,
        "error": "Max retries exceeded",
        "attempts": max_attempts,
        "type": "internal" if is_internal else "external"
    }

@router.post("/bulk/stop")
async def bulk_stop_services(request: BulkOperationRequest, manager = Depends(get_manager)):
    """
    Stop multiple services at once

    Args:
        request: Bulk operation configuration (if service_ids is None, stops all services)

    Returns:
        Results for each service
    """
    try:
        # If service_ids is None, stop all services
        service_ids = request.service_ids if request.service_ids is not None else list(manager.services.keys())

        logger.info(f"🛑 Bulk stopping {len(service_ids)} services")

        results = []
        failed_count = 0

        # Validate all service IDs first
        for service_id in service_ids:
            if service_id not in manager.services:
                results.append({
                    "service_id": service_id,
                    "success": False,
                    "error": "Service not found"
                })
                failed_count += 1

        # If respecting dependencies, stop in reverse order
        services_to_stop = [sid for sid in service_ids if sid in manager.services]

        if request.respect_dependencies:
            services_to_stop.reverse()

        # Stop services
        for service_id in services_to_stop:
            try:
                is_internal = manager._is_internal_service(service_id)

                if is_internal:
                    success = await manager._stop_internal_service(service_id)
                    result = {
                        "service_id": service_id,
                        "success": success,
                        "type": "internal"
                    }
                else:
                    result = manager.stop_service(service_id)

                results.append(result)

                if not result.get("success", False):
                    failed_count += 1

                # Small delay between services if not parallel
                if not request.parallel:
                    await asyncio.sleep(0.5)

            except Exception as e:
                results.append({
                    "service_id": service_id,
                    "success": False,
                    "error": str(e)
                })
                failed_count += 1

        return {
            "success": failed_count == 0,
            "total": len(service_ids),
            "stopped": len(results) - failed_count,
            "failed": failed_count,
            "results": results
        }

    except Exception as e:
        logger.error(f"❌ Error in bulk stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk/restart")
async def bulk_restart_services(request: BulkOperationRequest, manager = Depends(get_manager)):
    """
    Restart multiple services at once

    Args:
        request: Bulk operation configuration (if service_ids is None, restarts all services)

    Returns:
        Results for each service
    """
    try:
        # If service_ids is None, restart all services
        service_ids = request.service_ids if request.service_ids is not None else list(manager.services.keys())

        logger.info(f"🔄 Bulk restarting {len(service_ids)} services")

        # Stop all first
        stop_response = await bulk_stop_services(request, manager)

        # Wait a bit
        await asyncio.sleep(2)

        # Start all
        start_response = await bulk_start_services(request, manager)

        return {
            "success": stop_response["success"] and start_response["success"],
            "total": len(service_ids),
            "stop_results": stop_response["results"],
            "start_results": start_response["results"]
        }

    except Exception as e:
        logger.error(f"❌ Error in bulk restart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk/health")
async def bulk_health_check(request: BulkOperationRequest, manager = Depends(get_manager)):
    """
    Health check multiple services in parallel

    Args:
        request: Service IDs to check (if None, checks all services)

    Returns:
        Health status for each service
    """
    try:
        # If service_ids is None, check all services
        service_ids = request.service_ids if request.service_ids is not None else list(manager.services.keys())

        logger.info(f"🏥 Bulk health check for {len(service_ids)} services")

        results = []

        # Validate all service IDs
        for service_id in service_ids:
            if service_id not in manager.services:
                results.append({
                    "service_id": service_id,
                    "healthy": False,
                    "error": "Service not found"
                })
                continue

        # Perform health checks in parallel using existing method
        import time as time_module

        async def check_service_health(service_id) -> Dict[str, Any]:
            """Check health of a single service using aiohttp (non-blocking)"""
            service = manager.services[service_id]

            if service.process_status != ProcessStatus.RUNNING:
                return {
                    "service_id": service_id,
                    "healthy": False,
                    "status": service.process_status.value,
                    "message": "Service not running"
                }

            # Try health check endpoint with aiohttp (non-blocking)
            try:
                start_time = time_module.time()
                service_url = f"http://localhost:{service.port}/health"

                # Use aiohttp for non-blocking HTTP request
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(service_url) as response:
                        response_time_ms = (time_module.time() - start_time) * 1000
                        healthy = response.status == 200

                        # Record in metrics tracker
                        from src.core.service_manager.metrics_tracker import HealthStatus
                        status = HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY
                        manager.metrics_tracker.record_health_check(
                            service_id,
                            status,
                            response_time_ms=response_time_ms
                        )

                        return {
                            "service_id": service_id,
                            "healthy": healthy,
                            "status": "healthy" if healthy else "unhealthy",
                            "response_time_ms": round(response_time_ms, 2),
                            "http_status": response.status
                        }

            except Exception as e:
                # Record failed health check
                from src.core.service_manager.metrics_tracker import HealthStatus
                manager.metrics_tracker.record_health_check(
                    service_id,
                    HealthStatus.UNHEALTHY,
                    error=str(e)
                )

                return {
                    "service_id": service_id,
                    "healthy": False,
                    "status": "unhealthy",
                    "error": str(e)
                }

        # Run all health checks in parallel
        valid_service_ids = [sid for sid in service_ids if sid in manager.services]
        health_checks = [check_service_health(sid) for sid in valid_service_ids]
        check_results = await asyncio.gather(*health_checks)

        results.extend(check_results)

        # Count healthy/unhealthy
        healthy_count = sum(1 for r in results if r.get("healthy", False))
        unhealthy_count = len(results) - healthy_count

        return {
            "total": len(service_ids),
            "healthy": healthy_count,
            "unhealthy": unhealthy_count,
            "all_healthy": unhealthy_count == 0,
            "results": results
        }

    except Exception as e:
        logger.error(f"❌ Error in bulk health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk/validate")
async def bulk_validate_services(request: BulkOperationRequest, manager = Depends(get_manager)):
    """
    Validate multiple services using BulkInstaller.validate_all_services()

    This validates service installation, dependencies, and importability.
    Respects profile settings - disabled services are skipped automatically.

    🔥 HOT RELOAD TEST #3 - Service Manager auto-reload enabled!

    AUTOMATIC SERVICE STARTING:
    - Checks if services are running before validation
    - Automatically starts services that are not running
    - Waits for services to become healthy before validating

    Args:
        request: Service IDs to validate (if None, validates all enabled services)

    Returns:
        Validation results for each service
    """
    try:
        logger.info(f"DEBUG: bulk_validate called with service_ids = {request.service_ids}")

        from pathlib import Path
        import sys
        from src.core.service_manager.discovery.service_discovery import ServiceDiscovery
        from src.core.service_manager.discovery.bulk_installer import BulkInstaller, InstallationStatus
        import time as time_module

        # Create discovery
        services_dir = Path.cwd() / 'src' / 'services'
        discovery = ServiceDiscovery(services_dir=services_dir)
        discovery.discover_all_services()

        # Create installer
        installer = BulkInstaller(discovery=discovery)

        # Determine which services to validate
        logger.info(f"DEBUG: Checking if request.service_ids ({request.service_ids}) is None...")
        if request.service_ids is None:
            logger.info(f"🔍 Bulk validating all services using BulkInstaller.validate_all_services()")
            logger.info(f"   FORCE LOG: Starting GPU restriction check...")

            # Check profile restrictions for GPU services
            profile_allows_gpu = True
            try:
                from tests.profile_manager import get_profile_manager
                pm = get_profile_manager()
                active_profile = pm.get_active_profile()
                restrictions = active_profile.get('restrictions', {})
                profile_allows_gpu = restrictions.get('allow_gpu_services', True)

                if not profile_allows_gpu:
                    logger.warning(f"⚠️  Profile '{pm.get_active_profile_name()}' does not allow GPU services (allow_gpu_services: false)")
            except ImportError as e:
                logger.warning(f"⚠️  Profile manager not available (ImportError): {e}")
                pass  # Profile manager not available
            except Exception as e:
                logger.error(f"❌ Error checking profile restrictions: {e}")
                pass  # Continue with default (allow GPU)

            # Get all enabled services from profile
            enabled_service_ids = []
            blocked_gpu_services = []

            for service_id in discovery.services.keys():
                service_info = manager.execution_config.get_service_info(service_id)

                # Check if service is enabled
                is_enabled = True
                if service_info:
                    is_enabled = getattr(service_info, 'enabled', True)

                if not is_enabled:
                    # Skip disabled services
                    continue

                # Check if this is a GPU service
                service_config = manager.execution_config.services.get(service_id)
                is_gpu = service_config and hasattr(service_config, 'gpu_required') and service_config.gpu_required

                if is_gpu and not profile_allows_gpu:
                    # Block GPU services if profile doesn't allow them
                    blocked_gpu_services.append(service_id)
                    logger.warning(f"🚫 {service_id}: Blocked - GPU services not allowed in current profile")
                else:
                    # Include the service for validation
                    enabled_service_ids.append(service_id)

            if blocked_gpu_services:
                logger.warning(f"🚫 Blocked {len(blocked_gpu_services)} GPU services due to profile restrictions: {blocked_gpu_services}")

            logger.info(f"   Validating {len(enabled_service_ids)} enabled services (skipping {len(discovery.services) - len(enabled_service_ids) - len(blocked_gpu_services)} disabled, {len(blocked_gpu_services)} GPU blocked)")
            services_to_validate = enabled_service_ids

            # Debug: Show what's in the list
            logger.info(f"   DEBUG: services_to_validate list ({len(services_to_validate)} services): {services_to_validate}")
            logger.info(f"   DEBUG: blocked_gpu_services ({len(blocked_gpu_services)} services): {blocked_gpu_services}")
        else:
            logger.info(f"🔍 Bulk validating {len(request.service_ids)} specific services")
            services_to_validate = request.service_ids

        # ========================================================================
        # STEP 1: Check which services are running (using aiohttp for non-blocking I/O)
        # ========================================================================
        logger.info("🏥 Checking service availability before validation...")

        services_not_running = []
        services_running = []

        # Helper function for health checks with aiohttp
        async def check_service_running(service_id) -> None:
            """Check if a service is running using aiohttp (non-blocking)"""
            if service_id not in manager.services:
                logger.warning(f"⚠️  {service_id}: Service not found in manager")
                return None

            service = manager.services[service_id]

            # Check process status
            if service.process_status != ProcessStatus.RUNNING:
                services_not_running.append(service_id)
                logger.info(f"❌ {service_id}: Not running (status: {service.process_status.value})")
                return None

            # Try health check endpoint with aiohttp (non-blocking)
            try:
                service_url = f"http://localhost:{service.port}/health"
                timeout = aiohttp.ClientTimeout(total=2)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(service_url) as response:
                        if response.status == 200:
                            services_running.append(service_id)
                            logger.info(f"✅ {service_id}: Running and healthy")
                        else:
                            services_not_running.append(service_id)
                            logger.info(f"❌ {service_id}: Unhealthy (HTTP {response.status})")
            except Exception as e:
                services_not_running.append(service_id)
                logger.info(f"❌ {service_id}: Not responding to health checks ({e})")

        # Run health checks in parallel for all services
        await asyncio.gather(*[check_service_running(sid) for sid in services_to_validate], return_exceptions=True)

        logger.info(f"📊 Service availability: {len(services_running)} running, {len(services_not_running)} not running")

        # ========================================================================
        # STEP 2: Start services that are not running
        # ========================================================================
        start_results = {}

        if len(services_not_running) > 0:
            logger.info(f"🚀 Starting {len(services_not_running)} services before validation...")

            # Create request for bulk start
            start_request = BulkOperationRequest(
                service_ids=services_not_running,
                respect_dependencies=True,
                retry_on_failure=True,
                max_retries=2,
                parallel=False,
                gpu_cleanup=True,
                adjust_gpu_memory=True
            )

            # Call bulk_start_services
            try:
                start_response = await bulk_start_services(start_request, manager)
                start_results = start_response

                if start_response.get("success"):
                    logger.info(f"✅ Successfully started {start_response.get('started', 0)} services")
                else:
                    logger.warning(f"⚠️  Some services failed to start: {start_response.get('failed', 0)} failures")

            except Exception as start_error:
                logger.error(f"❌ Error starting services: {start_error}")
                # Continue with validation anyway

            # ========================================================================
            # STEP 3: Wait for services to become healthy (using aiohttp)
            # ========================================================================
            logger.info("⏳ Waiting for services to become healthy...")

            max_wait_seconds = 30
            check_interval = 2
            elapsed = 0

            # Helper function to check service health with aiohttp
            async def check_service_health_status(service_id) -> bool:
                """Check if a service is healthy using aiohttp (non-blocking)"""
                if service_id not in manager.services:
                    return True  # Skip if not found

                service = manager.services[service_id]

                if service.process_status != ProcessStatus.RUNNING:
                    return False

                try:
                    service_url = f"http://localhost:{service.port}/health"
                    timeout = aiohttp.ClientTimeout(total=2)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(service_url) as response:
                            return response.status == 200
                except Exception as e:
                    logger.debug(f"Health check failed for {service_id} during wait: {e}")
                    return False

            while elapsed < max_wait_seconds:
                # Check all services in parallel using aiohttp
                health_checks = [check_service_health_status(sid) for sid in services_not_running]
                health_results = await asyncio.gather(*health_checks, return_exceptions=True)

                # Filter out exceptions and check if all are healthy
                all_healthy = all(
                    result is True
                    for result in health_results
                    if not isinstance(result, Exception)
                )

                if all_healthy:
                    logger.info(f"✅ All services healthy after {elapsed}s")
                    break

                await asyncio.sleep(check_interval)
                elapsed += check_interval

            if elapsed >= max_wait_seconds:
                logger.warning(f"⚠️  Timeout waiting for services to become healthy (waited {max_wait_seconds}s)")

            # Remove GPU services that were blocked from starting
            if start_results and 'results' in start_results:
                blocked_services = [
                    r['service_id'] for r in start_results['results']
                    if r.get('blocked', False)
                ]
                if blocked_services:
                    logger.info(f"🚫 Removing {len(blocked_services)} blocked GPU services from validation: {blocked_services}")
                    services_to_validate = [
                        sid for sid in services_to_validate
                        if sid not in blocked_services
                    ]

        # ========================================================================
        # STEP 4: Proceed with validation
        # ========================================================================
        logger.info("🔍 Starting validation...")

        results_dict = {}
        for service_id in services_to_validate:
            result = installer.validate_service(service_id)
            results_dict[service_id] = result

        # Convert results to summary format
        passed = 0
        failed = 0
        results = []

        for service_id, result in results_dict.items():
            # Convert InstallationResult to API response format
            is_success = result.status == InstallationStatus.SUCCESS

            if is_success:
                passed += 1
                logger.info(f"✅ {service_id}: {result.message}")
            else:
                failed += 1
                logger.warning(f"❌ {service_id}: {result.message}")

            results.append({
                "service_id": service_id,
                "success": is_success,
                "status": result.status.value,
                "message": result.message,
                "duration_seconds": result.duration_seconds,
                "error": result.error
            })

        return {
            "success": failed == 0,
            "message": f"Validation completed: {passed} passed, {failed} failed",
            "summary": {
                "total_services": len(results_dict),
                "passed_services": passed,
                "failed_services": failed,
                "services_started": len(services_not_running),
                "services_already_running": len(services_running),
                "validation_method": "BulkInstaller (service structure, dependencies, imports)"
            },
            "results": results,
            "start_results": start_results if services_not_running else None,
            "timestamp": datetime.now().isoformat(),
            "note": "Automatically starts services before validation. Validates service installation, dependencies, and importability using BulkInstaller"
        }

    except Exception as e:
        logger.error(f"❌ Bulk validation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/bulk/validate/async")
async def bulk_validate_services_async(request: BulkOperationRequest, background_tasks: BackgroundTasks, manager = Depends(get_manager)):
    """
    Async version of bulk validation - returns tracking_id immediately

    Validates multiple services in parallel using worker pool for faster execution.
    Use this for validating many services to avoid timeout.

    Args:
        request: Service IDs to validate (if None, validates all services)

    Returns:
        tracking_id and status URL for checking results
    """
    try:
        import asyncio
        import aiohttp
        from concurrent.futures import ThreadPoolExecutor
        from src.core.task_manager import get_task_manager, TaskType, TaskStatus

        # If service_ids is None, validate all services
        service_ids = request.service_ids if request.service_ids is not None else list(manager.services.keys())

        # Get task manager instance
        task_mgr = get_task_manager("service-manager")

        # Create task entry - this returns the tracking_id
        tracking_id = task_mgr.create_task(
            task_type=TaskType.VALIDATE_ALL,
            metadata={
                "service_count": len(service_ids),
                "service_ids": service_ids
            }
        )

        logger.info(f"🚀 Starting async bulk validation for {len(service_ids)} services (tracking_id: {tracking_id})")

        async def validate_service_async(service_id: str, session: aiohttp.ClientSession) -> dict:
            """Validate a single service asynchronously"""
            try:
                if service_id not in manager.services:
                    return {
                        "service_id": service_id,
                        "success": False,
                        "error": "Service not found"
                    }

                service = manager.services[service_id]
                is_internal = manager._is_internal_service(service_id)

                if is_internal:
                    # Match the routing logic from start_internal_service (lines 442-447)
                    if service_id in ["session", "scenarios"]:
                        # These use /internal/ prefix
                        validate_url = f"http://localhost:8888/internal/{service_id}/validate"
                    else:
                        # All other internal services (orchestrator, storage, external_*) use /api/ prefix
                        validate_url = f"http://localhost:8888/api/{service_id}/validate"
                else:
                    if not manager.check_port(service.port):
                        return {
                            "service_id": service_id,
                            "name": service.name,
                            "port": service.port,
                            "success": False,
                            "status": "not_running",
                            "error": f"Service not running on port {service.port}"
                        }
                    validate_url = f"http://localhost:{service.port}/validate"

                logger.info(f"🔍 [{tracking_id}] Validating {service_id} at {validate_url}")

                async with session.get(validate_url, timeout=aiohttp.ClientTimeout(total=90)) as response:  # Increased from 45s to 90s
                    if response.status == 200:
                        validate_data = await response.json()

                        status = validate_data.get("status", "unknown")

                        if "summary" in validate_data:
                            summary = validate_data["summary"]
                            endpoints_tested = summary.get("total", 0)
                            endpoints_passed = summary.get("passed", 0)
                            endpoints_failed = summary.get("failed", 0)
                            success_rate_num = summary.get("success_rate", 0)
                            success_rate = f"{success_rate_num}%" if isinstance(success_rate_num, (int, float)) else str(success_rate_num)
                        else:
                            endpoints_tested = validate_data.get("endpoints_tested", 0)
                            endpoints_passed = validate_data.get("endpoints_passed", 0)
                            endpoints_failed = validate_data.get("endpoints_failed", 0)
                            success_rate = validate_data.get("success_rate", "0%")

                        is_success = status in ["healthy", "degraded"] and endpoints_failed == 0

                        if is_success:
                            logger.info(f"✅ [{tracking_id}] {service_id}: {success_rate}")
                        else:
                            logger.warning(f"❌ [{tracking_id}] {service_id}: {status}")

                        return {
                            "service_id": service_id,
                            "name": service.name,
                            "port": service.port,
                            "success": is_success,
                            "status": status,
                            "endpoints_tested": endpoints_tested,
                            "endpoints_passed": endpoints_passed,
                            "endpoints_failed": endpoints_failed,
                            "success_rate": success_rate,
                            "errors": validate_data.get("errors", []),
                            "timestamp": validate_data.get("timestamp")
                        }
                    else:
                        logger.error(f"❌ [{tracking_id}] {service_id}: HTTP {response.status}")
                        return {
                            "service_id": service_id,
                            "name": service.name,
                            "port": service.port,
                            "success": False,
                            "status": "validation_failed",
                            "error": f"HTTP {response.status}"
                        }

            except asyncio.TimeoutError:
                logger.error(f"❌ [{tracking_id}] {service_id}: Timeout")
                return {
                    "service_id": service_id,
                    "success": False,
                    "status": "timeout",
                    "error": "Validation timeout (45s)"
                }
            except Exception as e:
                logger.error(f"❌ [{tracking_id}] {service_id}: {e}")
                return {
                    "service_id": service_id,
                    "success": False,
                    "status": "error",
                    "error": str(e)
                }

        async def run_validation() -> Any:
            """Run all validations in parallel"""
            try:
                # Get task object and mark as running
                task = task_mgr.get_task(tracking_id)
                if task:
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    task.updated_at = datetime.now()
                    task_mgr.update_task_progress(tracking_id, current=0, total=len(service_ids), message="Starting validation")

                # Create aiohttp session
                connector = aiohttp.TCPConnector(limit=10)  # Max 10 concurrent connections
                async with aiohttp.ClientSession(connector=connector) as session:
                    # Run validations in parallel
                    tasks = [validate_service_async(service_id, session) for service_id in service_ids]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    processed_results = []
                    passed = 0
                    failed = 0
                    total_endpoints_tested = 0
                    total_endpoints_passed = 0
                    total_endpoints_failed = 0

                    for result in results:
                        if isinstance(result, Exception):
                            failed += 1
                            processed_results.append({
                                "success": False,
                                "error": str(result)
                            })
                        else:
                            processed_results.append(result)
                            if result.get("success"):
                                passed += 1
                            else:
                                failed += 1

                            total_endpoints_tested += result.get("endpoints_tested", 0)
                            total_endpoints_passed += result.get("endpoints_passed", 0)
                            total_endpoints_failed += result.get("endpoints_failed", 0)

                    # Calculate overall success
                    overall_success_rate = (total_endpoints_passed / total_endpoints_tested * 100) if total_endpoints_tested > 0 else 0

                    final_result = {
                        "success": failed == 0,
                        "message": f"Validation completed: {passed} passed, {failed} failed",
                        "summary": {
                            "total_services": len(service_ids),
                            "passed_services": passed,
                            "failed_services": failed,
                            "total_endpoints_tested": total_endpoints_tested,
                            "total_endpoints_passed": total_endpoints_passed,
                            "total_endpoints_failed": total_endpoints_failed,
                            "overall_success_rate": f"{overall_success_rate:.1f}%"
                        },
                        "results": processed_results,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Mark task as completed
                    task = task_mgr.get_task(tracking_id)
                    if task:
                        task_mgr._complete_task(task, final_result)
                    logger.info(f"✅ [{tracking_id}] Bulk validation completed: {passed}/{len(service_ids)} services passed")

            except Exception as e:
                logger.error(f"❌ [{tracking_id}] Bulk validation failed: {e}")
                task = task_mgr.get_task(tracking_id)
                if task:
                    task_mgr._fail_task(task, str(e))

        # Schedule background task
        background_tasks.add_task(run_validation)

        return {
            "tracking_id": tracking_id,
            "status": "pending",
            "message": f"Async bulk validation started for {len(service_ids)} services",
            "check_url": f"/tasks/{tracking_id}",
            "service_count": len(service_ids),
            "note": "Validations run in parallel for faster execution"
        }

    except Exception as e:
        logger.error(f"❌ Error in bulk validate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk/install")
async def bulk_install_venvs(request: BulkInstallRequest, manager = Depends(get_manager)):
    """
    Install virtual environments for multiple services

    Args:
        request: Installation configuration

    Returns:
        Results for each service
    """
    try:
        # If no service_ids specified, install all services with venv_path
        if request.service_ids is None:
            # Use the existing /venvs/install-all endpoint logic
            from pydantic import BaseModel as PydanticBaseModel

            # Create VenvInstallRequest
            venv_request = VenvInstallRequest(
                install_deps=request.install_deps,
                force=request.force
            )

            # Call existing install-all endpoint
            from src.core.service_manager.main import install_all_venvs
            return await install_all_venvs(venv_request)

        # Install specific services
        logger.info(f"🔨 Bulk installing venvs for {len(request.service_ids)} services")

        results = []
        installed = 0
        skipped = 0
        failed = 0

        for service_id in request.service_ids:
            try:
                # Check if service exists
                service_info = manager.execution_config.get_service_info(service_id)
                if not service_info:
                    results.append({
                        "service_id": service_id,
                        "status": "failed",
                        "message": "Service not found"
                    })
                    failed += 1
                    continue

                # Check if service has venv_path configured
                if not service_info.venv_path:
                    results.append({
                        "service_id": service_id,
                        "status": "skipped",
                        "message": "No venv_path configured"
                    })
                    skipped += 1
                    continue

                # Check if venv exists
                venv_exists = manager.venv_manager.venv_exists(service_id)

                if venv_exists and not request.force:
                    logger.info(f"✓ {service_id}: Venv already exists (skipping)")
                    skipped += 1
                    results.append({
                        "service_id": service_id,
                        "status": "skipped",
                        "message": "Venv already exists (use force=true to reinstall)"
                    })
                    continue

                # Remove if force and exists
                if request.force and venv_exists:
                    logger.info(f"🗑️  {service_id}: Force reinstall - removing existing venv")
                    if not manager.venv_manager.remove_venv(service_id):
                        failed += 1
                        results.append({
                            "service_id": service_id,
                            "status": "failed",
                            "message": "Failed to remove existing venv"
                        })
                        continue

                # Create venv
                logger.info(f"🔨 {service_id}: Creating venv...")
                if not manager.venv_manager.create_venv(service_id):
                    failed += 1
                    results.append({
                        "service_id": service_id,
                        "status": "failed",
                        "message": "Failed to create venv"
                    })
                    continue

                # Install requirements if requested
                install_msg = "Venv created successfully"
                if request.install_deps:
                    requirements_file = None
                    # Auto-detect requirements file
                    from pathlib import Path
                    req_file = Path(f"src/services/{service_id}/requirements.txt")
                    if req_file.exists():
                        requirements_file = req_file

                    if requirements_file and manager.venv_manager.install_requirements(service_id, requirements_file):
                        install_msg = "Venv created and requirements installed"
                    elif requirements_file:
                        install_msg = "Venv created (requirements installation failed)"

                installed += 1
                results.append({
                    "service_id": service_id,
                    "status": "installed",
                    "message": install_msg,
                    "venv_path": str(manager.venv_manager.get_venv_path(service_id))
                })
                logger.info(f"✅ {service_id}: {install_msg}")

            except Exception as e:
                failed += 1
                results.append({
                    "service_id": service_id,
                    "status": "failed",
                    "message": str(e)
                })
                logger.error(f"❌ {service_id}: {e}")

        return {
            "success": failed == 0,
            "message": f"Completed: {installed} installed, {skipped} skipped, {failed} failed",
            "total": len(request.service_ids) if request.service_ids else len(results),
            "installed": installed,
            "skipped": skipped,
            "failed": failed,
            "results": results
        }

    except Exception as e:
        logger.error(f"❌ Error in bulk venv install: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Individual Service Operations
# ============================================================================

@router.get("/{service_id}")
async def get_service(service_id: str, manager = Depends(get_manager)):
    """Obtém status de um serviço específico"""
    if service_id not in manager.services:
        raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

    status = manager.get_all_status()
    return status.get(service_id)

@router.post("/{service_id}/start")
async def start_service(service_id: str, request: ServiceRequest = ServiceRequest(), background_tasks: BackgroundTasks = BackgroundTasks(), manager = Depends(get_manager)):
    """Inicia um serviço de forma assíncrona com tracking"""
    if service_id not in manager.services and not manager._is_internal_service(service_id):
        raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

    # Check if service is enabled (unless force flag is set)
    service_info = manager.execution_config.get_service_info(service_id)
    if service_info and not request.force and not getattr(service_info, 'enabled', True):
        reason = getattr(service_info, 'reason_disabled', 'Service is disabled')
        logger.warning(f"⚠️  Attempt to start DISABLED service {service_id}: {reason}")
        raise HTTPException(
            status_code=403,
            detail=f"Service '{service_id}' is disabled: {reason}. Use force=true to override."
        )

    # Initialize start_tasks dict if not exists
    if not hasattr(manager, 'start_tasks'):
        manager.start_tasks = {}

    # Generate tracking ID
    tracking_id = str(uuid.uuid4())

    # Create task info
    manager.start_tasks[tracking_id] = {
        "service_id": service_id,
        "status": "pending",
        "completed": False,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "start_result": None,
        "error": None,
        "gpu_cleaned": False
    }

    # Background task to start service
    async def start_task() -> Any:
        try:
            manager.start_tasks[tracking_id]["status"] = "in_progress"

            # Clean GPU memory before starting (like GPU manager does)
            try:
                from src.core.gpu_memory_manager import get_gpu_manager
                gpu_mgr = get_gpu_manager()
                logger.info(f"🧹 Cleaning GPU memory before starting {service_id}...")
                gpu_info_before = gpu_mgr.get_gpu_info()
                freed_mb = 0
                if gpu_info_before:
                    gpu_mgr.cleanup_gpu_memory(level="soft")
                    gpu_info_after = gpu_mgr.get_gpu_info()
                    if gpu_info_after:
                        freed_mb = gpu_info_after.free_mb - gpu_info_before.free_mb
                        logger.info(f"✅ GPU cleanup completed - freed {freed_mb}MB")
                        manager.start_tasks[tracking_id]["gpu_cleaned"] = True
                        manager.start_tasks[tracking_id]["gpu_freed_mb"] = freed_mb
            except Exception as gpu_error:
                logger.warning(f"⚠️  GPU cleanup failed: {gpu_error}")
                # Continue anyway - GPU cleanup is not critical

            # Start service (async version handles both internal and external)
            start_result = await manager.start_service_async(service_id, request.force)

            manager.start_tasks[tracking_id]["start_result"] = start_result

            # Mark as completed
            manager.start_tasks[tracking_id]["status"] = "completed"
            manager.start_tasks[tracking_id]["completed"] = True
            manager.start_tasks[tracking_id]["end_time"] = datetime.now().isoformat()

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()

            # Log full error with traceback
            logger.error(f"❌ Failed to start {service_id}: {e}")
            logger.error(f"Full traceback:\n{error_traceback}")

            # Store in task info
            manager.start_tasks[tracking_id]["status"] = "error"
            manager.start_tasks[tracking_id]["error"] = str(e)
            manager.start_tasks[tracking_id]["error_traceback"] = error_traceback
            manager.start_tasks[tracking_id]["completed"] = True
            manager.start_tasks[tracking_id]["end_time"] = datetime.now().isoformat()

    # Add the start task to background
    background_tasks.add_task(start_task)

    return {
        "task_id": tracking_id,
        "status": "pending",
        "message": f"Start task for {service_id} queued",
        "check_url": f"/services/{service_id}/start/status/{tracking_id}"
    }

@router.get("/{service_id}/start/status/{task_id}")
async def get_start_status(service_id: str, task_id: str, manager = Depends(get_manager)):
    """Check status of a start task"""
    if not hasattr(manager, 'start_tasks') or task_id not in manager.start_tasks:
        raise HTTPException(status_code=404, detail=f"Start task {task_id} not found")

    task_info = manager.start_tasks[task_id]

    # Verify service ID matches
    if task_info["service_id"] != service_id:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found for service {service_id}")

    return task_info

@router.post("/{service_id}/stop")
async def stop_service(service_id: str, manager = Depends(get_manager)):
    """Para um serviço"""
    if service_id not in manager.services:
        raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

    result = manager.stop_service(service_id)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))

    return result

@router.post("/{service_id}/reload")
async def reload_service(service_id: str, manager = Depends(get_manager)):
    """Reload a service (hot-reload without restart) - works for both internal and external services"""
    if service_id not in manager.services:
        raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

    # Check if internal or external
    is_internal = manager._is_internal_service(service_id)

    if is_internal:
        # Internal service: use hot reload (reloads Python modules)
        success = await manager._reload_internal_service(service_id)

        if success:
            return {
                "success": True,
                "message": f"Service {service_id} reloaded successfully (hot reload)",
                "execution_type": "internal",
                "reload_method": "module_reload"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reload service {service_id}"
            )
    else:
        # External service: restart the process (acts as "reload")
        logger.info(f"🔥 External service {service_id}: using restart instead of hot reload")

        # Stop the service
        stop_result = manager.stop_service(service_id)
        if not stop_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to stop service {service_id}: {stop_result.get('error')}"
            )

        # Wait a moment
        await asyncio.sleep(2)

        # Start the service (force=True to kill any existing process from failed stop)
        start_result = manager.start_service(service_id, force=True)
        if not start_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to restart service {service_id}: {start_result.get('error')}"
            )

        return {
            "success": True,
            "message": f"Service {service_id} reloaded successfully (restart)",
            "execution_type": "external",
            "reload_method": "process_restart",
            "stop_result": stop_result,
            "start_result": start_result
        }

@router.post("/{service_id}/restart")
async def restart_service(service_id: str, background_tasks: BackgroundTasks, request: ServiceRequest = ServiceRequest(), manager = Depends(get_manager)):
    """Reinicia um serviço de forma assíncrona - retorna tracking ID imediatamente"""
    if service_id not in manager.services:
        raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

    # Generate tracking ID
    import uuid
    tracking_id = str(uuid.uuid4())

    # Initialize tracking entry
    if not hasattr(manager, 'restart_tasks'):
        manager.restart_tasks = {}

    manager.restart_tasks[tracking_id] = {
        "service_id": service_id,
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "stop_result": None,
        "start_result": None,
        "completed": False
    }

    async def restart_task() -> Any:
        """Background task to restart the service"""
        try:
            # Check if this is an internal service
            is_internal = manager._is_internal_service(service_id)

            # Update status
            manager.restart_tasks[tracking_id]["status"] = "stopping"

            # Stop the service
            if is_internal:
                # For internal services, use async stop
                stop_success = await manager._stop_internal_service(service_id)
                stop_result = {
                    "success": stop_success,
                    "status": "stopped" if stop_success else "error",
                    "message": f"{service_id} {'stopped' if stop_success else 'failed to stop'} (internal)"
                }
            else:
                # For external services, use sync stop
                stop_result = manager.stop_service(service_id)

            manager.restart_tasks[tracking_id]["stop_result"] = stop_result

            # Wait a bit
            await asyncio.sleep(2)

            # Update status
            manager.restart_tasks[tracking_id]["status"] = "starting"

            # Start the service
            if is_internal:
                # For internal services, use async start
                start_success = await manager._start_local_service(service_id, manager.app)
                start_result = {
                    "success": start_success,
                    "status": "running" if start_success else "error",
                    "message": f"{service_id} {'started' if start_success else 'failed to start'} (internal)",
                    "execution_type": "internal"
                }
            else:
                # For external services, use sync start
                start_result = manager.start_service(service_id, request.force)

            manager.restart_tasks[tracking_id]["start_result"] = start_result

            # Mark as completed
            manager.restart_tasks[tracking_id]["status"] = "completed"
            manager.restart_tasks[tracking_id]["completed"] = True
            manager.restart_tasks[tracking_id]["end_time"] = datetime.now().isoformat()

        except Exception as e:
            manager.restart_tasks[tracking_id]["status"] = "error"
            manager.restart_tasks[tracking_id]["error"] = str(e)
            manager.restart_tasks[tracking_id]["completed"] = True
            manager.restart_tasks[tracking_id]["end_time"] = datetime.now().isoformat()

    # Add the restart task to background
    background_tasks.add_task(restart_task)

    return {
        "tracking_id": tracking_id,
        "status": "pending",
        "message": f"Restart task for {service_id} queued",
        "check_url": f"/services/{service_id}/restart/status/{tracking_id}"
    }

@router.get("/{service_id}/restart/status/{tracking_id}")
async def get_restart_status(service_id: str, tracking_id: str, manager = Depends(get_manager)):
    """Check status of a restart task"""
    if not hasattr(manager, 'restart_tasks') or tracking_id not in manager.restart_tasks:
        raise HTTPException(status_code=404, detail=f"Restart task {tracking_id} not found")

    task_info = manager.restart_tasks[tracking_id]

    # Verify service ID matches
    if task_info["service_id"] != service_id:
        raise HTTPException(status_code=404, detail=f"Task {tracking_id} not found for service {service_id}")

    return task_info


@router.post("/{service_id}/deploy-remote")
async def deploy_remote_service(
    service_id: str,
    background_tasks: BackgroundTasks,
    create_tunnel: bool = True,
    manager = Depends(get_manager)
):
    """
    Deploy a remote service on RunPod Pod

    Steps:
    1. Verify service is remote (has runpod_pod_id)
    2. Create SSH tunnel (if requested)
    3. Commit + push local code
    4. Start Pod (if stopped)
    5. Git pull on Pod
    6. Start service on Pod
    7. Health check
    8. Register with Service Discovery

    Args:
        service_id: Service identifier (e.g., 'llm')
        create_tunnel: Create SSH tunnel for communication (default: True)

    Returns:
        Deploy task tracking ID
    """
    # Get service config
    service_info = manager.execution_config.get_service_info(service_id)

    if not service_info:
        raise HTTPException(status_code=404, detail=f"Service {service_id} not found in configuration")

    # Verify it's a remote service
    if not service_info.runpod_pod_id:
        raise HTTPException(
            status_code=400,
            detail=f"Service {service_id} is not a remote service (no runpod_pod_id configured)"
        )

    # Generate tracking ID
    import uuid
    tracking_id = str(uuid.uuid4())

    # Initialize tracking entry
    if not hasattr(manager, 'deploy_tasks'):
        manager.deploy_tasks = {}

    manager.deploy_tasks[tracking_id] = {
        "service_id": service_id,
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "steps_completed": [],
        "steps_failed": [],
        "tunnel_created": False,
        "pod_started": False,
        "code_synced": False,
        "service_started": False,
        "health_checked": False,
        "completed": False,
        "error": None
    }

    async def deploy_task() -> Any:
        """Background task to deploy remote service"""
        try:
            task = manager.deploy_tasks[tracking_id]
            task["status"] = "in_progress"

            # Step 1: Create SSH tunnel (if requested)
            if create_tunnel:
                logger.info(f"🔗 [{service_id}] Creating SSH tunnel...")
                task["status"] = "creating_tunnel"

                try:
                    from src.core.ssh_tunnel_manager import get_tunnel_manager
                    from src.services.runpod_llm.config import load_runpod_config

                    tunnel_manager = get_tunnel_manager()
                    runpod_config = load_runpod_config()

                    tunnel_created = tunnel_manager.create_tunnel(
                        name=service_id,
                        local_port=service_info.remote_port,
                        remote_port=service_info.remote_port,
                        ssh_user=runpod_config.pod.ssh_user
                    )

                    if tunnel_created:
                        task["tunnel_created"] = True
                        task["steps_completed"].append("tunnel_created")
                        logger.info(f"✅ SSH tunnel created: localhost:{service_info.remote_port}")
                    else:
                        logger.warning(f"⚠️  Failed to create SSH tunnel, but continuing...")
                        task["steps_failed"].append("tunnel_creation_failed")

                except Exception as e:
                    logger.warning(f"⚠️  SSH tunnel error: {e}")
                    task["steps_failed"].append(f"tunnel_error: {str(e)}")

            # Step 2: Deploy service using Remote Launcher
            logger.info(f"🚀 [{service_id}] Deploying remote service...")
            task["status"] = "deploying"

            # Initialize remote launcher if needed
            if not hasattr(manager, 'remote_launcher') or manager.remote_launcher is None:
                from src.core.service_manager.remote_launcher import get_remote_launcher
                manager.remote_launcher = await get_remote_launcher()

            # Launch service (this handles commit, push, pod start, git pull, setup, start)
            result = await manager.remote_launcher.launch_service(service_info)

            if result.success:
                task["pod_started"] = result.pod_started
                task["code_synced"] = True  # Git sync is done in launch_service
                task["service_started"] = result.command_executed
                task["service_url"] = result.service_url
                task["steps_completed"].extend([
                    "pod_started" if result.pod_started else "pod_already_running",
                    "code_synced",
                    "service_started"
                ])
                logger.info(f"✅ [{service_id}] Service deployed successfully")
            else:
                task["error"] = result.error_message
                task["steps_failed"].append(f"deployment_failed: {result.error_message}")
                logger.error(f"❌ [{service_id}] Deployment failed: {result.error_message}")

            # Mark as completed
            task["status"] = "completed" if result.success else "failed"
            task["completed"] = True
            task["end_time"] = datetime.now().isoformat()
            task["deploy_result"] = {
                "success": result.success,
                "service_url": result.service_url,
                "pod_started": result.pod_started,
                "command_executed": result.command_executed
            }

        except Exception as e:
            logger.error(f"❌ [{service_id}] Deploy task error: {e}", exc_info=True)
            task["status"] = "error"
            task["error"] = str(e)
            task["completed"] = True
            task["end_time"] = datetime.now().isoformat()

    # Add deploy task to background
    background_tasks.add_task(deploy_task)

    return {
        "task_id": tracking_id,
        "status": "pending",
        "service_id": service_id,
        "message": f"Remote deploy task for {service_id} queued",
        "check_url": f"/services/{service_id}/deploy-remote/status/{tracking_id}",
        "estimated_duration_seconds": 180  # ~3 minutes
    }


@router.get("/{service_id}/deploy-remote/status/{tracking_id}")
async def get_deploy_status(service_id: str, tracking_id: str, manager = Depends(get_manager)):
    """Check status of a remote deploy task"""
    if not hasattr(manager, 'deploy_tasks') or tracking_id not in manager.deploy_tasks:
        raise HTTPException(status_code=404, detail=f"Deploy task {tracking_id} not found")

    task_info = manager.deploy_tasks[tracking_id]

    # Verify service ID matches
    if task_info["service_id"] != service_id:
        raise HTTPException(status_code=404, detail=f"Task {tracking_id} not found for service {service_id}")

    return task_info

# ============================================================================
# Start/Stop All Status Endpoints
# ============================================================================

@router.get("/start-all/{tracking_id}")
async def get_startup_status(tracking_id: str, manager = Depends(get_manager)):
    """Get status of a service startup operation"""
    if tracking_id not in manager.startup_tasks:
        raise HTTPException(status_code=404, detail=f"Startup task {tracking_id} not found")

    task = manager.startup_tasks[tracking_id]

    response = {
        "tracking_id": task["tracking_id"],
        "status": task["status"],
        "created_at": task["created_at"]
    }

    if "started_at" in task:
        response["started_at"] = task["started_at"]
    if "completed_at" in task:
        response["completed_at"] = task["completed_at"]
    if "results" in task:
        response["results"] = task["results"]
    if "error" in task:
        response["error"] = task["error"]

    return response

@router.get("/stop-all/{tracking_id}")
async def get_shutdown_status(tracking_id: str, manager = Depends(get_manager)):
    """Get status of a service shutdown operation"""
    if tracking_id not in manager.shutdown_tasks:
        raise HTTPException(status_code=404, detail=f"Shutdown task {tracking_id} not found")

    task = manager.shutdown_tasks[tracking_id]

    response = {
        "tracking_id": task["tracking_id"],
        "status": task["status"],
        "created_at": task["created_at"]
    }

    if "started_at" in task:
        response["started_at"] = task["started_at"]
    if "completed_at" in task:
        response["completed_at"] = task["completed_at"]
    if "results" in task:
        response["results"] = task["results"]
    if "error" in task:
        response["error"] = task["error"]

    return response

# ============================================================================
# Virtual Environment Management Endpoints
# ============================================================================

@router.post("/{service_id}/venv/install")
async def install_service_venv(service_id: str, request: VenvInstallRequest, manager = Depends(get_manager)):
    """
    Create virtual environment and install dependencies for a service

    Args:
        service_id: Service identifier (e.g., "llm", "stt", "tts")
        request: Installation configuration

    Returns:
        Installation status and details
    """
    try:
        # Check if service exists in configuration
        service_info = manager.execution_config.get_service_info(service_id)
        if not service_info:
            raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found in configuration")

        # Check if service has venv configured
        venv_path = manager.execution_config.get_venv_path(service_id)
        if not venv_path:
            raise HTTPException(
                status_code=400,
                detail=f"Service '{service_id}' does not have venv_path configured. Update config/service_execution.yaml first."
            )

        logger.info(f"🔨 Installing venv for service: {service_id}")

        # If force=true, remove existing venv first
        if request.force and manager.venv_manager.venv_exists(service_id):
            logger.info(f"🗑️  Force reinstall: Removing existing venv for {service_id}")
            if not manager.venv_manager.remove_venv(service_id):
                raise HTTPException(status_code=500, detail=f"Failed to remove existing venv for {service_id}")

        # Create venv
        if not manager.venv_manager.create_venv(service_id):
            raise HTTPException(status_code=500, detail=f"Failed to create virtual environment for {service_id}")

        # Install requirements if requested
        install_output = None
        if request.install_deps:
            requirements_file = None
            if request.requirements_file:
                requirements_file = Path(request.requirements_file)
                if not requirements_file.exists():
                    raise HTTPException(status_code=404, detail=f"Requirements file not found: {request.requirements_file}")

            # Install requirements
            if not manager.venv_manager.install_requirements(service_id, requirements_file):
                logger.warning(f"⚠️  Failed to install requirements for {service_id}, but venv was created")
                install_output = "Warning: Failed to install requirements, but venv was created successfully"
            else:
                install_output = "Requirements installed successfully"

        python_exe = manager.venv_manager.get_python_executable(service_id)

        return {
            "success": True,
            "service_id": service_id,
            "venv_path": str(manager.venv_manager.get_venv_path(service_id)),
            "python_executable": str(python_exe) if python_exe else None,
            "install_output": install_output,
            "message": f"Virtual environment for {service_id} installed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error installing venv for {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{service_id}/install")
async def run_service_install_script(service_id: str, background_tasks: BackgroundTasks, manager = Depends(get_manager)):
    """
    Run the service's custom install.py script

    This endpoint executes the install.py located in src/services/{service_id}/install.py
    The script runs asynchronously in the background and returns a tracking ID immediately
    Logs are written to os.path.expanduser("~/.cache/ultravox-pipeline/")logs/{service_id}_install_{track_id}.log

    Args:
        service_id: Service identifier (e.g., "llm", "stt", "tts")

    Returns:
        Installation tracking ID, status and log file location
    """
    import uuid

    try:
        # Validate service_id to prevent path traversal
        service_id = validate_service_id(service_id)

        # Check if service exists
        service_info = manager.execution_config.get_service_info(service_id)
        if not service_info:
            raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found in configuration")

        # Check if install.py exists
        install_script = Path(f"src/services/{service_id}/install.py")
        if not install_script.exists():
            raise HTTPException(
                status_code=404,
                detail=f"install.py not found for service '{service_id}' at {install_script}"
            )

        # Generate unique tracking ID
        track_id = str(uuid.uuid4())[:8]

        # Create log file path with tracking ID (using portable cache directory)
        log_dir = Path.home() / ".cache" / "ultravox-pipeline" / "logs"
        log_file = log_dir / f"{service_id}_install_{track_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Running install script for {service_id} [track_id: {track_id}]")
        logger.info(f"📝 Logs will be written to: {log_file}")

        # Run install script in background
        def run_install() -> Any:
            try:
                result = subprocess.run(
                    ["python3", str(install_script)],
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes timeout
                )

                # Write output to log file
                with open(log_file, 'w') as f:
                    f.write(f"=== Installation for {service_id} ===\n")
                    f.write(f"Track ID: {track_id}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Return code: {result.returncode}\n\n")
                    f.write("=== STDOUT ===\n")
                    f.write(result.stdout)
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)

                if result.returncode == 0:
                    logger.info(f"✅ Installation completed successfully for {service_id} [track_id: {track_id}]")
                else:
                    logger.error(f"❌ Installation failed for {service_id} [track_id: {track_id}] with code {result.returncode}")

            except subprocess.TimeoutExpired:
                logger.error(f"⏱️  Installation timeout for {service_id} [track_id: {track_id}]")
                with open(log_file, 'a') as f:
                    f.write("\n\n=== ERROR ===\nInstallation timed out after 10 minutes\n")
            except Exception as e:
                logger.error(f"❌ Installation error for {service_id} [track_id: {track_id}]: {e}")
                with open(log_file, 'a') as f:
                    f.write(f"\n\n=== ERROR ===\n{str(e)}\n")

        # Add to background tasks
        background_tasks.add_task(run_install)

        return {
            "success": True,
            "track_id": track_id,
            "service_id": service_id,
            "message": f"Installation started for {service_id}",
            "log_file": str(log_file),
            "install_script": str(install_script),
            "status": "running"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting install for {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{service_id}/venv/status")
async def get_venv_status(service_id: str, manager = Depends(get_manager)):
    """
    Get virtual environment status for a service

    Args:
        service_id: Service identifier

    Returns:
        VenvStatusResponse with venv details
    """
    try:
        # Check if service exists
        service_info = manager.execution_config.get_service_info(service_id)
        if not service_info:
            raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")

        venv_path = manager.execution_config.get_venv_path(service_id)
        exists = manager.venv_manager.venv_exists(service_id)

        python_exe = None
        installed_packages = None

        if exists:
            python_exe = manager.venv_manager.get_python_executable(service_id)

            # Get installed packages
            if python_exe:
                try:
                    result = subprocess.run(
                        [str(python_exe), "-m", "pip", "list", "--format=freeze"],
                        capture_output=True,
                        text=True,
                        timeout=30  # Increased from 10 to 30 for large venvs
                    )
                    if result.returncode == 0:
                        installed_packages = result.stdout.strip().split('\n')
                except Exception as e:
                    logger.warning(f"⚠️  Failed to get package list: {e}")

        # Find requirements file
        requirements_file = None
        service_dir = Path(__file__).parent.parent.parent / "services" / service_id
        req_file = service_dir / "requirements.txt"
        if req_file.exists():
            requirements_file = str(req_file)

        return VenvStatusResponse(
            service_id=service_id,
            exists=exists,
            venv_path=str(manager.venv_manager.get_venv_path(service_id)) if venv_path else None,
            python_executable=str(python_exe) if python_exe else None,
            installed_packages=installed_packages,
            requirements_file=requirements_file
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting venv status for {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{service_id}/venv")
async def delete_service_venv(service_id: str, force: bool = False, manager = Depends(get_manager)):
    """
    Remove virtual environment for a service

    Args:
        service_id: Service identifier
        force: Force deletion even if service is running

    Returns:
        Deletion status
    """
    try:
        # Check if service exists
        service_info = manager.execution_config.get_service_info(service_id)
        if not service_info:
            raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")

        # Check if venv exists
        if not manager.venv_manager.venv_exists(service_id):
            return {
                "success": True,
                "message": f"No virtual environment found for {service_id}",
                "deleted": False
            }

        # Check if service is running (optional safety check)
        if not force and service_id in manager.services:
            service = manager.services[service_id]
            if service.process_status == ProcessStatus.RUNNING:
                raise HTTPException(
                    status_code=400,
                    detail=f"Service '{service_id}' is running. Stop the service first or use force=true"
                )

        logger.info(f"🗑️  Removing venv for service: {service_id}")

        if manager.venv_manager.remove_venv(service_id):
            return {
                "success": True,
                "service_id": service_id,
                "message": f"Virtual environment for {service_id} removed successfully",
                "deleted": True
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to remove virtual environment for {service_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error removing venv for {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{service_id}/venv/packages")
async def install_packages(service_id: str, request: PackageInstallRequest, manager = Depends(get_manager)):
    """
    Install specific packages in service's virtual environment

    Args:
        service_id: Service identifier
        request: Packages to install

    Returns:
        Installation result
    """
    try:
        # Check if service exists
        service_info = manager.execution_config.get_service_info(service_id)
        if not service_info:
            raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")

        # Check if venv exists
        if not manager.venv_manager.venv_exists(service_id):
            raise HTTPException(
                status_code=404,
                detail=f"No virtual environment found for {service_id}. Create it first with POST /services/{service_id}/venv/install"
            )

        python_exe = manager.venv_manager.get_python_executable(service_id)
        if not python_exe:
            raise HTTPException(status_code=500, detail=f"Could not find Python executable for {service_id}")

        logger.info(f"📦 Installing packages for {service_id}: {', '.join(request.packages)}")

        # Install packages
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "install"] + request.packages,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to install packages: {result.stderr}"
            )

        return {
            "success": True,
            "service_id": service_id,
            "packages": request.packages,
            "output": result.stdout,
            "message": f"Successfully installed {len(request.packages)} package(s)"
        }

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Package installation timed out after 5 minutes")
    except Exception as e:
        logger.error(f"❌ Error installing packages for {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Metrics & Monitoring Endpoints
# ============================================================================

@router.get("/{service_id}/metrics")
async def get_service_metrics(service_id: str, manager = Depends(get_manager)):
    """
    Get real-time metrics for a specific service

    Returns:
        CPU, memory, uptime, health status, request counts
    """
    try:
        if service_id not in manager.services:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

        service = manager.services[service_id]

        # Check if service is running
        if service.process_status != ProcessStatus.RUNNING:
            return {
                "service_id": service_id,
                "running": False,
                "message": f"Service is {service.process_status.value}"
            }

        # Check if internal or external service
        is_internal = manager._is_internal_service(service_id)

        if is_internal:
            # Collect metrics for internal service
            metrics = manager.metrics_tracker.collect_internal_service_metrics(service_id)
        else:
            # Collect metrics for external service (has PID)
            if not service.pid:
                return {
                    "service_id": service_id,
                    "running": True,
                    "message": "No PID available for metrics collection"
                }

            metrics = manager.metrics_tracker.collect_process_metrics(service_id, service.pid)

            if not metrics:
                raise HTTPException(status_code=500, detail="Failed to collect metrics (process may have stopped)")

        return metrics.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting metrics for {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/summary")
async def get_all_services_metrics(manager = Depends(get_manager)):
    """
    Get summary of metrics for all running services

    Returns:
        Total resource usage, per-service metrics, health breakdown
    """
    try:
        # Collect metrics for all running services
        for service_id, service in manager.services.items():
            if service.process_status == ProcessStatus.RUNNING:
                is_internal = manager._is_internal_service(service_id)

                if is_internal:
                    manager.metrics_tracker.collect_internal_service_metrics(service_id)
                elif service.pid:
                    manager.metrics_tracker.collect_process_metrics(service_id, service.pid)

        # Get summary
        summary = manager.metrics_tracker.get_all_metrics_summary()

        return summary

    except Exception as e:
        logger.error(f"❌ Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{service_id}/health/history")
async def get_service_health_history(service_id: str, limit: int = 100, manager = Depends(get_manager)):
    """
    Get health check history for a service

    Args:
        service_id: Service identifier
        limit: Maximum number of entries (default 100)

    Returns:
        List of health check entries (most recent first)
    """
    try:
        if service_id not in manager.services:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

        history = manager.metrics_tracker.get_health_history(service_id, limit=limit)

        return {
            "service_id": service_id,
            "total_entries": len(history),
            "limit": limit,
            "history": history
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting health history for {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Service Uninstall Endpoint
# ============================================================================

@router.delete("/{service_id}/uninstall")
async def uninstall_service(service_id: str, manager = Depends(get_manager)):
    """
    Uninstall service (delete tmp/ directory with models, venv, cache)

    This removes the tmp/ directory within the service directory, which typically contains:
    - Downloaded models (tmp/models/)
    - Virtual environment (tmp/venv/)
    - Cache files
    - Temporary data

    Args:
        service_id: Service identifier

    Returns:
        Uninstall status with size information
    """
    try:
        import shutil

        # Validate service_id to prevent path traversal
        service_id = validate_service_id(service_id)

        # Check if service exists in configuration
        service_info = manager.execution_config.get_service_info(service_id)
        if not service_info:
            raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found in configuration")

        # Find service directory
        service_dir = Path(__file__).parent.parent.parent.parent / "services" / service_id
        tmp_dir = service_dir / "tmp"

        if not tmp_dir.exists():
            return {
                "success": True,
                "service_id": service_id,
                "message": f"No tmp directory found for {service_id}",
                "deleted": False,
                "size_freed": "0B"
            }

        # Get size before deletion
        try:
            result = subprocess.run(
                ['du', '-sh', str(tmp_dir)],
                capture_output=True,
                text=True,
                timeout=30  # Increased from 10 to 30 for large directories
            )
            size_freed = result.stdout.split()[0] if result.returncode == 0 else "unknown"
        except (ValueError, KeyError, RuntimeError):
            size_freed = "unknown"

        logger.info(f"🗑️  Uninstalling service {service_id} - removing tmp directory ({size_freed})")

        # Delete tmp directory
        try:
            shutil.rmtree(tmp_dir)
            logger.info(f"✅ Service {service_id} uninstalled - deleted {tmp_dir}")

            return {
                "success": True,
                "service_id": service_id,
                "message": f"Service {service_id} uninstalled successfully - tmp directory removed",
                "deleted": True,
                "size_freed": size_freed,
                "path": str(tmp_dir)
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to uninstall service - could not delete tmp directory: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uninstalling service {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
