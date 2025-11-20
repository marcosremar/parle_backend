"""
Common route helper functions to reduce duplication across services (DRY v5.2)

This module provides factory functions for standard endpoints that are
duplicated across multiple services (/validate, /health, /info).

Integrated with unified error handling system - uses typed exceptions instead of HTTPException.

Usage:
    from src.core.route_helpers import add_standard_endpoints

    router = APIRouter()
    add_standard_endpoints(router, service_instance, "my_service")
"""

from typing import Dict, Any, Callable, Optional
from datetime import datetime
from fastapi import APIRouter
from loguru import logger

# Use unified error handling instead of HTTPException
from .exceptions import ServiceUnavailableError


def create_validation_endpoint(service_name: str) -> Callable:
    """
    Factory to create /validate endpoint (DRY - eliminates duplication across services).

    Args:
        service_name: Name of the service for validation

    Returns:
        Async function that performs validation
    """

    async def validate_service() -> Dict[str, Any]:
        """Run validation tests using ValidationManager"""
        try:
            from src.lib.validation_manager import ValidationManager

            manager = ValidationManager(service_name=service_name)
            results = manager.run_pytest_tests(verbose=False)

            logger.info(
                "Validation completed",
                service=service_name,
                passed=results["summary"]["passed"],
                total=results["summary"]["total"],
            )

            return results
        except Exception as e:
            logger.error("Validation error", service=service_name, error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    return validate_service


def create_health_endpoint(service_instance) -> Callable:
    """
    Factory to create /health endpoint.

    Args:
        service_instance: Instance of the service (must have health_check method)

    Returns:
        Async function that performs health check

    Note:
        Uses ServiceUnavailableError (typed exception) instead of HTTPException.
        Exception is auto-converted to HTTP 503 by FastAPI error handlers.
    """

    async def health_check() -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            return await service_instance.health_check()
        except Exception as e:
            service_name = service_instance.__class__.__name__
            logger.error(
                "Health check failed",
                service=service_name,
                error=str(e),
            )
            # Raise typed exception - auto-converted to HTTP 503
            raise ServiceUnavailableError(
                service_name=service_name,
                original_error=e
            )

    return health_check


def create_info_endpoint(service_instance) -> Callable:
    """
    Factory to create /info endpoint with auto-discovery.

    Args:
        service_instance: Instance of the service

    Returns:
        Async function that returns service info
    """

    async def service_info() -> Dict[str, Any]:
        """Service information endpoint"""
        try:
            # Try to use service's own get_service_info if available
            if hasattr(service_instance, "get_service_info"):
                return service_instance.get_service_info()

            # Fallback to basic info
            return {
                "service": service_instance.__class__.__name__,
                "version": getattr(service_instance, "version", "1.0.0"),
                "description": service_instance.__doc__ or "No description",
                "status": "running",
            }
        except Exception as e:
            logger.error(
                "Failed to get service info",
                service=service_instance.__class__.__name__,
                error=str(e),
            )
            return {
                "service": service_instance.__class__.__name__,
                "status": "error",
                "error": str(e),
            }

    return service_info


def add_standard_endpoints(
    router: APIRouter,
    service_instance,
    service_name: Optional[str] = None,
    include_validate: bool = True,
    include_health: bool = True,
    include_info: bool = True,
) -> None:
    """
    Add standard endpoints (/validate, /health, /info) to a router.

    This eliminates duplication across 32+ services.

    Args:
        router: FastAPI router to add endpoints to
        service_instance: Instance of the service
        service_name: Name for validation (defaults to class name)
        include_validate: Whether to include /validate endpoint
        include_health: Whether to include /health endpoint
        include_info: Whether to include /info endpoint

    Example:
        router = APIRouter()
        add_standard_endpoints(router, self, "my_service")
    """
    if service_name is None:
        service_name = service_instance.__class__.__name__

    if include_validate:
        router.add_api_route(
            "/validate",
            create_validation_endpoint(service_name),
            methods=["GET"],
            tags=["validation"],
            summary="Run service validation tests",
            response_description="Validation results",
        )

    if include_health:
        router.add_api_route(
            "/health",
            create_health_endpoint(service_instance),
            methods=["GET"],
            tags=["health"],
            summary="Health check",
            response_description="Service health status",
        )

    if include_info:
        router.add_api_route(
            "/info",
            create_info_endpoint(service_instance),
            methods=["GET"],
            tags=["info"],
            summary="Service information",
            response_description="Service metadata and info",
        )


def get_standard_router(
    service_instance, service_name: Optional[str] = None
) -> APIRouter:
    """
    Create a new router with standard endpoints already included.

    Args:
        service_instance: Instance of the service
        service_name: Name for validation (defaults to class name)

    Returns:
        APIRouter with standard endpoints

    Example:
        # In service routes.py
        from src.core.route_helpers import get_standard_router

        def create_router(service_instance):
            router = get_standard_router(service_instance, "my_service")
            # Add custom routes
            return router
    """
    router = APIRouter()
    add_standard_endpoints(router, service_instance, service_name)
    return router
