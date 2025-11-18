#!/usr/bin/env python3
"""
Router Manager Module

Handles FastAPI route mounting and organization.
Extracted from ServiceManager to improve modularity.

Responsibilities:
- Mount service routers to main FastAPI app
- Handle route prefixes and tags
- Manage route conflicts
- Provide route introspection
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, APIRouter
from loguru import logger


class RouterManager:
    """
    Router Manager - Manages FastAPI route mounting

    Handles:
    - Mounting service routers with prefixes
    - Route conflict detection
    - Route documentation and introspection
    """

    def __init__(self):
        """Initialize Router Manager"""
        self.mounted_routes = {}  # Track mounted routes by service_id
        self.route_registry = []  # All routes for introspection

        logger.info("🛤️  Router Manager initialized")

    def mount_service_router(
        self,
        app: FastAPI,
        service_id: str,
        service_instance: Any,
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Mount service router to FastAPI app

        Args:
            app: FastAPI application instance
            service_id: Service identifier
            service_instance: Service instance with get_router() method
            prefix: Route prefix (default: /api/{service_id})
            tags: OpenAPI tags (default: [service_id])

        Returns:
            True if mounting successful
        """
        try:
            if not hasattr(service_instance, 'get_router'):
                logger.warning(f"⚠️  {service_id} has no get_router() method")
                return False

            # Get router from service
            router = service_instance.get_router()

            if not isinstance(router, APIRouter):
                logger.error(f"❌ {service_id}.get_router() did not return APIRouter")
                return False

            # Use default prefix if not provided
            if prefix is None:
                prefix = f"/api/{service_id}"

            # Use default tags if not provided
            if tags is None:
                tags = [service_id]

            # Check for route conflicts
            conflict = self._check_route_conflict(prefix)
            if conflict:
                logger.error(
                    f"❌ Route conflict: {prefix} already mounted by {conflict}",
                    service=service_id
                )
                return False

            # Mount router
            app.include_router(router, prefix=prefix, tags=tags)

            # Track mounted route
            self.mounted_routes[service_id] = {
                "prefix": prefix,
                "tags": tags,
                "route_count": len(router.routes)
            }

            # Register routes for introspection
            for route in router.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    for method in route.methods:
                        self.route_registry.append({
                            "service": service_id,
                            "path": f"{prefix}{route.path}",
                            "method": method,
                            "name": route.name if hasattr(route, 'name') else None
                        })

            logger.info(
                f"✅ Mounted {service_id} routes",
                prefix=prefix,
                routes=len(router.routes),
                tags=tags
            )

            return True

        except Exception as e:
            logger.error(f"❌ Failed to mount {service_id} routes: {e}")
            import traceback
            traceback.print_exc()
            return False

    def unmount_service_router(
        self,
        app: FastAPI,
        service_id: str
    ) -> bool:
        """
        Unmount service router from FastAPI app

        Note: FastAPI doesn't support dynamic route removal,
        so this marks the service as unmounted but routes remain.

        Args:
            app: FastAPI application instance
            service_id: Service identifier

        Returns:
            True if unmounting successful
        """
        try:
            if service_id not in self.mounted_routes:
                logger.warning(f"⚠️  {service_id} not mounted, cannot unmount")
                return False

            # Remove from tracking
            route_info = self.mounted_routes.pop(service_id)

            # Remove from registry
            self.route_registry = [
                r for r in self.route_registry
                if r["service"] != service_id
            ]

            logger.info(
                f"🗑️  Unmounted {service_id} routes (marked as inactive)",
                prefix=route_info["prefix"]
            )

            logger.warning(
                "⚠️  Note: FastAPI routes remain active (requires app restart to fully remove)"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Failed to unmount {service_id} routes: {e}")
            return False

    def mount_all_services(
        self,
        app: FastAPI,
        service_instances: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Mount all service routers

        Args:
            app: FastAPI application instance
            service_instances: Dictionary of service instances

        Returns:
            Dict mapping service_id -> success status
        """
        results = {}

        logger.info(f"🛤️  Mounting {len(service_instances)} service routers")

        for service_id, service_instance in service_instances.items():
            success = self.mount_service_router(app, service_id, service_instance)
            results[service_id] = success

        success_count = sum(1 for r in results.values() if r)
        logger.info(f"✅ Mounted {success_count}/{len(service_instances)} routers successfully")

        return results

    def _check_route_conflict(self, prefix: str) -> Optional[str]:
        """
        Check if prefix conflicts with existing routes

        Args:
            prefix: Route prefix to check

        Returns:
            Service ID of conflicting route, or None if no conflict
        """
        for service_id, route_info in self.mounted_routes.items():
            if route_info["prefix"] == prefix:
                return service_id

        return None

    def get_mounted_services(self) -> List[str]:
        """
        Get list of mounted service IDs

        Returns:
            List of service identifiers
        """
        return list(self.mounted_routes.keys())

    def get_all_routes(self, service_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all registered routes

        Args:
            service_id: Optional filter by service

        Returns:
            List of route dictionaries
        """
        if service_id:
            return [r for r in self.route_registry if r["service"] == service_id]
        return self.route_registry.copy()

    def get_route_summary(self) -> Dict[str, Any]:
        """
        Get summary of all mounted routes

        Returns:
            Summary dictionary with statistics
        """
        services = {}

        for service_id, route_info in self.mounted_routes.items():
            # Count routes by method
            service_routes = [r for r in self.route_registry if r["service"] == service_id]
            methods = {}
            for route in service_routes:
                method = route["method"]
                methods[method] = methods.get(method, 0) + 1

            services[service_id] = {
                "prefix": route_info["prefix"],
                "total_routes": len(service_routes),
                "methods": methods,
                "tags": route_info["tags"]
            }

        return {
            "total_services": len(self.mounted_routes),
            "total_routes": len(self.route_registry),
            "services": services
        }

    def find_route(self, path: str, method: str = "GET") -> Optional[Dict[str, Any]]:
        """
        Find route by path and method

        Args:
            path: Route path (e.g., "/api/session/create")
            method: HTTP method

        Returns:
            Route dictionary or None if not found
        """
        for route in self.route_registry:
            if route["path"] == path and route["method"] == method:
                return route

        return None

    def get_status(self) -> Dict[str, Any]:
        """
        Get router manager status

        Returns:
            Status dictionary
        """
        return {
            "mounted_services": len(self.mounted_routes),
            "total_routes": len(self.route_registry),
            "services": list(self.mounted_routes.keys())
        }
