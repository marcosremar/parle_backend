#!/usr/bin/env python3
"""
Route Discovery System

Automatically discovers service routes by scanning services directory
and extracting endpoints from routes.py files.

This enables:
- Auto-registration of routes in API Gateway
- Auto-generation of Service Manager Client
- Documentation generation
"""

import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EndpointMethod(Enum):
    """HTTP methods supported by endpoints"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


@dataclass
class EndpointInfo:
    """Information about a single endpoint"""
    path: str
    methods: Set[str]
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    requires_auth: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "path": self.path,
            "methods": list(self.methods),
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "requires_auth": self.requires_auth
        }


@dataclass
class ServiceRoutes:
    """Complete route information for a service"""
    service_id: str
    service_path: Path
    has_routes_file: bool
    has_service_file: bool
    endpoints: List[EndpointInfo] = field(default_factory=list)
    router_prefix: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "service_id": self.service_id,
            "service_path": str(self.service_path),
            "has_routes_file": self.has_routes_file,
            "has_service_file": self.has_service_file,
            "router_prefix": self.router_prefix,
            "description": self.description,
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "endpoint_count": len(self.endpoints)
        }


class RouteDiscovery:
    """
    Discovers and catalogs all service routes in the Ultravox Pipeline.

    Scans src/services/ directory and extracts route information from
    services that follow the standard pattern (have routes.py with create_router()).
    """

    def __init__(self, services_dir: Optional[Path] = None):
        """
        Initialize route discovery.

        Args:
            services_dir: Path to services directory (default: src/services)
        """
        # Determine project root
        self.project_root = Path(__file__).parent.parent.parent
        self.services_dir = services_dir or self.project_root / "src" / "services"

        # Registry of discovered services
        self.services: Dict[str, ServiceRoutes] = {}

        # Add project root to Python path for imports
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

    def discover_all_services(self) -> Dict[str, ServiceRoutes]:
        """
        Discover all services and their routes.

        Returns:
            Dictionary mapping service_id -> ServiceRoutes
        """
        logger.info(f"🔍 Discovering services in {self.services_dir}")

        self.services.clear()

        # Scan all subdirectories
        for service_dir in self.services_dir.iterdir():
            if not service_dir.is_dir():
                continue

            # Skip __pycache__ and hidden directories
            if service_dir.name.startswith('_') or service_dir.name.startswith('.'):
                continue

            service_id = service_dir.name

            # Check for required files
            routes_file = service_dir / "routes.py"
            service_file = service_dir / "service.py"

            has_routes = routes_file.exists()
            has_service = service_file.exists()

            # Create service entry
            service_routes = ServiceRoutes(
                service_id=service_id,
                service_path=service_dir,
                has_routes_file=has_routes,
                has_service_file=has_service
            )

            # Extract routes if routes.py exists
            if has_routes:
                try:
                    self._extract_routes(service_id, routes_file, service_routes)
                    logger.info(f"✅ {service_id}: {len(service_routes.endpoints)} endpoints")
                except Exception as e:
                    logger.warning(f"⚠️  {service_id}: Failed to extract routes - {e}")
            else:
                logger.debug(f"⏭️  {service_id}: No routes.py (old pattern)")

            self.services[service_id] = service_routes

        # Summary
        total_services = len(self.services)
        with_routes = sum(1 for s in self.services.values() if s.has_routes_file)
        total_endpoints = sum(len(s.endpoints) for s in self.services.values())

        logger.info(f"📊 Discovery complete: {total_services} services, {with_routes} with routes.py, {total_endpoints} total endpoints")

        return self.services

    def _extract_routes(self, service_id: str, routes_file: Path, service_routes: ServiceRoutes):
        """
        Extract route information from a routes.py file.

        Args:
            service_id: Service identifier
            routes_file: Path to routes.py
            service_routes: ServiceRoutes object to populate
        """
        # Import the routes module
        module_name = f"src.services.{service_id}.routes"

        try:
            spec = importlib.util.spec_from_file_location(module_name, routes_file)
            if not spec or not spec.loader:
                raise ImportError(f"Cannot load module spec for {module_name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Check for create_router function
            if not hasattr(module, 'create_router'):
                logger.warning(f"⚠️  {service_id}: routes.py exists but no create_router() function")
                return

            # Call create_router to get the router
            create_router = getattr(module, 'create_router')
            router = create_router()

            # Extract service description from module docstring
            if module.__doc__:
                service_routes.description = module.__doc__.strip().split('\n')[0]

            # Extract endpoints from router
            if hasattr(router, 'routes'):
                for route in router.routes:
                    # Skip non-route items (like APIRoute, WebSocket, etc.)
                    if not hasattr(route, 'path'):
                        continue

                    # Extract endpoint info
                    endpoint = EndpointInfo(
                        path=route.path,
                        methods=set(route.methods) if hasattr(route, 'methods') else set(),
                        name=route.name if hasattr(route, 'name') else "",
                        description=self._extract_endpoint_description(route),
                        tags=list(route.tags) if hasattr(route, 'tags') else []
                    )

                    service_routes.endpoints.append(endpoint)

            # Set router prefix if any
            if hasattr(router, 'prefix'):
                service_routes.router_prefix = router.prefix

        except Exception as e:
            logger.error(f"❌ Error extracting routes from {service_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _extract_endpoint_description(self, route) -> str:
        """Extract description from route endpoint function"""
        if hasattr(route, 'endpoint') and route.endpoint:
            endpoint_func = route.endpoint
            if endpoint_func.__doc__:
                # Get first line of docstring
                return endpoint_func.__doc__.strip().split('\n')[0]
        return ""

    def get_service_routes(self, service_id: str) -> Optional[ServiceRoutes]:
        """
        Get routes for a specific service.

        Args:
            service_id: Service identifier

        Returns:
            ServiceRoutes or None if service not found
        """
        return self.services.get(service_id)

    def get_services_with_routes(self) -> List[ServiceRoutes]:
        """
        Get all services that have routes.py files.

        Returns:
            List of ServiceRoutes for services with routes
        """
        return [s for s in self.services.values() if s.has_routes_file]

    def get_services_without_routes(self) -> List[ServiceRoutes]:
        """
        Get all services that DON'T have routes.py files.

        These services use the old pattern (routes defined in service.py).

        Returns:
            List of ServiceRoutes for services without routes
        """
        return [s for s in self.services.values() if not s.has_routes_file and s.has_service_file]

    def get_all_endpoints(self) -> List[Dict[str, Any]]:
        """
        Get flat list of all endpoints across all services.

        Returns:
            List of endpoint dictionaries with service_id included
        """
        all_endpoints = []

        for service_id, service_routes in self.services.items():
            for endpoint in service_routes.endpoints:
                endpoint_dict = endpoint.to_dict()
                endpoint_dict['service_id'] = service_id
                all_endpoints.append(endpoint_dict)

        return all_endpoints

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of route discovery.

        Returns:
            Dictionary with statistics and summaries
        """
        services_with_routes = self.get_services_with_routes()
        services_without_routes = self.get_services_without_routes()

        return {
            "total_services": len(self.services),
            "services_with_routes": len(services_with_routes),
            "services_without_routes": len(services_without_routes),
            "total_endpoints": sum(len(s.endpoints) for s in self.services.values()),
            "services": {
                "with_routes": [s.service_id for s in services_with_routes],
                "without_routes": [s.service_id for s in services_without_routes]
            },
            "endpoint_breakdown": {
                s.service_id: len(s.endpoints)
                for s in services_with_routes
            }
        }


def main():
    """CLI for testing route discovery"""
    import json

    logging.basicConfig(level=logging.INFO)

    discovery = RouteDiscovery()
    discovery.discover_all_services()

    # Print summary
    summary = discovery.generate_summary_report()
    print("\n" + "="*80)
    print("ROUTE DISCOVERY SUMMARY")
    print("="*80)
    print(json.dumps(summary, indent=2))

    # Print detailed routes for services with routes.py
    print("\n" + "="*80)
    print("SERVICES WITH ROUTES.PY")
    print("="*80)
    for service in discovery.get_services_with_routes():
        print(f"\n{service.service_id}: {len(service.endpoints)} endpoints")
        for endpoint in service.endpoints:
            methods_str = ', '.join(sorted(endpoint.methods))
            print(f"  [{methods_str}] {endpoint.path} - {endpoint.description or endpoint.name}")


if __name__ == "__main__":
    main()
