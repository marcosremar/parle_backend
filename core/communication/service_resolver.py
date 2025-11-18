"""
Service Resolver - Resolves service URLs (internal vs external).

Part of Communication Manager refactoring (Phase 2).
Handles URL resolution and service location.
"""

from typing import Optional, Dict
from loguru import logger
import os


class ServiceResolver:
    """
    Resolves service URLs for communication.

    Handles:
    - Internal service URLs (localhost)
    - External service URLs (remote hosts)
    - Service discovery integration
    - Port resolution from configuration

    SOLID Principles:
    - Single Responsibility: Only handles URL resolution
    - Open/Closed: Easy to add new resolution strategies
    """

    def __init__(self, service_ports: Optional[Dict[str, int]] = None):
        """
        Initialize service resolver.

        Args:
            service_ports: Dict mapping service names to ports
                (loaded from services_config.yaml)
        """
        self.service_ports = service_ports or {}
        self.service_manager_url = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")

        # Cache for resolved URLs
        self._url_cache: Dict[str, str] = {}

        logger.info(f"🔍 Service Resolver initialized with {len(self.service_ports)} known ports")

    def resolve_url(
        self,
        service_name: str,
        endpoint: str = "",
        is_internal: bool = False,
        custom_host: Optional[str] = None,
        custom_port: Optional[int] = None,
    ) -> str:
        """
        Resolve full URL for a service endpoint.

        Args:
            service_name: Name of the service
            endpoint: API endpoint path (e.g., "/generate")
            is_internal: Whether service runs in-process (use Service Manager URL)
            custom_host: Override host (for remote services)
            custom_port: Override port

        Returns:
            Full URL (e.g., "http://localhost:8100/generate")

        Example:
            resolver = ServiceResolver({"llm": 8100})

            # Internal service (module)
            url = resolver.resolve_url("external_llm", "/generate", is_internal=True)
            # Returns: "http://localhost:8888/external_llm/generate"

            # External service
            url = resolver.resolve_url("llm", "/generate")
            # Returns: "http://localhost:8100/generate"
        """
        # Check cache first
        cache_key = f"{service_name}:{endpoint}:{is_internal}:{custom_host}:{custom_port}"
        if cache_key in self._url_cache:
            return self._url_cache[cache_key]

        # Build URL
        if is_internal:
            # Internal/module service - route through Service Manager
            base_url = self.service_manager_url.rstrip("/")
            full_url = f"{base_url}/{service_name}{endpoint}"
        else:
            # External service - use its own port
            host = custom_host or "localhost"
            port = custom_port or self.service_ports.get(service_name)

            if not port:
                logger.warning(
                    f"⚠️  Port not found for service '{service_name}'. "
                    "Using default 8000 or relying on service discovery."
                )
                port = 8000

            protocol = "https" if port == 443 else "http"
            full_url = f"{protocol}://{host}:{port}{endpoint}"

        # Cache the result
        self._url_cache[cache_key] = full_url

        logger.debug(f"Resolved URL for {service_name}{endpoint}: {full_url}")
        return full_url

    def resolve_base_url(
        self, service_name: str, is_internal: bool = False, custom_host: Optional[str] = None
    ) -> str:
        """
        Resolve base URL for a service (without endpoint).

        Args:
            service_name: Name of the service
            is_internal: Whether service is internal
            custom_host: Override host

        Returns:
            Base URL (e.g., "http://localhost:8100")
        """
        return self.resolve_url(service_name, "", is_internal, custom_host)

    def update_port(self, service_name: str, port: int) -> None:
        """
        Update port for a service.

        Args:
            service_name: Name of the service
            port: New port number
        """
        self.service_ports[service_name] = port
        self.invalidate_cache(service_name)
        logger.info(f"Updated port for {service_name}: {port}")

    def invalidate_cache(self, service_name: Optional[str] = None) -> None:
        """
        Invalidate URL cache.

        Args:
            service_name: If provided, only invalidate this service.
                         If None, clear entire cache.
        """
        if service_name:
            # Remove all cache entries for this service
            keys_to_remove = [key for key in self._url_cache if key.startswith(f"{service_name}:")]
            for key in keys_to_remove:
                del self._url_cache[key]
            logger.debug(f"Invalidated cache for {service_name}")
        else:
            # Clear entire cache
            self._url_cache.clear()
            logger.debug("Invalidated entire URL cache")

    def is_service_known(self, service_name: str) -> bool:
        """
        Check if service is in known ports configuration.

        Args:
            service_name: Name of the service

        Returns:
            True if port is configured, False otherwise
        """
        return service_name in self.service_ports

    def get_all_services(self) -> Dict[str, int]:
        """
        Get all known services and their ports.

        Returns:
            Dict mapping service names to ports
        """
        return self.service_ports.copy()
