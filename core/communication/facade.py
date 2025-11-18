"""
Communication Facade - Simple interface for service communication.

Part of Communication Manager refactoring (Phase 2).
Orchestrates ProtocolSelector, ServiceResolver, and ResilienceManager.

SOLID Principles:
- Facade Pattern: Provides simple interface to complex subsystems
- Dependency Inversion: Depends on abstractions (components), not concretions
- Single Responsibility: Only orchestrates, delegates to specialists
"""

from typing import Any, Dict, Optional, Callable
import aiohttp
from loguru import logger

from .protocol_selector import ProtocolSelector, Protocol
from .service_resolver import ServiceResolver
from .resilience_manager import ResilienceManager
from src.core.exceptions import (
    ServiceUnavailableError,
    NetworkError,
    wrap_exception,
)


class CommunicationFacade:
    """
    Facade for inter-service communication.

    Provides a simple interface that orchestrates:
    - Protocol selection (ProtocolSelector)
    - URL resolution (ServiceResolver)
    - Resilience patterns (ResilienceManager)

    Example:
        facade = CommunicationFacade(service_ports={"llm": 8100})

        # Simple call
        result = await facade.call_service(
            "llm",
            "/generate",
            method="POST",
            data={"prompt": "Hello"}
        )

        # With resilience options
        result = await facade.call_service(
            "llm",
            "/generate",
            method="POST",
            data={"prompt": "Hello"},
            retries=5,
            timeout_ms=10000
        )
    """

    def __init__(
        self,
        service_ports: Optional[Dict[str, int]] = None,
        internal_services: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize communication facade.

        Args:
            service_ports: Dict mapping service names to ports
            internal_services: Dict of in-process service instances
        """
        # Initialize components
        self.protocol_selector = ProtocolSelector()
        self.service_resolver = ServiceResolver(service_ports)
        self.resilience_manager = ResilienceManager()

        # Track internal services for direct calls
        self.internal_services = internal_services or {}

        # HTTP session for external calls
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info("🔗 CommunicationFacade initialized")

    async def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        is_internal: bool = False,
        retries: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        fallback: Optional[Callable] = None,
    ) -> Any:
        """
        Call a service with automatic protocol selection and resilience.

        Args:
            service_name: Name of target service
            endpoint: API endpoint (e.g., "/generate")
            method: HTTP method (GET, POST, etc.)
            data: Form data or binary data
            json: JSON data
            is_internal: Whether service is in-process
            retries: Number of retries (optional)
            timeout_ms: Timeout in milliseconds (optional)
            fallback: Fallback function if call fails (optional)

        Returns:
            Response from service

        Raises:
            ServiceUnavailableError: If service is unreachable
            NetworkError: If network error occurs
            Other exceptions wrapped appropriately

        Example:
            result = await facade.call_service(
                "llm",
                "/generate",
                method="POST",
                json={"prompt": "Hello"},
                retries=3,
                timeout_ms=5000
            )
        """
        # 1. Check for direct call (in-process service)
        if is_internal and service_name in self.internal_services:
            return await self._call_direct(service_name, endpoint, method, json or data)

        # 2. Select protocol
        data_type = "audio" if "audio" in str(data) else "json"
        protocol = self.protocol_selector.select_protocol(
            service_name, is_internal=is_internal, data_type=data_type
        )

        # 3. Resolve URL
        url = self.service_resolver.resolve_url(service_name, endpoint, is_internal)

        # 4. Execute with resilience
        async def execute_call():
            return await self._execute_protocol(protocol, url, method, data, json)

        try:
            result = await self.resilience_manager.execute_with_resilience(
                execute_call,
                service_name=service_name,
                retries=retries,
                timeout_ms=timeout_ms,
                fallback=fallback,
            )
            return result

        except Exception as e:
            logger.error(f"❌ Failed to call {service_name}{endpoint}: {e}")
            raise wrap_exception(e, service_name=service_name, operation=f"call {endpoint}")

    async def _call_direct(
        self, service_name: str, endpoint: str, method: str, data: Any
    ) -> Any:
        """
        Call in-process service directly (zero overhead).

        Args:
            service_name: Name of service
            endpoint: Endpoint path
            method: HTTP method
            data: Request data

        Returns:
            Response from service
        """
        service = self.internal_services[service_name]

        # Direct method call on service instance
        # Assumes service has methods matching endpoints
        endpoint_method = endpoint.strip("/").replace("/", "_")

        if hasattr(service, endpoint_method):
            result = await getattr(service, endpoint_method)(data)
            logger.debug(f"✅ Direct call to {service_name}.{endpoint_method}")
            return result
        else:
            raise AttributeError(f"Service {service_name} has no method {endpoint_method}")

    async def _execute_protocol(
        self,
        protocol: Protocol,
        url: str,
        method: str,
        data: Optional[Any] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute call using selected protocol.

        Args:
            protocol: Selected protocol
            url: Target URL
            method: HTTP method
            data: Request data
            json: JSON data

        Returns:
            Response data
        """
        if protocol == Protocol.DIRECT:
            # Should not reach here (handled in call_service)
            raise ValueError("Direct protocol should be handled before this point")

        elif protocol in (Protocol.HTTP_JSON, Protocol.HTTP_BINARY):
            # Use HTTP
            return await self._execute_http(url, method, data, json)

        elif protocol == Protocol.GRPC:
            # TODO: Implement gRPC
            logger.warning("gRPC not yet implemented, falling back to HTTP")
            return await self._execute_http(url, method, data, json)

        elif protocol == Protocol.ZEROMQ:
            # TODO: Implement ZeroMQ
            logger.warning("ZeroMQ not yet implemented, falling back to HTTP")
            return await self._execute_http(url, method, data, json)

        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    async def _execute_http(
        self, url: str, method: str, data: Optional[Any] = None, json: Optional[Dict] = None
    ) -> Any:
        """
        Execute HTTP call.

        Args:
            url: Target URL
            method: HTTP method
            data: Form/binary data
            json: JSON data

        Returns:
            Response data (parsed JSON or raw)
        """
        # Ensure session exists
        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            async with self._session.request(
                method, url, data=data, json=json
            ) as response:
                response.raise_for_status()

                # Try to parse as JSON
                try:
                    return await response.json()
                except Exception:
                    # Return raw response
                    return await response.text()

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling {url}: {e}")
            raise NetworkError(url, e)

    async def close(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.debug("HTTP session closed")

    def register_internal_service(self, service_name: str, service_instance: Any) -> None:
        """
        Register an in-process service for direct calls.

        Args:
            service_name: Name of the service
            service_instance: Service instance

        Example:
            facade.register_internal_service("llm", llm_service_instance)
        """
        self.internal_services[service_name] = service_instance
        logger.info(f"Registered internal service: {service_name}")

    def update_service_port(self, service_name: str, port: int) -> None:
        """
        Update port for a service.

        Args:
            service_name: Name of service
            port: New port number
        """
        self.service_resolver.update_port(service_name, port)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics.

        Returns:
            Dict with stats from all components
        """
        return {
            "protocol_usage": self.protocol_selector.get_usage_stats(),
            "circuit_breakers": self.resilience_manager.get_circuit_status(),
            "known_services": list(self.service_resolver.get_all_services().keys()),
        }
