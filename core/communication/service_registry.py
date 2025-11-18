"""
Service Registry - Manages in-process service registration.

Extracted from ServiceCommunicationManager (Phase 2 refactoring).

SOLID Principles:
- Single Responsibility: Only handles service registration/lookup
- Open/Closed: Easy to add new registration strategies
- Interface Segregation: Implements IServiceRegistry
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from .interfaces import IServiceRegistry


@dataclass
class RegisteredService:
    """
    Metadata for a registered in-process service.

    Attributes:
        service_name: Unique service identifier
        instance: Service instance (must have callable methods)
        parent_service: Optional parent composite service
        registered_at: Timestamp when registered
        metadata: Optional additional metadata
    """
    service_name: str
    instance: Any
    parent_service: Optional[str] = None
    registered_at: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "service_name": self.service_name,
            "parent_service": self.parent_service,
            "registered_at": self.registered_at,
            "uptime_seconds": round(time.time() - self.registered_at, 2),
            "metadata": self.metadata or {},
            "instance_type": type(self.instance).__name__,
        }


class ServiceRegistry:
    """
    Registry for in-process services (direct calls, zero overhead).

    Manages services that run in the same process and can be called
    directly without network overhead. Useful for:
    - Composite services (service that contains other services)
    - Module-based execution (services loaded as Python modules)
    - Testing (mock services)

    Example:
        registry = ServiceRegistry()

        # Register service
        registry.register_service("session", session_service_instance)

        # Check if registered
        if registry.is_registered("session"):
            service = registry.get_service("session")
            result = await service.create_session(...)

        # Unregister
        registry.unregister_service("session")
    """

    def __init__(self):
        """Initialize service registry."""
        self._services: Dict[str, RegisteredService] = {}

        # Track registration by parent (for composite services)
        self._by_parent: Dict[str, set] = {}

        logger.info("📝 ServiceRegistry initialized")

    def register_service(
        self,
        service_name: str,
        service_instance: Any,
        parent_service: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an in-process service for direct calls.

        Args:
            service_name: Unique service identifier
            service_instance: Service instance (BaseService or any callable object)
            parent_service: Optional parent composite service
            metadata: Optional metadata dict

        Raises:
            ValueError: If service is already registered

        Example:
            # Register standalone service
            registry.register_service("session", session_service)

            # Register service as part of composite
            registry.register_service(
                "external_stt",
                stt_service,
                parent_service="orchestrator"
            )
        """
        if service_name in self._services:
            logger.warning(
                f"⚠️  Service '{service_name}' already registered. "
                "Replacing with new instance."
            )

        # Create registration record
        registered = RegisteredService(
            service_name=service_name,
            instance=service_instance,
            parent_service=parent_service,
            metadata=metadata
        )

        self._services[service_name] = registered

        # Track by parent
        if parent_service:
            if parent_service not in self._by_parent:
                self._by_parent[parent_service] = set()
            self._by_parent[parent_service].add(service_name)

        logger.info(
            f"✅ Registered in-process service: {service_name}"
            + (f" (parent: {parent_service})" if parent_service else "")
        )

    def unregister_service(self, service_name: str) -> None:
        """
        Unregister a service.

        Args:
            service_name: Service to unregister

        Example:
            registry.unregister_service("session")
        """
        if service_name not in self._services:
            logger.warning(f"⚠️  Service '{service_name}' not registered")
            return

        registered = self._services[service_name]

        # Remove from parent tracking
        if registered.parent_service:
            if registered.parent_service in self._by_parent:
                self._by_parent[registered.parent_service].discard(service_name)

        # Remove from registry
        del self._services[service_name]

        logger.info(f"🗑️  Unregistered service: {service_name}")

    def is_registered(self, service_name: str) -> bool:
        """
        Check if a service is registered.

        Args:
            service_name: Service identifier

        Returns:
            True if service is registered, False otherwise

        Example:
            if registry.is_registered("session"):
                service = registry.get_service("session")
        """
        return service_name in self._services

    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get registered service instance.

        Args:
            service_name: Service identifier

        Returns:
            Service instance or None if not registered

        Example:
            session_service = registry.get_service("session")
            if session_service:
                await session_service.create_session(...)
        """
        registered = self._services.get(service_name)
        return registered.instance if registered else None

    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all registered services.

        Returns:
            Dict mapping service names to instances

        Example:
            all_services = registry.get_all_services()
            for name, instance in all_services.items():
                print(f"Service: {name}, Type: {type(instance)}")
        """
        return {
            name: registered.instance
            for name, registered in self._services.items()
        }

    def get_service_metadata(self, service_name: str) -> Optional[RegisteredService]:
        """
        Get complete registration metadata for a service.

        Args:
            service_name: Service identifier

        Returns:
            RegisteredService dataclass or None

        Example:
            metadata = registry.get_service_metadata("session")
            if metadata:
                print(f"Registered at: {metadata.registered_at}")
                print(f"Parent: {metadata.parent_service}")
        """
        return self._services.get(service_name)

    def get_services_by_parent(self, parent_service: str) -> Dict[str, Any]:
        """
        Get all services registered under a parent (composite pattern).

        Args:
            parent_service: Parent service name

        Returns:
            Dict of service_name -> instance for all children

        Example:
            # Get all services in orchestrator composite
            children = registry.get_services_by_parent("orchestrator")
            # {"external_stt": <instance>, "external_llm": <instance>, ...}
        """
        child_names = self._by_parent.get(parent_service, set())
        return {
            name: self._services[name].instance
            for name in child_names
            if name in self._services
        }

    def count_services(self) -> int:
        """
        Get total number of registered services.

        Returns:
            Number of services

        Example:
            count = registry.count_services()
            print(f"Total services: {count}")
        """
        return len(self._services)

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with counts, uptime, etc.

        Example:
            stats = registry.get_registry_stats()
            print(f"Total services: {stats['total_services']}")
            print(f"Composite services: {stats['composite_services']}")
        """
        total = len(self._services)
        with_parent = sum(
            1 for r in self._services.values()
            if r.parent_service is not None
        )
        composite_count = len(self._by_parent)

        return {
            "total_services": total,
            "standalone_services": total - with_parent,
            "child_services": with_parent,
            "composite_services": composite_count,
            "services": [
                {
                    "name": name,
                    "parent": reg.parent_service,
                    "type": type(reg.instance).__name__,
                    "uptime_seconds": round(time.time() - reg.registered_at, 2)
                }
                for name, reg in self._services.items()
            ]
        }

    def clear_registry(self) -> None:
        """
        Clear all registered services.

        Warning: This will remove all service registrations!

        Example:
            registry.clear_registry()
        """
        count = len(self._services)
        self._services.clear()
        self._by_parent.clear()
        logger.info(f"🧹 Cleared service registry ({count} services removed)")

    def update_metadata(
        self,
        service_name: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update metadata for a registered service.

        Args:
            service_name: Service identifier
            metadata: New metadata dict

        Example:
            registry.update_metadata("session", {"version": "2.0", "max_sessions": 100})
        """
        if service_name not in self._services:
            logger.warning(f"⚠️  Cannot update metadata for unregistered service: {service_name}")
            return

        registered = self._services[service_name]
        registered.metadata = metadata

        logger.debug(f"Updated metadata for {service_name}: {metadata}")

    def has_parent(self, service_name: str) -> bool:
        """
        Check if service has a parent (is part of a composite).

        Args:
            service_name: Service identifier

        Returns:
            True if service has a parent, False otherwise
        """
        registered = self._services.get(service_name)
        return registered.parent_service is not None if registered else False

    def get_parent(self, service_name: str) -> Optional[str]:
        """
        Get parent service name.

        Args:
            service_name: Service identifier

        Returns:
            Parent service name or None
        """
        registered = self._services.get(service_name)
        return registered.parent_service if registered else None

    def search_services(self, pattern: str) -> Dict[str, Any]:
        """
        Search services by name pattern (case-insensitive).

        Args:
            pattern: Search pattern (substring match)

        Returns:
            Dict of matching service_name -> instance

        Example:
            # Find all "external_*" services
            external_services = registry.search_services("external_")
        """
        pattern_lower = pattern.lower()
        return {
            name: registered.instance
            for name, registered in self._services.items()
            if pattern_lower in name.lower()
        }


# ============================================================================
# Singleton Instance (optional - can also use DI)
# ============================================================================

_service_registry_instance: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """
    Get global ServiceRegistry instance (singleton pattern).

    Returns:
        ServiceRegistry instance

    Example:
        registry = get_service_registry()
        registry.register_service("session", session_instance)
    """
    global _service_registry_instance

    if _service_registry_instance is None:
        _service_registry_instance = ServiceRegistry()

    return _service_registry_instance
