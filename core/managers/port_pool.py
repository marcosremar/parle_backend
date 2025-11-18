"""
Port Pool Manager - Automatic port allocation for services

Prevents port conflicts by managing a pool of available ports and allocating them
automatically to services that don't have explicit port configuration.

Features:
- Port availability checking
- Preferred port allocation
- Automatic fallback to next available port
- Port reservation and release
- Thread-safe operations
- Port range configuration
"""

import socket
import threading
from typing import Optional, Set, Dict
from dataclasses import dataclass
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


@dataclass
class PortAllocation:
    """Port allocation record."""
    service_name: str
    port: int
    preferred: bool  # True if allocated preferred port, False if fallback


class PortPool:
    """
    Manages port allocation for services.

    Ensures no port conflicts by tracking allocated ports and checking
    availability before allocation.

    Example:
        pool = PortPool(start=8000, end=9000)

        # Allocate with preferred port
        port = pool.allocate("llm_service", preferred=8100)

        # Allocate next available
        port = pool.allocate("new_service")

        # Release when service stops
        pool.release("llm_service")
    """

    def __init__(
        self,
        start: int = 8000,
        end: int = 9000,
        exclude: Optional[Set[int]] = None,
        host: str = "0.0.0.0"
    ):
        """
        Initialize port pool.

        Args:
            start: Start of port range (inclusive)
            end: End of port range (exclusive)
            exclude: Set of ports to exclude from allocation
            host: Host to bind when checking port availability
        """
        self.start = start
        self.end = end
        self.host = host
        self.exclude = exclude or set()

        # Thread-safe operations
        self._lock = threading.Lock()

        # Track allocations
        self._allocated: Dict[str, PortAllocation] = {}
        self._ports_in_use: Set[int] = set()

        logger.info(
            f"PortPool initialized: range={start}-{end}, "
            f"excluded={len(self.exclude)} ports"
        )

    def is_port_free(self, port: int) -> bool:
        """
        Check if a port is available for binding.

        Args:
            port: Port number to check

        Returns:
            True if port is free, False otherwise
        """
        # Check if already allocated
        if port in self._ports_in_use:
            return False

        # Check if in excluded set
        if port in self.exclude:
            return False

        # Try to bind to the port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, port))
                return True
        except OSError:
            return False

    def allocate(
        self,
        service_name: str,
        preferred: Optional[int] = None
    ) -> int:
        """
        Allocate a port for a service.

        Args:
            service_name: Name of the service requesting port
            preferred: Preferred port number (will try this first)

        Returns:
            Allocated port number

        Raises:
            ValueError: If service already has allocated port
            RuntimeError: If no ports available in range
        """
        with self._lock:
            # Check if service already has a port
            if service_name in self._allocated:
                existing = self._allocated[service_name]
                logger.warning(
                    f"Service '{service_name}' already has port {existing.port}"
                )
                return existing.port

            # Try preferred port first
            if preferred is not None:
                if self.is_port_free(preferred):
                    self._allocate_port(service_name, preferred, is_preferred=True)
                    logger.info(
                        f"✅ Allocated preferred port {preferred} to '{service_name}'"
                    )
                    return preferred
                else:
                    logger.warning(
                        f"Preferred port {preferred} for '{service_name}' not available, "
                        f"falling back to next free port"
                    )

            # Find next available port in range
            for port in range(self.start, self.end):
                if self.is_port_free(port):
                    self._allocate_port(service_name, port, is_preferred=False)
                    logger.info(
                        f"✅ Allocated fallback port {port} to '{service_name}'"
                    )
                    return port

            # No ports available
            raise RuntimeError(
                f"No available ports in range {self.start}-{self.end} "
                f"for service '{service_name}'"
            )

    def _allocate_port(self, service_name: str, port: int, is_preferred: bool):
        """Internal: Mark port as allocated."""
        allocation = PortAllocation(
            service_name=service_name,
            port=port,
            preferred=is_preferred
        )
        self._allocated[service_name] = allocation
        self._ports_in_use.add(port)

    def release(self, service_name: str) -> bool:
        """
        Release a service's allocated port.

        Args:
            service_name: Name of the service

        Returns:
            True if port was released, False if service had no allocation
        """
        with self._lock:
            if service_name not in self._allocated:
                logger.warning(f"Service '{service_name}' has no allocated port")
                return False

            allocation = self._allocated.pop(service_name)
            self._ports_in_use.remove(allocation.port)

            logger.info(
                f"Released port {allocation.port} from '{service_name}'"
            )
            return True

    def get_allocation(self, service_name: str) -> Optional[PortAllocation]:
        """
        Get port allocation for a service.

        Args:
            service_name: Name of the service

        Returns:
            PortAllocation if service has allocated port, None otherwise
        """
        with self._lock:
            return self._allocated.get(service_name)

    def get_all_allocations(self) -> Dict[str, PortAllocation]:
        """
        Get all port allocations.

        Returns:
            Dictionary mapping service names to allocations
        """
        with self._lock:
            return self._allocated.copy()

    def get_port(self, service_name: str) -> Optional[int]:
        """
        Get allocated port for a service.

        Args:
            service_name: Name of the service

        Returns:
            Port number if allocated, None otherwise
        """
        allocation = self.get_allocation(service_name)
        return allocation.port if allocation else None

    def is_allocated(self, service_name: str) -> bool:
        """Check if a service has an allocated port."""
        with self._lock:
            return service_name in self._allocated

    def get_available_count(self) -> int:
        """Get number of available ports in range."""
        with self._lock:
            return sum(
                1 for port in range(self.start, self.end)
                if self.is_port_free(port)
            )

    def reset(self):
        """Release all allocations (for testing)."""
        with self._lock:
            self._allocated.clear()
            self._ports_in_use.clear()
            logger.info("PortPool reset: all allocations released")

    def __repr__(self) -> str:
        return (
            f"PortPool(range={self.start}-{self.end}, "
            f"allocated={len(self._allocated)}, "
            f"available={self.get_available_count()})"
        )


# Singleton instance for global port management
_global_port_pool: Optional[PortPool] = None


def get_port_pool() -> PortPool:
    """
    Get global port pool instance.

    Returns:
        Global PortPool singleton
    """
    global _global_port_pool
    if _global_port_pool is None:
        _global_port_pool = PortPool()
    return _global_port_pool


def reset_port_pool():
    """Reset global port pool (for testing)."""
    global _global_port_pool
    if _global_port_pool is not None:
        _global_port_pool.reset()
    _global_port_pool = None
