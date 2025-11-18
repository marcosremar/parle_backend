#!/usr/bin/env python3
"""
Port Pool Manager - Centralized Port Allocation System

Provides intelligent port allocation and management for services.
Prevents port conflicts and enables dynamic port reallocation during auto-recovery.

Features:
- Centralized port pool (8000-9000 by default)
- Intelligent allocation (preferred → previous → next available)
- Thread-safe operations
- Port release on service shutdown
- Allocation tracking and statistics

Author: Claude Code
Date: 2025-11-16
"""

import threading
from typing import Optional, Dict, Set, List
from dataclasses import dataclass
from datetime import datetime, timezone
from loguru import logger


@dataclass
class PortAllocation:
    """Port allocation record"""
    service_name: str
    port: int
    allocated_at: datetime
    previous_port: Optional[int] = None
    allocation_count: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "service_name": self.service_name,
            "port": self.port,
            "allocated_at": self.allocated_at.isoformat(),
            "previous_port": self.previous_port,
            "allocation_count": self.allocation_count
        }


class PortPoolManager:
    """
    Centralized Port Pool Manager

    Manages a pool of available ports and handles allocation/deallocation
    for services with intelligent fallback strategies.

    Thread-safe for concurrent operations.
    """

    def __init__(
        self,
        base_port: int = 8000,
        pool_size: int = 1000,
        reserved_ports: Optional[List[int]] = None
    ):
        """
        Initialize Port Pool Manager

        Args:
            base_port: Starting port number (default: 8000)
            pool_size: Number of ports in pool (default: 1000)
            reserved_ports: List of ports to exclude from pool
        """
        self.base_port = base_port
        self.pool_size = pool_size

        # Thread safety
        self._lock = threading.RLock()

        # Port tracking
        self._available_ports: Set[int] = set(range(base_port, base_port + pool_size))
        self._allocated_ports: Dict[str, PortAllocation] = {}  # service_name → allocation
        self._port_to_service: Dict[int, str] = {}  # port → service_name
        self._port_history: Dict[str, int] = {}  # service_name → last_port (preserved after release)

        # Remove reserved ports from available pool
        if reserved_ports:
            self._available_ports -= set(reserved_ports)
            logger.debug(f"Reserved {len(reserved_ports)} ports: {sorted(reserved_ports)}")

        logger.info(
            f"🎱 Port Pool Manager initialized: "
            f"range {base_port}-{base_port + pool_size - 1}, "
            f"available: {len(self._available_ports)} ports"
        )

    def allocate_port(
        self,
        service_name: str,
        preferred_port: Optional[int] = None,
        allow_reuse: bool = True
    ) -> int:
        """
        Allocate a port for a service

        Allocation strategy:
        1. If service already has a port and allow_reuse=True, return it
        2. Try preferred_port if specified and available
        3. Try previous port (if service had one before)
        4. Allocate next available port from pool

        Args:
            service_name: Name of the service
            preferred_port: Preferred port number (optional)
            allow_reuse: If True, return existing allocation if service already has port

        Returns:
            Allocated port number

        Raises:
            RuntimeError: If no ports available in pool
        """
        with self._lock:
            # Check if service already has a port
            if allow_reuse and service_name in self._allocated_ports:
                existing_allocation = self._allocated_ports[service_name]
                logger.debug(
                    f"🔌 Service {service_name} already has port {existing_allocation.port}, reusing"
                )
                return existing_allocation.port

            # Track previous port and allocation count for history
            previous_port = None
            allocation_count = 1
            if service_name in self._allocated_ports:
                previous_port = self._allocated_ports[service_name].port
                allocation_count = self._allocated_ports[service_name].allocation_count + 1
                # Release previous allocation
                self._release_port_internal(service_name)
            elif service_name in self._port_history:
                # Service was previously allocated but released
                previous_port = self._port_history[service_name]

            # Try preferred port first
            if preferred_port and preferred_port in self._available_ports:
                port = preferred_port
                logger.info(f"🎯 Allocated preferred port {port} to {service_name}")
            # Try previous port (if available)
            elif previous_port and previous_port in self._available_ports:
                port = previous_port
                logger.info(f"🔄 Reallocated previous port {port} to {service_name}")
            # Allocate next available port
            elif self._available_ports:
                port = min(self._available_ports)  # Get lowest available port
                logger.info(f"🆕 Allocated new port {port} to {service_name}")
            else:
                # No ports available!
                raise RuntimeError(
                    f"Port pool exhausted! No available ports for {service_name}. "
                    f"Allocated: {len(self._allocated_ports)}, "
                    f"Available: {len(self._available_ports)}"
                )

            # Remove from available pool
            self._available_ports.discard(port)

            allocation = PortAllocation(
                service_name=service_name,
                port=port,
                allocated_at=datetime.now(timezone.utc),
                previous_port=previous_port,
                allocation_count=allocation_count
            )

            # Track allocation
            self._allocated_ports[service_name] = allocation
            self._port_to_service[port] = service_name
            self._port_history[service_name] = port  # Save to history for future reallocation

            logger.info(
                f"✅ Port {port} allocated to {service_name} "
                f"(attempt #{allocation_count}, previous: {previous_port or 'none'})"
            )

            return port

    def release_port(self, service_name: str) -> bool:
        """
        Release a port allocated to a service

        Args:
            service_name: Name of the service

        Returns:
            True if port was released, False if service had no allocation
        """
        with self._lock:
            return self._release_port_internal(service_name)

    def _release_port_internal(self, service_name: str) -> bool:
        """Internal method to release port (must be called within lock)"""
        if service_name not in self._allocated_ports:
            logger.debug(f"Service {service_name} has no allocated port")
            return False

        allocation = self._allocated_ports[service_name]
        port = allocation.port

        # Return port to available pool
        self._available_ports.add(port)

        # Remove from tracking
        del self._allocated_ports[service_name]
        del self._port_to_service[port]

        logger.info(f"🔓 Released port {port} from {service_name}")
        return True

    def get_port(self, service_name: str) -> Optional[int]:
        """
        Get currently allocated port for a service

        Args:
            service_name: Name of the service

        Returns:
            Port number if allocated, None otherwise
        """
        with self._lock:
            allocation = self._allocated_ports.get(service_name)
            return allocation.port if allocation else None

    def get_service_for_port(self, port: int) -> Optional[str]:
        """
        Get service name using a specific port

        Args:
            port: Port number

        Returns:
            Service name if port is allocated, None otherwise
        """
        with self._lock:
            return self._port_to_service.get(port)

    def is_port_available(self, port: int) -> bool:
        """
        Check if a specific port is available

        Args:
            port: Port number to check

        Returns:
            True if port is available, False if allocated or out of range
        """
        with self._lock:
            return port in self._available_ports

    def get_allocation_info(self, service_name: str) -> Optional[Dict]:
        """
        Get detailed allocation information for a service

        Args:
            service_name: Name of the service

        Returns:
            Allocation info dict or None if not allocated
        """
        with self._lock:
            allocation = self._allocated_ports.get(service_name)
            return allocation.to_dict() if allocation else None

    def get_stats(self) -> Dict:
        """
        Get port pool statistics

        Returns:
            Statistics dictionary with counts and allocations
        """
        with self._lock:
            return {
                "base_port": self.base_port,
                "pool_size": self.pool_size,
                "available_ports": len(self._available_ports),
                "allocated_ports": len(self._allocated_ports),
                "utilization_percent": round(
                    (len(self._allocated_ports) / self.pool_size) * 100, 2
                ),
                "allocations": {
                    name: alloc.to_dict()
                    for name, alloc in self._allocated_ports.items()
                }
            }

    def get_all_allocations(self) -> Dict[str, int]:
        """
        Get all current port allocations

        Returns:
            Dictionary mapping service_name → port
        """
        with self._lock:
            return {
                name: alloc.port
                for name, alloc in self._allocated_ports.items()
            }

    def reset(self):
        """
        Reset port pool (release all allocations)

        WARNING: This will clear all port allocations!
        Use with caution, typically only in testing.
        """
        with self._lock:
            # Return all allocated ports to available pool
            for allocation in self._allocated_ports.values():
                self._available_ports.add(allocation.port)

            # Clear tracking
            self._allocated_ports.clear()
            self._port_to_service.clear()
            self._port_history.clear()

            logger.warning("🔄 Port pool reset - all allocations cleared")


# ============================================================================
# Global Port Pool Instance
# ============================================================================

_global_port_pool: Optional[PortPoolManager] = None
_pool_lock = threading.Lock()


def get_port_pool() -> PortPoolManager:
    """
    Get global Port Pool Manager instance (singleton)

    Returns:
        Global PortPoolManager instance
    """
    global _global_port_pool

    if _global_port_pool is None:
        with _pool_lock:
            if _global_port_pool is None:  # Double-check locking
                _global_port_pool = PortPoolManager()
                logger.info("🎱 Global Port Pool Manager initialized")

    return _global_port_pool


def reset_port_pool():
    """
    Reset global port pool instance

    WARNING: For testing only! Clears all allocations.
    """
    global _global_port_pool
    with _pool_lock:
        if _global_port_pool:
            _global_port_pool.reset()
            _global_port_pool = None
            logger.warning("🗑️ Global Port Pool Manager reset")
