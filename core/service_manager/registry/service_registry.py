#!/usr/bin/env python3
"""
Service Discovery Registry
Central registry for service discovery and health monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    INITIALIZING = "initializing"
    UNKNOWN = "unknown"


@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    path: str
    method: str = "GET"
    description: str = ""


@dataclass
class ServiceRegistration:
    """Service registration record"""
    service_id: str
    name: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.UNKNOWN
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    protocols: List[str] = field(default_factory=lambda: ["http"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    requires_heartbeat: bool = True  # Internal services don't need heartbeat

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['registered_at'] = self.registered_at.isoformat()
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        data['uptime_seconds'] = (datetime.utcnow() - self.registered_at).total_seconds()
        return data

    def is_healthy(self, heartbeat_timeout: int = 30) -> bool:
        """Check if service is healthy based on heartbeat"""
        if self.status == ServiceStatus.DOWN:
            return False

        # Internal services don't require heartbeat (they're in-process)
        if not self.requires_heartbeat:
            return True

        time_since_heartbeat = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < heartbeat_timeout

    def get_base_url(self) -> str:
        """Get base URL for this service"""
        protocol = "http" if "http" in self.protocols else self.protocols[0]
        return f"{protocol}://{self.host}:{self.port}"


class ServiceRegistry:
    """
    Central service discovery registry

    Features:
    - Service auto-registration
    - Health monitoring via heartbeats
    - Endpoint discovery
    - Service lookup and routing
    """

    def __init__(self, heartbeat_timeout: int = 30, cleanup_interval: int = 60):
        """
        Initialize service registry

        Args:
            heartbeat_timeout: Seconds before service is marked as down
            cleanup_interval: Seconds between cleanup runs
        """
        self._services: Dict[str, ServiceRegistration] = {}
        self._heartbeat_timeout = heartbeat_timeout
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        logger.info("🔍 Service Registry initialized")

    async def start(self):
        """Start registry background tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_services())
        logger.info("✅ Service Registry background tasks started")

    async def stop(self):
        """Stop registry background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Service Registry stopped")

    async def register(
        self,
        name: str,
        host: str,
        port: int,
        endpoints: Optional[List[Dict[str, str]]] = None,
        protocols: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        requires_heartbeat: bool = True
    ) -> ServiceRegistration:
        """
        Register a new service or update existing registration

        Args:
            name: Service name
            host: Service host
            port: Service port
            endpoints: List of endpoint definitions
            protocols: Supported protocols (http, grpc, etc)
            metadata: Additional service metadata
            version: Service version
            requires_heartbeat: Whether service requires heartbeat (False for internal services)

        Returns:
            ServiceRegistration object
        """
        service_id = f"{name}-{port}-{uuid.uuid4().hex[:8]}"

        # Parse endpoints
        endpoint_list = []
        if endpoints:
            for ep in endpoints:
                endpoint_list.append(ServiceEndpoint(
                    path=ep.get('path', '/'),
                    method=ep.get('method', 'GET'),
                    description=ep.get('description', '')
                ))

        registration = ServiceRegistration(
            service_id=service_id,
            name=name,
            host=host,
            port=port,
            status=ServiceStatus.INITIALIZING,
            endpoints=endpoint_list,
            protocols=protocols or ["http"],
            metadata=metadata or {},
            version=version,
            requires_heartbeat=requires_heartbeat
        )

        self._services[name] = registration

        logger.info(f"📝 Service registered: {name} at {host}:{port} (id: {service_id})")
        return registration

    async def deregister(self, name: str) -> bool:
        """
        Deregister a service

        Args:
            name: Service name

        Returns:
            True if service was deregistered, False if not found
        """
        if name in self._services:
            service = self._services.pop(name)
            logger.info(f"📤 Service deregistered: {name} (was at {service.host}:{service.port})")
            return True
        return False

    async def heartbeat(self, name: str, status: Optional[ServiceStatus] = None) -> bool:
        """
        Update service heartbeat

        Args:
            name: Service name
            status: Optional status update

        Returns:
            True if heartbeat recorded, False if service not found
        """
        if name not in self._services:
            logger.warning(f"⚠️ Heartbeat from unregistered service: {name}")
            return False

        service = self._services[name]
        service.last_heartbeat = datetime.utcnow()

        if status:
            service.status = status
        elif service.status == ServiceStatus.INITIALIZING:
            service.status = ServiceStatus.HEALTHY

        logger.debug(f"💓 Heartbeat: {name} (status: {service.status.value})")
        return True

    async def update_status(self, name: str, status: ServiceStatus) -> bool:
        """
        Update service status

        Args:
            name: Service name
            status: New status

        Returns:
            True if updated, False if service not found
        """
        if name not in self._services:
            return False

        old_status = self._services[name].status
        self._services[name].status = status

        logger.info(f"🔄 Service status updated: {name} ({old_status.value} → {status.value})")
        return True

    def get_service(self, name: str) -> Optional[ServiceRegistration]:
        """
        Get service registration by name

        Args:
            name: Service name

        Returns:
            ServiceRegistration or None
        """
        return self._services.get(name)

    def get_all_services(self) -> Dict[str, ServiceRegistration]:
        """Get all registered services"""
        return self._services.copy()

    def get_healthy_services(self) -> Dict[str, ServiceRegistration]:
        """Get only healthy services"""
        return {
            name: svc
            for name, svc in self._services.items()
            if svc.is_healthy(self._heartbeat_timeout)
        }

    def discover(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Discover service endpoint information

        Args:
            name: Service name

        Returns:
            Service discovery info or None
        """
        service = self.get_service(name)
        if not service:
            return None

        return {
            "name": service.name,
            "base_url": service.get_base_url(),
            "host": service.host,
            "port": service.port,
            "status": service.status.value,
            "protocols": service.protocols,
            "endpoints": [
                {"path": ep.path, "method": ep.method, "description": ep.description}
                for ep in service.endpoints
            ],
            "metadata": service.metadata
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total = len(self._services)
        healthy = len([s for s in self._services.values() if s.is_healthy(self._heartbeat_timeout)])
        degraded = len([s for s in self._services.values() if s.status == ServiceStatus.DEGRADED])
        down = total - healthy - degraded

        return {
            "total_services": total,
            "healthy": healthy,
            "degraded": degraded,
            "down": down,
            "services": list(self._services.keys())
        }

    async def _cleanup_stale_services(self):
        """Background task to cleanup stale services"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)

                now = datetime.utcnow()
                stale_services = []

                for name, service in self._services.items():
                    # Skip heartbeat check for internal services (they're in-process)
                    if not service.requires_heartbeat:
                        continue

                    time_since_heartbeat = (now - service.last_heartbeat).total_seconds()

                    if time_since_heartbeat > self._heartbeat_timeout * 2:
                        # Mark as DOWN after 2x timeout
                        if service.status != ServiceStatus.DOWN:
                            service.status = ServiceStatus.DOWN
                            logger.warning(f"⚠️ Service marked as DOWN: {name} (no heartbeat for {time_since_heartbeat:.0f}s)")

                        # Remove after 4x timeout
                        if time_since_heartbeat > self._heartbeat_timeout * 4:
                            stale_services.append(name)

                    elif time_since_heartbeat > self._heartbeat_timeout:
                        # Mark as DEGRADED after 1x timeout
                        if service.status == ServiceStatus.HEALTHY:
                            service.status = ServiceStatus.DEGRADED
                            logger.warning(f"⚠️ Service marked as DEGRADED: {name} (no heartbeat for {time_since_heartbeat:.0f}s)")

                # Remove stale services
                for name in stale_services:
                    service = self._services.pop(name)
                    logger.info(f"🗑️ Removed stale service: {name} (inactive for {(now - service.last_heartbeat).total_seconds():.0f}s)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in cleanup task: {e}")


# Global registry instance
_registry_instance: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    """Get global service registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ServiceRegistry()
    return _registry_instance
