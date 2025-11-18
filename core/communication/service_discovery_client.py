"""
Service Discovery Client - Discovers services via Service Manager.

Extracted from ServiceCommunicationManager (Phase 2 refactoring).

SOLID Principles:
- Single Responsibility: Only handles service discovery
- Open/Closed: Easy to add new discovery backends
- Interface Segregation: Implements IServiceDiscoveryClient
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import aiohttp
from loguru import logger

from .interfaces import IServiceDiscoveryClient


@dataclass
class DiscoveredService:
    """
    Metadata for a discovered service.

    Attributes:
        service_name: Service identifier
        host: Service host (e.g., "localhost")
        port: Service port
        health: Health status ("healthy", "unhealthy", "unknown")
        discovered_at: Timestamp when discovered
        metadata: Additional service metadata
    """
    service_name: str
    host: str
    port: int
    health: str = "unknown"
    discovered_at: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

    @property
    def base_url(self) -> str:
        """Get base URL for the service."""
        protocol = "https" if self.port == 443 else "http"
        return f"{protocol}://{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "service_name": self.service_name,
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "health": self.health,
            "discovered_at": self.discovered_at,
            "age_seconds": round(time.time() - self.discovered_at, 2),
            "metadata": self.metadata or {}
        }


class ServiceDiscoveryClient:
    """
    Client for Service Discovery (integrates with Service Manager).

    Discovers services dynamically via the Service Manager's discovery API.
    Caches results to avoid repeated lookups.

    Example:
        client = ServiceDiscoveryClient()

        # Discover service
        info = await client.lookup_service("llm")
        if info:
            print(f"LLM service at {info['host']}:{info['port']}")

        # With cache
        info2 = await client.lookup_service("llm")  # Returns cached result

        # Invalidate cache
        client.invalidate_discovery_cache("llm")
    """

    def __init__(
        self,
        service_manager_url: Optional[str] = None,
        cache_ttl_seconds: float = 60.0,
        discovery_timeout: float = 5.0,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize service discovery client.

        Args:
            service_manager_url: Service Manager URL (default: from env or localhost:8888)
            cache_ttl_seconds: Cache TTL in seconds (default: 60s)
            discovery_timeout: Discovery request timeout in seconds (default: 5s)
            session: Optional aiohttp session (creates own if None)
        """
        self.service_manager_url = (
            service_manager_url
            or os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")
        ).rstrip("/")

        self.cache_ttl_seconds = cache_ttl_seconds
        self.discovery_timeout = discovery_timeout

        # HTTP session
        self._session = session
        self._own_session = session is None

        # Discovery cache
        self._cache: Dict[str, DiscoveredService] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            f"🔍 ServiceDiscoveryClient initialized "
            f"(manager={self.service_manager_url}, cache_ttl={cache_ttl_seconds}s)"
        )

    async def lookup_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Lookup service via Service Discovery.

        Args:
            service_name: Service to discover

        Returns:
            Service info dict or None if not found

        Example:
            info = await client.lookup_service("llm")
            if info:
                url = f"http://{info['host']}:{info['port']}"
        """
        # Check cache first
        cached = self._get_from_cache(service_name)
        if cached:
            self._cache_hits += 1
            logger.debug(
                f"📦 Cache HIT for {service_name} "
                f"(age={time.time() - cached.discovered_at:.1f}s)"
            )
            return cached.to_dict()

        self._cache_misses += 1

        # Make discovery request
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()

            discovery_url = f"{self.service_manager_url}/discovery/lookup/{service_name}"

            async with self._session.get(
                discovery_url,
                timeout=aiohttp.ClientTimeout(total=self.discovery_timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Create discovered service record
                    discovered = DiscoveredService(
                        service_name=service_name,
                        host=data.get("host", "localhost"),
                        port=data.get("port", 8000),
                        health=data.get("health", "unknown"),
                        metadata=data.get("metadata", {})
                    )

                    # Cache it
                    self._cache[service_name] = discovered

                    logger.info(
                        f"✅ Discovered {service_name} at {discovered.base_url} "
                        f"(health={discovered.health})"
                    )

                    return discovered.to_dict()

                elif response.status == 404:
                    logger.debug(f"Service {service_name} not found in discovery")
                    return None

                else:
                    logger.warning(
                        f"⚠️  Discovery lookup failed for {service_name}: "
                        f"HTTP {response.status}"
                    )
                    return None

        except asyncio.TimeoutError:
            logger.warning(
                f"⏱️  Discovery lookup timeout for {service_name} "
                f"({self.discovery_timeout}s)"
            )
            return None

        except Exception as e:
            logger.debug(f"Discovery lookup failed for {service_name}: {e}")
            return None

    def invalidate_discovery_cache(self, service_name: Optional[str] = None) -> None:
        """
        Invalidate discovery cache.

        Args:
            service_name: Service to invalidate, or None for all

        Example:
            # Invalidate specific service
            client.invalidate_discovery_cache("llm")

            # Invalidate all
            client.invalidate_discovery_cache()
        """
        if service_name:
            if service_name in self._cache:
                del self._cache[service_name]
                logger.debug(f"🧹 Invalidated discovery cache for {service_name}")
        else:
            count = len(self._cache)
            self._cache.clear()
            logger.debug(f"🧹 Invalidated entire discovery cache ({count} entries)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, hit rate, etc.

        Example:
            stats = client.get_cache_stats()
            print(f"Cache hit rate: {stats['hit_rate']:.1%}")
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 3),
            "cached_services": list(self._cache.keys())
        }

    def is_cached(self, service_name: str) -> bool:
        """
        Check if service is in cache (and not expired).

        Args:
            service_name: Service to check

        Returns:
            True if cached and not expired, False otherwise
        """
        return self._get_from_cache(service_name) is not None

    async def discover_all_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover all available services.

        Returns:
            Dict mapping service_name -> service_info

        Example:
            all_services = await client.discover_all_services()
            for name, info in all_services.items():
                print(f"{name}: {info['host']}:{info['port']}")
        """
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()

            discovery_url = f"{self.service_manager_url}/discovery/services"

            async with self._session.get(
                discovery_url,
                timeout=aiohttp.ClientTimeout(total=self.discovery_timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    services = data.get("services", {})

                    logger.info(f"✅ Discovered {len(services)} services")

                    # Cache all discovered services
                    for service_name, service_info in services.items():
                        discovered = DiscoveredService(
                            service_name=service_name,
                            host=service_info.get("host", "localhost"),
                            port=service_info.get("port", 8000),
                            health=service_info.get("health", "unknown"),
                            metadata=service_info.get("metadata", {})
                        )
                        self._cache[service_name] = discovered

                    return services

                else:
                    logger.warning(
                        f"⚠️  Failed to discover all services: HTTP {response.status}"
                    )
                    return {}

        except Exception as e:
            logger.error(f"❌ Failed to discover all services: {e}")
            return {}

    async def health_check(self, service_name: str) -> str:
        """
        Check health of a discovered service.

        Args:
            service_name: Service to check

        Returns:
            Health status ("healthy", "unhealthy", "unknown")

        Example:
            health = await client.health_check("llm")
            if health == "healthy":
                # Use service
        """
        info = await self.lookup_service(service_name)

        if not info:
            return "unknown"

        return info.get("health", "unknown")

    def clear_cache(self) -> None:
        """Clear discovery cache and reset stats."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("🧹 Discovery cache cleared")

    async def close(self) -> None:
        """Close HTTP session if owned."""
        if self._own_session and self._session:
            await self._session.close()
            self._session = None
            logger.debug("HTTP session closed")

    def _get_from_cache(self, service_name: str) -> Optional[DiscoveredService]:
        """
        Get service from cache if not expired.

        Args:
            service_name: Service to retrieve

        Returns:
            DiscoveredService or None if not cached/expired
        """
        if service_name not in self._cache:
            return None

        discovered = self._cache[service_name]

        # Check if expired
        age = time.time() - discovered.discovered_at
        if age > self.cache_ttl_seconds:
            # Expired, remove from cache
            del self._cache[service_name]
            logger.debug(
                f"⏰ Cache entry expired for {service_name} (age={age:.1f}s)"
            )
            return None

        return discovered


# ============================================================================
# Singleton Instance (optional - can also use DI)
# ============================================================================

_discovery_client_instance: Optional[ServiceDiscoveryClient] = None


def get_service_discovery_client() -> ServiceDiscoveryClient:
    """
    Get global ServiceDiscoveryClient instance (singleton pattern).

    Returns:
        ServiceDiscoveryClient instance

    Example:
        client = get_service_discovery_client()
        info = await client.lookup_service("llm")
    """
    global _discovery_client_instance

    if _discovery_client_instance is None:
        _discovery_client_instance = ServiceDiscoveryClient()

    return _discovery_client_instance
