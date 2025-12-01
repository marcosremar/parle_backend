"""
Health Checker Engine

Service health checking functionality.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HealthChecker:
    """Checks health of downstream services."""

    def __init__(self, clients: dict[str, Any]) -> None:
        """
        Initialize health checker.

        Args:
            clients: Dictionary of service clients to check
        """
        self.clients = clients

    async def check_all_services(self) -> dict[str, bool]:
        """
        Check health of all downstream services.

        Returns:
            Dictionary mapping service names to health status (bool)
        """
        logger.info("🔍 Checking downstream services health...")

        health_status: dict[str, bool] = {}
        for name, client in self.clients.items():
            try:
                is_healthy = await client.health_check()
                health_status[name] = is_healthy
                status_icon = "✅" if is_healthy else "❌"
                logger.info(f"   {status_icon} {name}: {'healthy' if is_healthy else 'unhealthy'}")
            except Exception as e:
                health_status[name] = False
                logger.warning(f"   ❌ {name}: {e}")

        # Log summary
        healthy_count = sum(1 for v in health_status.values() if v)
        total_count = len(health_status)
        logger.info(f"📊 Services health: {healthy_count}/{total_count} healthy")

        return health_status
