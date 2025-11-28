"""
Gateway Service Clients

Clients for API Gateway and other gateway services.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from .base import BaseServiceClient, ServiceClientError

logger = logging.getLogger(__name__)


class APIGatewayClient(BaseServiceClient):
    """API Gateway client (for reverse communication)"""

    def __init__(self) -> None:
        super().__init__("api_gateway")

    async def register_route(self, route: str, service: str, endpoint: str) -> bool:
        """
        Register route in API Gateway.
        
        Args:
            route: Route path
            service: Service name
            endpoint: Endpoint URL
            
        Returns:
            True if registration successful
        """
        try:
            await self._post(
                "/api/routes/register",
                json_data={"route": route, "service": service, "endpoint": endpoint}
            )
            return True
        except ServiceClientError:
            return False

    async def get_routes(self) -> List[Dict[str, Any]]:
        """Get all registered routes"""
        try:
            result = await self._get("/api/routes")
            return result.get("routes", [])
        except ServiceClientError:
            return []
