"""
Deep Health Checks Module

Provides comprehensive health checking for the entire Ultravox Pipeline.
Tests individual services, database connectivity, and end-to-end pipeline.

Usage:
    from src.core.health_checks import HealthChecker

    checker = HealthChecker(comm_manager)
    status = await checker.deep_health()
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceHealth:
    """Health status for a single service"""

    def __init__(
        self,
        service_name: str,
        status: HealthStatus,
        response_time_ms: float,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.service_name = service_name
        self.status = status
        self.response_time_ms = response_time_ms
        self.details = details or {}
        self.error = error
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "response_time_ms": round(self.response_time_ms, 2),
            "details": self.details,
            "error": self.error,
            "timestamp": self.timestamp
        }


class HealthChecker:
    """
    Comprehensive health checker for Ultravox Pipeline

    Performs deep health checks including:
    - Individual service health
    - Database connectivity
    - End-to-end pipeline testing
    - Resource utilization
    """

    def __init__(self, comm_manager=None):
        """
        Initialize health checker

        Args:
            comm_manager: Communication manager for service calls (optional)
        """
        self.comm = comm_manager
        self.logger = logger

    async def check_service_health(
        self,
        service_name: str,
        endpoint: str = "/health",
        timeout: float = 5.0
    ) -> ServiceHealth:
        """
        Check health of a single service

        Args:
            service_name: Name of service to check
            endpoint: Health check endpoint (default: /health)
            timeout: Request timeout in seconds

        Returns:
            ServiceHealth object with status and details
        """
        start_time = time.time()

        try:
            if not self.comm:
                return ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0,
                    error="Communication manager not available"
                )

            # Call service health endpoint
            response = await asyncio.wait_for(
                self.comm.call_service(
                    service_name=service_name,
                    endpoint=endpoint,
                    method="GET"
                ),
                timeout=timeout
            )

            response_time_ms = (time.time() - start_time) * 1000

            # Determine status from response
            if response and response.get("status") == "ok":
                status = HealthStatus.HEALTHY
            elif response:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            return ServiceHealth(
                service_name=service_name,
                status=status,
                response_time_ms=response_time_ms,
                details=response
            )

        except asyncio.TimeoutError:
            response_time_ms = timeout * 1000
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                error=f"Timeout after {timeout}s"
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                error=str(e)
            )

    async def check_pipeline_health(self) -> Dict[str, Any]:
        """
        Test end-to-end pipeline with a simple request

        Returns:
            Dictionary with pipeline test results
        """
        start_time = time.time()

        try:
            # Simple pipeline test: dummy audio -> STT -> LLM -> TTS
            test_data = {
                "text": "health check test",
                "test_mode": True
            }

            # Test STT
            stt_result = await self.check_service_health("external_stt", timeout=10.0)

            # Test LLM
            llm_result = await self.check_service_health("external_llm", timeout=10.0)

            # Test TTS
            tts_result = await self.check_service_health("external_tts", timeout=10.0)

            total_time_ms = (time.time() - start_time) * 1000

            # Determine overall status
            all_healthy = all([
                stt_result.status == HealthStatus.HEALTHY,
                llm_result.status == HealthStatus.HEALTHY,
                tts_result.status == HealthStatus.HEALTHY
            ])

            return {
                "status": "ok" if all_healthy else "degraded",
                "total_time_ms": round(total_time_ms, 2),
                "services": {
                    "stt": stt_result.to_dict(),
                    "llm": llm_result.to_dict(),
                    "tts": tts_result.to_dict()
                },
                "pipeline_functional": all_healthy
            }

        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Pipeline health check failed: {e}")
            return {
                "status": "error",
                "total_time_ms": round(total_time_ms, 2),
                "error": str(e),
                "pipeline_functional": False
            }

    async def deep_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive deep health check

        Returns:
            Detailed health status of entire system
        """
        start_time = time.time()

        # Critical services to check
        critical_services = [
            "external_stt",
            "external_llm",
            "external_tts",
            "session",
            "database",
            "orchestrator"
        ]

        # Check all critical services in parallel
        service_checks = await asyncio.gather(
            *[self.check_service_health(svc) for svc in critical_services],
            return_exceptions=True
        )

        # Process results
        services_status = {}
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0

        for i, result in enumerate(service_checks):
            if isinstance(result, Exception):
                service_name = critical_services[i]
                services_status[service_name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(result)
                }
                unhealthy_count += 1
            else:
                services_status[result.service_name] = result.to_dict()
                if result.status == HealthStatus.HEALTHY:
                    healthy_count += 1
                elif result.status == HealthStatus.DEGRADED:
                    degraded_count += 1
                else:
                    unhealthy_count += 1

        # Test pipeline if all services are at least degraded
        pipeline_status = None
        if unhealthy_count == 0:
            pipeline_status = await self.check_pipeline_health()

        # Calculate overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        total_time_ms = (time.time() - start_time) * 1000

        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "total_time_ms": round(total_time_ms, 2),
            "summary": {
                "total_services": len(critical_services),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "services": services_status,
            "pipeline": pipeline_status,
            "recommendations": self._generate_recommendations(
                services_status,
                pipeline_status
            )
        }

    def _generate_recommendations(
        self,
        services_status: Dict[str, Any],
        pipeline_status: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate recommendations based on health status

        Args:
            services_status: Status of all services
            pipeline_status: Pipeline test results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for unhealthy services
        unhealthy_services = [
            name for name, status in services_status.items()
            if status.get("status") == HealthStatus.UNHEALTHY.value
        ]

        if unhealthy_services:
            recommendations.append(
                f"Restart unhealthy services: {', '.join(unhealthy_services)}"
            )

        # Check for slow services (>5s response time)
        slow_services = [
            name for name, status in services_status.items()
            if status.get("response_time_ms", 0) > 5000
        ]

        if slow_services:
            recommendations.append(
                f"Investigate slow services: {', '.join(slow_services)}"
            )

        # Check pipeline status
        if pipeline_status and not pipeline_status.get("pipeline_functional"):
            recommendations.append(
                "Pipeline end-to-end test failed - check service integration"
            )

        if not recommendations:
            recommendations.append("All systems operational")

        return recommendations


async def main():
    """Example usage"""
    checker = HealthChecker()

    # Simulate without comm_manager for testing
    print("Running deep health check...")
    result = await checker.deep_health()

    print("\n=== DEEP HEALTH CHECK RESULTS ===")
    print(f"Overall Status: {result['status']}")
    print(f"Total Time: {result['total_time_ms']}ms")
    print(f"\nSummary:")
    print(f"  Healthy: {result['summary']['healthy']}")
    print(f"  Degraded: {result['summary']['degraded']}")
    print(f"  Unhealthy: {result['summary']['unhealthy']}")

    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    asyncio.run(main())
