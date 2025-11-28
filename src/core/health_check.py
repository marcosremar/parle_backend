"""
Health Check Utilities - Readiness and Liveness Probes
Provides standardized health check functionality for all services
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"


class HealthCheckResult:
    """Result of a health check"""
    
    def __init__(
        self,
        status: HealthStatus,
        service_name: str,
        message: str = "",
        dependencies: Optional[Dict[str, bool]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.status = status
        self.service_name = service_name
        self.message = message
        self.dependencies = dependencies or {}
        self.metrics = metrics or {}
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        return {
            "status": self.status.value,
            "service": self.service_name,
            "message": self.message,
            "dependencies": self.dependencies,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }
    
    @property
    def is_healthy(self) -> bool:
        """Check if status is healthy"""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready (healthy or degraded)"""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class HealthChecker:
    """
    Health checker with support for readiness and liveness probes
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.dependency_checks: List[Callable] = []
        self.metric_collectors: List[Callable] = []
        self.start_time = time.time()
    
    def add_dependency_check(self, name: str, check_func: Callable) -> None:
        """Add a dependency health check"""
        async def wrapped_check():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                return result
            except Exception as e:
                logger.warning(f"Dependency {name} check failed: {e}")
                return False
        
        self.dependency_checks.append((name, wrapped_check))
    
    def add_metric_collector(self, name: str, collector_func: Callable) -> None:
        """Add a metric collector function"""
        self.metric_collectors.append((name, collector_func))
    
    async def check_liveness(self) -> HealthCheckResult:
        """
        Liveness probe - checks if service is alive
        
        Returns:
            HealthCheckResult with liveness status
        """
        try:
            # Basic liveness check - service is running
            uptime = time.time() - self.start_time
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                service_name=self.service_name,
                message="Service is alive",
                metrics={"uptime_seconds": int(uptime)}
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                service_name=self.service_name,
                message=f"Liveness check failed: {e}"
            )
    
    async def check_readiness(self) -> HealthCheckResult:
        """
        Readiness probe - checks if service is ready to accept traffic
        
        Returns:
            HealthCheckResult with readiness status including dependencies
        """
        try:
            # Check dependencies
            dependency_status = {}
            dependency_healthy = True
            
            for dep_name, check_func in self.dependency_checks:
                try:
                    is_healthy = await check_func()
                    dependency_status[dep_name] = is_healthy
                    if not is_healthy:
                        dependency_healthy = False
                except Exception as e:
                    logger.warning(f"Dependency {dep_name} check error: {e}")
                    dependency_status[dep_name] = False
                    dependency_healthy = False
            
            # Collect metrics
            metrics = {}
            for metric_name, collector_func in self.metric_collectors:
                try:
                    value = await collector_func() if asyncio.iscoroutinefunction(collector_func) else collector_func()
                    metrics[metric_name] = value
                except Exception as e:
                    logger.warning(f"Metric {metric_name} collection error: {e}")
            
            # Determine overall status
            if dependency_healthy:
                status = HealthStatus.HEALTHY
                message = "Service is ready"
            else:
                status = HealthStatus.DEGRADED
                message = "Service is ready but some dependencies are unhealthy"
            
            uptime = time.time() - self.start_time
            metrics["uptime_seconds"] = int(uptime)
            
            return HealthCheckResult(
                status=status,
                service_name=self.service_name,
                message=message,
                dependencies=dependency_status,
                metrics=metrics
            )
        
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                service_name=self.service_name,
                message=f"Readiness check failed: {e}"
            )
    
    async def check_detailed(self) -> HealthCheckResult:
        """
        Detailed health check with all information
        
        Returns:
            HealthCheckResult with comprehensive health information
        """
        # Run both liveness and readiness checks
        liveness = await self.check_liveness()
        readiness = await self.check_readiness()
        
        # Combine results
        combined_metrics = {**liveness.metrics, **readiness.metrics}
        
        # Determine overall status
        if not liveness.is_healthy:
            status = HealthStatus.UNHEALTHY
            message = "Service is not alive"
        elif readiness.is_ready:
            status = readiness.status
            message = readiness.message
        else:
            status = HealthStatus.UNHEALTHY
            message = "Service is not ready"
        
        return HealthCheckResult(
            status=status,
            service_name=self.service_name,
            message=message,
            dependencies=readiness.dependencies,
            metrics=combined_metrics
        )


# Helper functions for common dependency checks

async def check_http_service(url: str, timeout: float = 2.0) -> bool:
    """Check if HTTP service is available"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def check_database_connection(connection_func: Callable) -> bool:
    """Check database connection"""
    try:
        result = await connection_func() if asyncio.iscoroutinefunction(connection_func) else connection_func()
        return result is not None
    except Exception:
        return False

