#!/usr/bin/env python3
"""
Service Metrics Tracker - Coleta e armazena métricas de serviços
Rastreia CPU, memória, uptime, health status com histórico
"""

import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum


class HealthStatus(Enum):
    """Health status of a service"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceMetrics:
    """Snapshot of service metrics at a point in time"""
    service_id: str
    timestamp: str
    uptime_seconds: float

    # Process metrics (for external services)
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    memory_percent: Optional[float] = None
    num_threads: Optional[int] = None

    # Health metrics
    health_status: str = HealthStatus.UNKNOWN.value
    health_response_time_ms: Optional[float] = None

    # Request metrics (if available)
    request_count: Optional[int] = None
    error_count: Optional[int] = None

    # Service state
    is_running: bool = False
    pid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class HealthCheckEntry:
    """Single health check entry"""
    timestamp: str
    status: str  # healthy, degraded, unhealthy, unknown
    response_time_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ServiceMetricsTracker:
    """
    Tracks metrics for all services
    - Real-time metrics (CPU, RAM, uptime)
    - Health check history (last 100 checks per service)
    - Request/error counters
    """

    def __init__(self, max_health_history: int = 100):
        """
        Initialize metrics tracker

        Args:
            max_health_history: Maximum health check entries to keep per service
        """
        self.max_health_history = max_health_history

        # Store current metrics snapshot for each service
        self.current_metrics: Dict[str, ServiceMetrics] = {}

        # Store health check history (circular buffer)
        self.health_history: Dict[str, deque] = {}

        # Track service start times for uptime calculation
        self.start_times: Dict[str, float] = {}

        # Track request/error counts
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}

    def record_service_start(self, service_id: str) -> None:
        """Record that a service has started"""
        self.start_times[service_id] = time.time()

        # Initialize health history if not exists
        if service_id not in self.health_history:
            self.health_history[service_id] = deque(maxlen=self.max_health_history)

        # Initialize counters
        if service_id not in self.request_counts:
            self.request_counts[service_id] = 0
        if service_id not in self.error_counts:
            self.error_counts[service_id] = 0

    def record_service_stop(self, service_id: str) -> None:
        """Record that a service has stopped"""
        if service_id in self.start_times:
            del self.start_times[service_id]

        # Clear current metrics
        if service_id in self.current_metrics:
            del self.current_metrics[service_id]

    def record_health_check(
        self,
        service_id: str,
        status: HealthStatus,
        response_time_ms: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Record a health check result

        Args:
            service_id: Service identifier
            status: Health status
            response_time_ms: Response time in milliseconds
            error: Error message if unhealthy
        """
        if service_id not in self.health_history:
            self.health_history[service_id] = deque(maxlen=self.max_health_history)

        entry = HealthCheckEntry(
            timestamp=datetime.now().isoformat(),
            status=status.value,
            response_time_ms=response_time_ms,
            error=error
        )

        self.health_history[service_id].append(entry)

    def get_health_history(self, service_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get health check history for a service

        Args:
            service_id: Service identifier
            limit: Maximum number of entries to return (None = all)

        Returns:
            List of health check entries (most recent first)
        """
        if service_id not in self.health_history:
            return []

        history = list(self.health_history[service_id])
        history.reverse()  # Most recent first

        if limit:
            history = history[:limit]

        return [entry.to_dict() for entry in history]

    def collect_process_metrics(self, service_id: str, pid: int) -> Optional[ServiceMetrics]:
        """
        Collect metrics for an external service (using psutil)

        Args:
            service_id: Service identifier
            pid: Process ID

        Returns:
            ServiceMetrics or None if process not found
        """
        try:
            process = psutil.Process(pid)

            # Get CPU and memory metrics
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            memory_percent = process.memory_percent()
            num_threads = process.num_threads()

            # Calculate uptime
            uptime_seconds = 0
            if service_id in self.start_times:
                uptime_seconds = time.time() - self.start_times[service_id]

            # Get latest health status
            health_status = HealthStatus.UNKNOWN.value
            health_response_time_ms = None
            if service_id in self.health_history and len(self.health_history[service_id]) > 0:
                latest_health = self.health_history[service_id][-1]
                health_status = latest_health.status
                health_response_time_ms = latest_health.response_time_ms

            metrics = ServiceMetrics(
                service_id=service_id,
                timestamp=datetime.now().isoformat(),
                uptime_seconds=uptime_seconds,
                cpu_percent=round(cpu_percent, 2),
                memory_mb=round(memory_mb, 2),
                memory_percent=round(memory_percent, 2),
                num_threads=num_threads,
                health_status=health_status,
                health_response_time_ms=health_response_time_ms,
                request_count=self.request_counts.get(service_id, 0),
                error_count=self.error_counts.get(service_id, 0),
                is_running=True,
                pid=pid
            )

            # Store current metrics
            self.current_metrics[service_id] = metrics

            return metrics

        except psutil.NoSuchProcess:
            return None
        except Exception as e:
            print(f"Error collecting metrics for {service_id}: {e}")
            return None

    def collect_internal_service_metrics(self, service_id: str) -> ServiceMetrics:
        """
        Collect metrics for an internal service (no separate process)

        Args:
            service_id: Service identifier

        Returns:
            ServiceMetrics (limited metrics available)
        """
        # Calculate uptime
        uptime_seconds = 0
        if service_id in self.start_times:
            uptime_seconds = time.time() - self.start_times[service_id]

        # Get latest health status
        health_status = HealthStatus.UNKNOWN.value
        health_response_time_ms = None
        if service_id in self.health_history and len(self.health_history[service_id]) > 0:
            latest_health = self.health_history[service_id][-1]
            health_status = latest_health.status
            health_response_time_ms = latest_health.response_time_ms

        metrics = ServiceMetrics(
            service_id=service_id,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime_seconds,
            cpu_percent=None,  # Not available for internal services
            memory_mb=None,
            memory_percent=None,
            num_threads=None,
            health_status=health_status,
            health_response_time_ms=health_response_time_ms,
            request_count=self.request_counts.get(service_id, 0),
            error_count=self.error_counts.get(service_id, 0),
            is_running=True,
            pid=None
        )

        # Store current metrics
        self.current_metrics[service_id] = metrics

        return metrics

    def get_current_metrics(self, service_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current metrics snapshot for a service

        Args:
            service_id: Service identifier

        Returns:
            Metrics dictionary or None if not available
        """
        if service_id in self.current_metrics:
            return self.current_metrics[service_id].to_dict()
        return None

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of metrics for all services

        Returns:
            Dictionary with summary statistics
        """
        total_services = len(self.current_metrics)
        running_services = sum(1 for m in self.current_metrics.values() if m.is_running)

        # Calculate total resource usage
        total_cpu = sum(m.cpu_percent or 0 for m in self.current_metrics.values())
        total_memory_mb = sum(m.memory_mb or 0 for m in self.current_metrics.values())

        # Count by health status
        healthy_count = sum(
            1 for m in self.current_metrics.values()
            if m.health_status == HealthStatus.HEALTHY.value
        )
        degraded_count = sum(
            1 for m in self.current_metrics.values()
            if m.health_status == HealthStatus.DEGRADED.value
        )
        unhealthy_count = sum(
            1 for m in self.current_metrics.values()
            if m.health_status == HealthStatus.UNHEALTHY.value
        )
        unknown_count = sum(
            1 for m in self.current_metrics.values()
            if m.health_status == HealthStatus.UNKNOWN.value
        )

        # Service-by-service metrics
        services_metrics = {
            service_id: metrics.to_dict()
            for service_id, metrics in self.current_metrics.items()
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_services": total_services,
                "running_services": running_services,
                "total_cpu_percent": round(total_cpu, 2),
                "total_memory_mb": round(total_memory_mb, 2),
                "health_breakdown": {
                    "healthy": healthy_count,
                    "degraded": degraded_count,
                    "unhealthy": unhealthy_count,
                    "unknown": unknown_count
                }
            },
            "services": services_metrics
        }

    def increment_request_count(self, service_id: str) -> None:
        """Increment request counter for a service"""
        if service_id not in self.request_counts:
            self.request_counts[service_id] = 0
        self.request_counts[service_id] += 1

    def increment_error_count(self, service_id: str) -> None:
        """Increment error counter for a service"""
        if service_id not in self.error_counts:
            self.error_counts[service_id] = 0
        self.error_counts[service_id] += 1

    def reset_counters(self, service_id: str) -> None:
        """Reset request and error counters for a service"""
        self.request_counts[service_id] = 0
        self.error_counts[service_id] = 0
