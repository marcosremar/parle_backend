"""
Prometheus Metrics for Service Manager Health Monitoring

Provides comprehensive metrics for:
- Health checks (success/failure rates)
- Service restarts (total, failures, duration)
- Circuit breaker status
- Auto-restart statistics
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any
from loguru import logger


# ============================================================================
# Health Check Metrics
# ============================================================================

# Total health checks performed
health_checks_total = Counter(
    'service_manager_health_checks_total',
    'Total number of health checks performed',
    ['service_id', 'status']  # status: healthy, unhealthy, timeout, error
)

# Health check duration
health_check_duration_seconds = Histogram(
    'service_manager_health_check_duration_seconds',
    'Time spent performing health checks',
    ['service_id'],
    buckets=[.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0]
)

# Current health status (1=healthy, 0=unhealthy)
service_health_status = Gauge(
    'service_manager_service_health_status',
    'Current health status of service (1=healthy, 0=unhealthy)',
    ['service_id']
)


# ============================================================================
# Restart Metrics
# ============================================================================

# Total service restarts attempted
service_restarts_total = Counter(
    'service_manager_service_restarts_total',
    'Total number of service restart attempts',
    ['service_id', 'reason', 'status']  # status: success, failure
)

# Restart duration
service_restart_duration_seconds = Histogram(
    'service_manager_service_restart_duration_seconds',
    'Time spent restarting services',
    ['service_id'],
    buckets=[.1, .25, .5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0]
)

# Auto-restart triggers
auto_restart_triggered_total = Counter(
    'service_manager_auto_restart_triggered_total',
    'Number of times auto-restart was triggered',
    ['service_id', 'trigger_reason']  # trigger_reason: threshold_reached, manual
)

# Restart loop prevention blocks
restart_loop_prevented_total = Counter(
    'service_manager_restart_loop_prevented_total',
    'Number of times restart loop prevention blocked a restart',
    ['service_id']
)


# ============================================================================
# Circuit Breaker Metrics
# ============================================================================

# Circuit breaker status (0=closed, 1=open)
circuit_breaker_status = Gauge(
    'service_manager_circuit_breaker_status',
    'Circuit breaker status (0=closed, 1=open)',
    ['service_id']
)

# Consecutive failures
circuit_breaker_consecutive_failures = Gauge(
    'service_manager_circuit_breaker_consecutive_failures',
    'Number of consecutive failures for service',
    ['service_id']
)

# Circuit breaker resets
circuit_breaker_resets_total = Counter(
    'service_manager_circuit_breaker_resets_total',
    'Number of times circuit breaker was reset',
    ['service_id']
)


# ============================================================================
# Service Manager Statistics
# ============================================================================

# Total services monitored
services_monitored_total = Gauge(
    'service_manager_services_monitored_total',
    'Total number of services being monitored'
)

# Active health checks
active_health_checks = Gauge(
    'service_manager_active_health_checks',
    'Number of health checks currently running'
)

# Service uptime percentage (calculated from history)
service_uptime_percentage = Gauge(
    'service_manager_service_uptime_percentage',
    'Service uptime percentage based on health check history',
    ['service_id']
)


# ============================================================================
# Service Manager Info
# ============================================================================

service_manager_info = Info(
    'service_manager',
    'Service Manager information'
)


# ============================================================================
# Helper Functions
# ============================================================================

def record_health_check(service_id: str, status: str, duration: float):
    """
    Record a health check metric.

    Args:
        service_id: Service identifier
        status: Health check status (healthy, unhealthy, timeout, error)
        duration: Duration of health check in seconds
    """
    health_checks_total.labels(service_id=service_id, status=status).inc()
    health_check_duration_seconds.labels(service_id=service_id).observe(duration)

    # Update health status gauge
    if status == 'healthy':
        service_health_status.labels(service_id=service_id).set(1)
    else:
        service_health_status.labels(service_id=service_id).set(0)


def record_restart(service_id: str, reason: str, status: str, duration: float):
    """
    Record a service restart metric.

    Args:
        service_id: Service identifier
        reason: Restart reason (auto_restart, manual, health_check_failure)
        status: Restart status (success, failure)
        duration: Duration of restart in seconds
    """
    service_restarts_total.labels(
        service_id=service_id,
        reason=reason,
        status=status
    ).inc()
    service_restart_duration_seconds.labels(service_id=service_id).observe(duration)


def record_auto_restart_trigger(service_id: str, trigger_reason: str):
    """
    Record when auto-restart is triggered.

    Args:
        service_id: Service identifier
        trigger_reason: Why auto-restart was triggered
    """
    auto_restart_triggered_total.labels(
        service_id=service_id,
        trigger_reason=trigger_reason
    ).inc()


def record_restart_loop_prevented(service_id: str):
    """
    Record when restart loop prevention blocks a restart.

    Args:
        service_id: Service identifier
    """
    restart_loop_prevented_total.labels(service_id=service_id).inc()


def update_circuit_breaker_status(service_id: str, consecutive_failures: int, threshold: int):
    """
    Update circuit breaker metrics.

    Args:
        service_id: Service identifier
        consecutive_failures: Current consecutive failure count
        threshold: Circuit breaker threshold
    """
    is_open = 1 if consecutive_failures >= threshold else 0
    circuit_breaker_status.labels(service_id=service_id).set(is_open)
    circuit_breaker_consecutive_failures.labels(service_id=service_id).set(consecutive_failures)


def record_circuit_breaker_reset(service_id: str):
    """
    Record when circuit breaker is manually reset.

    Args:
        service_id: Service identifier
    """
    circuit_breaker_resets_total.labels(service_id=service_id).inc()


def update_service_uptime(service_id: str, uptime_percentage: float):
    """
    Update service uptime percentage.

    Args:
        service_id: Service identifier
        uptime_percentage: Uptime percentage (0-100)
    """
    service_uptime_percentage.labels(service_id=service_id).set(uptime_percentage)


def set_services_monitored(count: int):
    """
    Set total number of services being monitored.

    Args:
        count: Number of services
    """
    services_monitored_total.set(count)


def set_active_health_checks(count: int):
    """
    Set number of active health checks.

    Args:
        count: Number of active checks
    """
    active_health_checks.set(count)


def initialize_metrics(version: str = "1.0.0"):
    """
    Initialize Service Manager metrics.

    Args:
        version: Service Manager version
    """
    service_manager_info.info({
        'component': 'service_manager',
        'version': version,
        'features': 'health_monitoring,auto_restart,circuit_breaker'
    })
    logger.info("📊 Service Manager Prometheus metrics initialized")


def get_all_metrics() -> Dict[str, Any]:
    """
    Get all available metrics for testing/inspection.

    Returns:
        Dictionary of all metric objects
    """
    return {
        'health_checks_total': health_checks_total,
        'health_check_duration_seconds': health_check_duration_seconds,
        'service_health_status': service_health_status,
        'service_restarts_total': service_restarts_total,
        'service_restart_duration_seconds': service_restart_duration_seconds,
        'auto_restart_triggered_total': auto_restart_triggered_total,
        'restart_loop_prevented_total': restart_loop_prevented_total,
        'circuit_breaker_status': circuit_breaker_status,
        'circuit_breaker_consecutive_failures': circuit_breaker_consecutive_failures,
        'circuit_breaker_resets_total': circuit_breaker_resets_total,
        'services_monitored_total': services_monitored_total,
        'active_health_checks': active_health_checks,
        'service_uptime_percentage': service_uptime_percentage,
    }
