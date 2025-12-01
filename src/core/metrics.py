"""
Prometheus Metrics for Performance Monitoring
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# Request metrics
request_count = Counter(
    "http_requests_total", "Total number of HTTP requests", ["method", "endpoint", "status"]
)

request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# Module metrics
module_initialization_time = Histogram(
    "module_initialization_seconds", "Time taken to initialize modules", ["module_name"]
)

module_errors = Counter(
    "module_errors_total", "Total number of module errors", ["module_name", "error_type"]
)

# Performance metrics
active_sessions = Gauge("active_sessions", "Number of active sessions")

conversation_count = Counter("conversations_total", "Total number of conversations", ["user_id"])

# Resource metrics
memory_usage = Gauge("memory_usage_bytes", "Memory usage in bytes")

cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage")

# Application info
app_info = Info("app_info", "Application information")


def record_request(method: str, endpoint: str, status: int, duration: float):
    """
    Record HTTP request metrics

    Args:
        method: HTTP method
        endpoint: Endpoint path
        status: HTTP status code
        duration: Request duration in seconds
    """
    request_count.labels(method=method, endpoint=endpoint, status=status).inc()
    request_duration.labels(method=method, endpoint=endpoint).observe(duration)


def record_module_init(module_name: str, duration: float):
    """
    Record module initialization time

    Args:
        module_name: Name of the module
        duration: Initialization duration in seconds
    """
    module_initialization_time.labels(module_name=module_name).observe(duration)


def record_module_error(module_name: str, error_type: str):
    """
    Record module error

    Args:
        module_name: Name of the module
        error_type: Type of error
    """
    module_errors.labels(module_name=module_name, error_type=error_type).inc()


def update_active_sessions(count: int):
    """
    Update active sessions count

    Args:
        count: Number of active sessions
    """
    active_sessions.set(count)


def update_memory_usage(bytes_used: int):
    """
    Update memory usage

    Args:
        bytes_used: Memory used in bytes
    """
    memory_usage.set(bytes_used)


def update_cpu_usage(percent: float):
    """
    Update CPU usage

    Args:
        percent: CPU usage percentage
    """
    cpu_usage.set(percent)


def set_app_info(version: str, environment: str):
    """
    Set application information

    Args:
        version: Application version
        environment: Environment name
    """
    app_info.info({"version": version, "environment": environment})
