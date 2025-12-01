"""
Core Module
Shared functionality for services
"""

# Core utilities - optional imports to avoid breaking if modules don't exist
try:
    from .route_helpers import add_standard_endpoints
except ImportError:

    def add_standard_endpoints(*args, **kwargs):
        pass


try:
    from .metrics import increment_metric, set_gauge
except ImportError:

    def increment_metric(*args, **kwargs):
        pass

    def set_gauge(*args, **kwargs):
        pass


try:
    from .exceptions import ServiceUnavailableError, UltravoxError
except ImportError:

    class ServiceUnavailableError(Exception):
        pass

    class UltravoxError(Exception):
        pass


__all__ = [
    "ServiceUnavailableError",
    "UltravoxError",
    "add_standard_endpoints",
    "increment_metric",
    "set_gauge",
]
