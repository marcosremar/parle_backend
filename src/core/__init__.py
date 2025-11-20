"""
Core Module
Shared functionality for services
"""

# Core utilities
from .route_helpers import add_standard_endpoints
from .metrics import increment_metric, set_gauge
from .exceptions import ServiceUnavailableError, UltravoxError

__all__ = [
    'add_standard_endpoints',
    'increment_metric',
    'set_gauge',
    'ServiceUnavailableError',
    'UltravoxError',
]
