"""
Service Registry Module

Centralized service registration and discovery.
"""

from .service_registry import get_registry, ServiceRegistry

__all__ = [
    "get_registry",
    "ServiceRegistry",
]
