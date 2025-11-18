"""Utilities - systemd detection, resource limits"""

from .systemd_detect import has_systemd, get_systemd_version
from .resource_limits import apply_resource_limits, create_preexec_fn

__all__ = [
    "has_systemd",
    "get_systemd_version",
    "apply_resource_limits",
    "create_preexec_fn",
]
