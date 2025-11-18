"""Resource limit utilities for subprocess mode"""

import resource
import os
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def apply_resource_limits(
    memory_mb: int = 150,
    cpu_nice: int = 10,
    max_fds: int = 1024
) -> None:
    """
    Apply resource limits to the current process
    Used as preexec_fn in subprocess.Popen()

    Args:
        memory_mb: Memory limit in MB (RLIMIT_AS)
        cpu_nice: CPU nice value (higher = lower priority)
        max_fds: Maximum number of file descriptors

    Note:
        This function is called in the child process after fork()
        but before exec(). Exceptions here will cause the child to fail.
    """
    try:
        # Memory limit (address space)
        if memory_mb > 0:
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # CPU priority (nice value)
        if cpu_nice > 0:
            os.nice(cpu_nice)

        # File descriptor limit
        if max_fds > 0:
            resource.setrlimit(resource.RLIMIT_NOFILE, (max_fds, max_fds))

    except Exception as e:
        # Log to stderr since we're in child process
        import sys
        print(f"ERROR: Failed to apply resource limits: {e}", file=sys.stderr)
        # Don't raise - let the process continue


def create_preexec_fn(
    memory_mb: int = 150,
    cpu_nice: int = 10,
    max_fds: int = 1024
) -> Callable[[], None]:
    """
    Create a preexec_fn for subprocess.Popen() with resource limits

    Args:
        memory_mb: Memory limit in MB
        cpu_nice: CPU nice value
        max_fds: Maximum number of file descriptors

    Returns:
        Function to be used as preexec_fn parameter

    Example:
        >>> import subprocess
        >>> preexec = create_preexec_fn(memory_mb=200, cpu_nice=15)
        >>> proc = subprocess.Popen(['python', 'worker.py'], preexec_fn=preexec)
    """

    def preexec():
        apply_resource_limits(memory_mb, cpu_nice, max_fds)

    return preexec


def get_current_limits() -> dict:
    """
    Get current resource limits for the process

    Returns:
        Dictionary with current limits
    """
    try:
        as_soft, as_hard = resource.getrlimit(resource.RLIMIT_AS)
        nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        return {
            "memory_soft_mb": as_soft / (1024 * 1024) if as_soft != resource.RLIM_INFINITY else None,
            "memory_hard_mb": as_hard / (1024 * 1024) if as_hard != resource.RLIM_INFINITY else None,
            "max_fds_soft": nofile_soft,
            "max_fds_hard": nofile_hard,
            "nice": os.nice(0),  # Get current nice value
        }
    except Exception as e:
        logger.error(f"Error getting current limits: {e}")
        return {}
