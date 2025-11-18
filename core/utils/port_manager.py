"""
Port Management Utilities
Provides functions to manage port conflicts and cleanup
"""

import subprocess
import time
from typing import Optional
from loguru import logger


def kill_process_on_port(port: int, wait_time: float = 1.0) -> bool:
    """
    Kill any process using the specified port

    Args:
        port: Port number to check and clean
        wait_time: Time to wait after killing process (seconds)

    Returns:
        True if a process was killed, False otherwise

    Example:
        >>> from src.core.utils.port_manager import kill_process_on_port
        >>> if kill_process_on_port(8022):
        ...     print("Port cleaned")
    """
    try:
        # Find process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            killed_count = 0

            for pid in pids:
                pid = pid.strip()
                if pid:
                    try:
                        logger.info(f"🔪 Killing process {pid} on port {port}")
                        subprocess.run(["kill", "-9", pid], timeout=2, check=True)
                        killed_count += 1
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to kill process {pid}: {e}")

            if killed_count > 0:
                logger.info(f"✅ Killed {killed_count} process(es) on port {port}")
                if wait_time > 0:
                    time.sleep(wait_time)  # Give OS time to release the port
                return True

        return False

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout while checking port {port}")
        return False
    except FileNotFoundError:
        logger.warning("lsof command not found - cannot kill process on port")
        return False
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {e}")
        return False


def is_port_in_use(port: int) -> bool:
    """
    Check if a port is currently in use

    Args:
        port: Port number to check

    Returns:
        True if port is in use, False otherwise
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.debug(f"Could not check port {port} using lsof")
        return False


def ensure_port_available(port: int, max_attempts: int = 3) -> bool:
    """
    Ensure a port is available by killing existing processes if needed

    Args:
        port: Port number to ensure is available
        max_attempts: Maximum number of cleanup attempts

    Returns:
        True if port is available, False otherwise
    """
    for attempt in range(max_attempts):
        if not is_port_in_use(port):
            logger.debug(f"Port {port} is available")
            return True

        logger.info(f"Port {port} is in use, attempting cleanup (attempt {attempt + 1}/{max_attempts})")
        if not kill_process_on_port(port, wait_time=1.0):
            logger.warning(f"Failed to cleanup port {port}")
            return False

    # Final check
    available = not is_port_in_use(port)
    if available:
        logger.info(f"✅ Port {port} is now available")
    else:
        logger.error(f"❌ Port {port} is still in use after {max_attempts} attempts")

    return available


def get_process_info_on_port(port: int) -> Optional[str]:
    """
    Get information about the process using a port

    Args:
        port: Port number to check

    Returns:
        Process information string or None if no process found
    """
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    except Exception as e:
        logger.error(f"Error getting process info for port {port}: {e}")
        return None
