"""Systemd detection utilities"""

import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def has_systemd(user_mode: bool = True) -> bool:
    """
    Check if systemd is available

    Args:
        user_mode: Check for user systemd (--user)

    Returns:
        True if systemd is available, False otherwise
    """
    try:
        cmd = ["systemctl"]
        if user_mode:
            cmd.append("--user")
        cmd.append("--version")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=2
        )

        if result.returncode == 0:
            logger.debug(f"systemd detected (user_mode={user_mode})")
            return True
        else:
            logger.debug(f"systemd not available (user_mode={user_mode})")
            return False

    except FileNotFoundError:
        logger.debug("systemctl command not found")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("systemctl --version timed out")
        return False
    except Exception as e:
        logger.debug(f"Error detecting systemd: {e}")
        return False


def get_systemd_version(user_mode: bool = True) -> Optional[str]:
    """
    Get systemd version string

    Args:
        user_mode: Check for user systemd (--user)

    Returns:
        Version string or None if systemd not available
    """
    try:
        cmd = ["systemctl"]
        if user_mode:
            cmd.append("--user")
        cmd.append("--version")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=2
        )

        if result.returncode == 0:
            # First line usually contains version: "systemd 250 (250.3-1-arch)"
            first_line = result.stdout.split("\n")[0]
            return first_line.strip()
        else:
            return None

    except Exception as e:
        logger.debug(f"Error getting systemd version: {e}")
        return None


def is_systemd_unit_active(unit_name: str, user_mode: bool = True) -> bool:
    """
    Check if a systemd unit is active

    Args:
        unit_name: Name of the systemd unit
        user_mode: Use --user flag

    Returns:
        True if unit is active, False otherwise
    """
    try:
        cmd = ["systemctl"]
        if user_mode:
            cmd.append("--user")
        cmd.extend(["is-active", unit_name])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=2
        )

        # is-active returns 0 if active, non-zero otherwise
        return result.returncode == 0

    except Exception as e:
        logger.debug(f"Error checking if unit {unit_name} is active: {e}")
        return False
