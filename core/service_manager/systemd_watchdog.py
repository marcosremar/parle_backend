"""
Systemd Watchdog Integration for Service Manager
Provides heartbeat monitoring and lifecycle notifications
"""

import os
import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemdWatchdog:
    """
    Systemd watchdog integration

    Sends periodic heartbeat to systemd to prove service is alive.
    If heartbeat stops, systemd will restart the service.
    """

    def __init__(self):
        self.enabled = False
        self.interval_seconds = 10  # Default
        self.watchdog_task: Optional[asyncio.Task] = None
        self._systemd_available = False
        self._last_heartbeat: Optional[datetime] = None

        # Try to import systemd module
        try:
            import systemd.daemon
            self.systemd = systemd.daemon
            self._systemd_available = True
            logger.info("✅ systemd Python module available")
        except ImportError:
            logger.warning("⚠️  systemd Python module not available - watchdog disabled")
            self.systemd = None

    def initialize(self) -> bool:
        """
        Initialize watchdog from environment

        Returns:
            True if watchdog is enabled
        """
        if not self._systemd_available:
            logger.info("Systemd watchdog: disabled (module not available)")
            return False

        # Check if systemd watchdog is enabled
        watchdog_usec = os.getenv('WATCHDOG_USEC')
        watchdog_pid = os.getenv('WATCHDOG_PID')

        if not watchdog_usec:
            logger.info("Systemd watchdog: disabled (WATCHDOG_USEC not set)")
            return False

        # Verify PID matches
        if watchdog_pid and int(watchdog_pid) != os.getpid():
            logger.warning(f"Watchdog PID mismatch: expected {watchdog_pid}, got {os.getpid()}")
            return False

        # Calculate interval (send heartbeat at half the timeout)
        timeout_usec = int(watchdog_usec)
        timeout_sec = timeout_usec / 1_000_000
        self.interval_seconds = timeout_sec / 2

        self.enabled = True
        logger.info(f"✅ Systemd watchdog enabled: interval={self.interval_seconds}s, timeout={timeout_sec}s")
        return True

    async def start(self):
        """Start watchdog heartbeat loop"""
        if not self.enabled:
            logger.debug("Watchdog not enabled, skipping start")
            return

        logger.info("🐕 Starting systemd watchdog heartbeat loop")
        self.watchdog_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop watchdog"""
        if self.watchdog_task:
            logger.info("Stopping watchdog")
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to systemd"""
        while True:
            try:
                await asyncio.sleep(self.interval_seconds)
                self.heartbeat()
            except asyncio.CancelledError:
                logger.info("Watchdog heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Watchdog heartbeat error: {e}")

    def heartbeat(self):
        """Send heartbeat to systemd"""
        if not self.enabled or not self.systemd:
            return

        try:
            self.systemd.notify('WATCHDOG=1')
            self._last_heartbeat = datetime.now()
            logger.debug(f"💓 Watchdog heartbeat sent at {self._last_heartbeat}")
        except Exception as e:
            logger.error(f"Failed to send watchdog heartbeat: {e}")

    def notify_ready(self):
        """Notify systemd that service is ready"""
        if not self._systemd_available or not self.systemd:
            logger.debug("systemd not available, skipping READY notification")
            return

        try:
            self.systemd.notify('READY=1')
            logger.info("📢 Notified systemd: READY=1")
        except Exception as e:
            logger.error(f"Failed to notify systemd READY: {e}")

    def notify_reloading(self):
        """Notify systemd that service is reloading"""
        if not self._systemd_available or not self.systemd:
            return

        try:
            self.systemd.notify('RELOADING=1')
            logger.info("📢 Notified systemd: RELOADING=1")
        except Exception as e:
            logger.error(f"Failed to notify systemd RELOADING: {e}")

    def notify_stopping(self):
        """Notify systemd that service is stopping"""
        if not self._systemd_available or not self.systemd:
            return

        try:
            self.systemd.notify('STOPPING=1')
            logger.info("📢 Notified systemd: STOPPING=1")
        except Exception as e:
            logger.error(f"Failed to notify systemd STOPPING: {e}")

    def notify_status(self, status: str):
        """
        Send status message to systemd

        Args:
            status: Status message (visible in systemctl status)
        """
        if not self._systemd_available or not self.systemd:
            return

        try:
            self.systemd.notify(f'STATUS={status}')
            logger.debug(f"📢 Notified systemd: STATUS={status}")
        except Exception as e:
            logger.error(f"Failed to notify systemd STATUS: {e}")

    def get_status(self) -> dict:
        """Get watchdog status"""
        return {
            "enabled": self.enabled,
            "systemd_available": self._systemd_available,
            "interval_seconds": self.interval_seconds,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None
        }


# Global instance
_watchdog_instance: Optional[SystemdWatchdog] = None


def get_watchdog() -> SystemdWatchdog:
    """Get global watchdog instance"""
    global _watchdog_instance
    if _watchdog_instance is None:
        _watchdog_instance = SystemdWatchdog()
    return _watchdog_instance
