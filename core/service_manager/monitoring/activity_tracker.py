"""
Activity Tracker - Monitors service usage for auto-scaling
Tracks when services receive requests and calculates idle time
"""

import logging
import asyncio
from typing import Dict, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ServiceActivity:
    """Tracks activity metrics for a single service"""
    service_id: str
    last_request_time: Optional[datetime] = None
    total_requests: int = 0
    idle_timeout_seconds: int = 300  # 5 minutes default
    auto_scale_enabled: bool = False
    on_idle_callback: Optional[Callable[[], Awaitable[None]]] = None

    def __post_init__(self):
        """Initialize last_request_time if not set"""
        if self.last_request_time is None:
            self.last_request_time = datetime.now()

    @property
    def idle_seconds(self) -> float:
        """Calculate current idle time in seconds"""
        if self.last_request_time is None:
            return 0.0
        return (datetime.now() - self.last_request_time).total_seconds()

    @property
    def is_idle(self) -> bool:
        """Check if service has exceeded idle timeout"""
        return self.idle_seconds >= self.idle_timeout_seconds

    def record_request(self):
        """Record a new request"""
        self.last_request_time = datetime.now()
        self.total_requests += 1
        logger.debug(
            f"📊 [{self.service_id}] Request recorded "
            f"(total: {self.total_requests}, idle reset)"
        )


class ActivityTracker:
    """
    Monitors service activity and triggers auto-scaling actions

    Features:
    - Track last request time per service
    - Calculate idle time
    - Trigger callbacks when services become idle
    - Background monitoring loop
    """

    def __init__(self):
        self._services: Dict[str, ServiceActivity] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval: int = 30  # Check every 30 seconds
        self._running: bool = False

    def register_service(
        self,
        service_id: str,
        idle_timeout_seconds: int = 300,
        auto_scale_enabled: bool = False,
        on_idle_callback: Optional[Callable[[], Awaitable[None]]] = None
    ):
        """
        Register a service for activity tracking

        Args:
            service_id: Service identifier
            idle_timeout_seconds: Idle timeout in seconds (default: 300 = 5 minutes)
            auto_scale_enabled: Enable auto-scaling for this service
            on_idle_callback: Async callback to execute when service becomes idle
        """
        self._services[service_id] = ServiceActivity(
            service_id=service_id,
            idle_timeout_seconds=idle_timeout_seconds,
            auto_scale_enabled=auto_scale_enabled,
            on_idle_callback=on_idle_callback
        )

        logger.info(
            f"📊 Registered service for activity tracking: {service_id} "
            f"(timeout: {idle_timeout_seconds}s, auto-scale: {auto_scale_enabled})"
        )

    def unregister_service(self, service_id: str):
        """Remove a service from tracking"""
        if service_id in self._services:
            del self._services[service_id]
            logger.info(f"📊 Unregistered service: {service_id}")

    def record_activity(self, service_id: str):
        """
        Record activity for a service (call this on every request)

        Args:
            service_id: Service identifier
        """
        if service_id in self._services:
            self._services[service_id].record_request()
        else:
            logger.warning(
                f"⚠️  Activity recorded for unregistered service: {service_id}"
            )

    def get_idle_time(self, service_id: str) -> Optional[float]:
        """
        Get current idle time for a service

        Args:
            service_id: Service identifier

        Returns:
            Idle time in seconds, or None if service not registered
        """
        if service_id in self._services:
            return self._services[service_id].idle_seconds
        return None

    def is_service_idle(self, service_id: str) -> bool:
        """
        Check if a service has exceeded its idle timeout

        Args:
            service_id: Service identifier

        Returns:
            True if idle, False otherwise
        """
        if service_id in self._services:
            return self._services[service_id].is_idle
        return False

    def get_activity_stats(self, service_id: str) -> Optional[Dict]:
        """
        Get activity statistics for a service

        Args:
            service_id: Service identifier

        Returns:
            Dict with stats, or None if service not registered
        """
        if service_id not in self._services:
            return None

        activity = self._services[service_id]
        return {
            "service_id": service_id,
            "total_requests": activity.total_requests,
            "last_request_time": activity.last_request_time.isoformat() if activity.last_request_time else None,
            "idle_seconds": activity.idle_seconds,
            "idle_timeout_seconds": activity.idle_timeout_seconds,
            "is_idle": activity.is_idle,
            "auto_scale_enabled": activity.auto_scale_enabled
        }

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get activity statistics for all services"""
        return {
            service_id: self.get_activity_stats(service_id)
            for service_id in self._services.keys()
        }

    async def _monitoring_loop(self):
        """Background loop to monitor service activity and trigger callbacks"""
        logger.info(
            f"📊 Activity monitoring started (interval: {self._monitoring_interval}s)"
        )

        while self._running:
            try:
                for service_id, activity in self._services.items():
                    if activity.auto_scale_enabled and activity.is_idle:
                        logger.info(
                            f"⏱️  [{service_id}] Service idle for "
                            f"{activity.idle_seconds:.0f}s "
                            f"(threshold: {activity.idle_timeout_seconds}s)"
                        )

                        # Execute idle callback if registered
                        if activity.on_idle_callback:
                            try:
                                logger.info(
                                    f"🔄 [{service_id}] Triggering auto-scale callback"
                                )
                                await activity.on_idle_callback()

                                # Disable auto-scale after callback to prevent repeated calls
                                activity.auto_scale_enabled = False
                                logger.info(
                                    f"✅ [{service_id}] Auto-scale callback completed, "
                                    f"auto-scale disabled"
                                )

                            except Exception as e:
                                logger.error(
                                    f"❌ [{service_id}] Error in idle callback: {e}",
                                    exc_info=True
                                )

                # Wait before next check
                await asyncio.sleep(self._monitoring_interval)

            except asyncio.CancelledError:
                logger.info("📊 Activity monitoring cancelled")
                break
            except Exception as e:
                logger.error(
                    f"❌ Error in activity monitoring loop: {e}",
                    exc_info=True
                )
                await asyncio.sleep(self._monitoring_interval)

    async def start_monitoring(self):
        """Start background monitoring task"""
        if self._running:
            logger.warning("⚠️  Activity monitoring already running")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("✅ Activity monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring task"""
        if not self._running:
            return

        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("✅ Activity monitoring stopped")

    def set_monitoring_interval(self, interval_seconds: int):
        """Set the monitoring check interval"""
        self._monitoring_interval = interval_seconds
        logger.info(f"📊 Monitoring interval set to {interval_seconds}s")


# Global singleton instance
_activity_tracker: Optional[ActivityTracker] = None


def get_activity_tracker() -> ActivityTracker:
    """Get global activity tracker instance"""
    global _activity_tracker

    if _activity_tracker is None:
        _activity_tracker = ActivityTracker()

    return _activity_tracker
