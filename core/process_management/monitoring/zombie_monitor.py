"""Zombie process monitor and cleanup"""

import asyncio
import logging
import signal
import psutil
from typing import List, Optional

from ..config import ZombieInfo
from .registry import PIDRegistry

logger = logging.getLogger(__name__)


class ZombieMonitor:
    """
    Monitor and cleanup zombie processes
    Runs in background and automatically reaps zombies
    """

    def __init__(
        self,
        pid_registry: PIDRegistry,
        check_interval: int = 30,
        auto_cleanup: bool = True
    ):
        """
        Initialize zombie monitor

        Args:
            pid_registry: PID registry instance
            check_interval: Check interval in seconds
            auto_cleanup: Automatically reap zombies
        """
        self.pid_registry = pid_registry
        self.check_interval = check_interval
        self.auto_cleanup = auto_cleanup
        self.task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"ZombieMonitor initialized "
            f"(interval={check_interval}s, auto_cleanup={auto_cleanup})"
        )

    async def start(self) -> None:
        """Start background monitoring"""
        if self._running:
            logger.warning("ZombieMonitor already running")
            return

        self._running = True
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info("✅ ZombieMonitor started")

    async def stop(self) -> None:
        """Stop monitoring"""
        if not self._running:
            return

        self._running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        logger.info("ZombieMonitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._running:
            try:
                # Find zombies
                zombies = self.find_zombies()

                if zombies:
                    logger.warning(f"⚠️  Found {len(zombies)} zombie processes")

                    for zombie in zombies:
                        parent_info = (
                            f" (parent: {zombie.parent_name} PID {zombie.ppid})"
                            if zombie.parent_name
                            else f" (parent PID {zombie.ppid})"
                        )
                        logger.warning(
                            f"   Zombie: PID {zombie.pid} - {zombie.name}{parent_info}"
                        )

                    # Auto-cleanup if enabled
                    if self.auto_cleanup:
                        reaped = self.reap_zombies(zombies)
                        logger.info(f"✅ Attempted to reap {reaped} zombies")

                # Cleanup stale PIDs from registry
                cleaned = self.pid_registry.cleanup_stale()
                if cleaned:
                    logger.info(
                        f"Cleaned {len(cleaned)} stale PIDs: {', '.join(cleaned)}"
                    )

            except Exception as e:
                logger.error(f"Error in zombie monitor loop: {e}", exc_info=True)

            # Wait for next check
            await asyncio.sleep(self.check_interval)

    def find_zombies(self) -> List[ZombieInfo]:
        """
        Find all zombie processes

        Returns:
            List of ZombieInfo objects
        """
        zombies = []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'status', 'ppid']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        # Get parent info
                        parent_name = None
                        try:
                            parent = psutil.Process(proc.info['ppid'])
                            parent_name = parent.name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                        zombie = ZombieInfo(
                            pid=proc.info['pid'],
                            name=proc.info['name'],
                            ppid=proc.info['ppid'],
                            parent_name=parent_name
                        )
                        zombies.append(zombie)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Error finding zombies: {e}")

        return zombies

    def reap_zombies(self, zombies: List[ZombieInfo]) -> int:
        """
        Attempt to reap zombie processes by signaling parent

        Args:
            zombies: List of zombie processes

        Returns:
            Number of zombies for which reaping was attempted
        """
        count = 0

        for zombie in zombies:
            try:
                # Send SIGCHLD to parent to force reaping
                parent = psutil.Process(zombie.ppid)
                parent.send_signal(signal.SIGCHLD)

                logger.debug(
                    f"Sent SIGCHLD to parent {zombie.ppid} "
                    f"to reap zombie {zombie.pid}"
                )
                count += 1

            except psutil.NoSuchProcess:
                # Parent doesn't exist - zombie will be adopted by init
                logger.debug(
                    f"Parent {zombie.ppid} doesn't exist, "
                    f"zombie {zombie.pid} will be adopted by init"
                )
            except psutil.AccessDenied:
                logger.debug(
                    f"Access denied to send signal to parent {zombie.ppid}"
                )
            except Exception as e:
                logger.error(
                    f"Error reaping zombie {zombie.pid}: {e}"
                )

        return count

    def get_zombie_count(self) -> int:
        """
        Get current number of zombie processes

        Returns:
            Number of zombies
        """
        return len(self.find_zombies())

    def has_zombies(self) -> bool:
        """
        Check if there are any zombie processes

        Returns:
            True if zombies exist
        """
        return self.get_zombie_count() > 0
