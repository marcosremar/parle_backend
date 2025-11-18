"""Orphan process detection and cleanup"""

import re
import time
import logging
import psutil
from typing import List, Set, Pattern

from ..config import OrphanInfo
from .registry import PIDRegistry

logger = logging.getLogger(__name__)


class OrphanCleaner:
    """
    Detect and cleanup orphan processes
    Finds ultravox processes not registered in PID registry
    """

    def __init__(
        self,
        pid_registry: PIDRegistry,
        patterns: List[str] = None,
        min_age_seconds: float = 300  # 5 minutes
    ):
        """
        Initialize orphan cleaner

        Args:
            pid_registry: PID registry instance
            patterns: List of regex patterns to match ultravox processes
            min_age_seconds: Minimum age for a process to be considered orphan
        """
        self.pid_registry = pid_registry
        self.min_age_seconds = min_age_seconds

        # Default patterns for ultravox processes
        if patterns is None:
            patterns = [
                r'http_server_template\.py',
                r'ultravox-pipeline',
                r'service_manager\.main',
                r'ultravox-.*\.service',  # systemd units
            ]

        self.patterns: List[Pattern] = [re.compile(p) for p in patterns]

        logger.info(
            f"OrphanCleaner initialized "
            f"(min_age={min_age_seconds}s, {len(patterns)} patterns)"
        )

    def find_orphans(self) -> List[OrphanInfo]:
        """
        Find orphan processes not in registry

        Returns:
            List of OrphanInfo objects
        """
        registered_pids: Set[int] = set(self.pid_registry.get_all_pids())
        orphans = []
        current_time = time.time()

        try:
            for proc in psutil.process_iter([
                'pid', 'name', 'cmdline', 'create_time',
                'memory_info', 'cpu_percent'
            ]):
                try:
                    # Skip if in registry
                    if proc.info['pid'] in registered_pids:
                        continue

                    # Check if matches ultravox patterns
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if not self._matches_pattern(cmdline):
                        continue

                    # Check age
                    age = current_time - proc.info['create_time']
                    if age < self.min_age_seconds:
                        continue

                    # This is an orphan
                    mem_info = proc.info['memory_info']
                    memory_mb = mem_info.rss / (1024 * 1024) if mem_info else 0

                    orphan = OrphanInfo(
                        pid=proc.info['pid'],
                        name=proc.info['name'],
                        cmdline=cmdline,
                        age_seconds=age,
                        memory_mb=memory_mb,
                        cpu_percent=proc.info.get('cpu_percent', 0.0)
                    )
                    orphans.append(orphan)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Error finding orphans: {e}")

        return orphans

    def cleanup_orphans(self, dry_run: bool = False) -> int:
        """
        Terminate orphan processes

        Args:
            dry_run: If True, only log what would be done

        Returns:
            Number of orphans terminated (or would be terminated in dry_run)
        """
        orphans = self.find_orphans()

        if not orphans:
            logger.info("No orphan processes found")
            return 0

        logger.warning(f"Found {len(orphans)} orphan processes")

        count = 0
        for orphan in orphans:
            if dry_run:
                logger.info(
                    f"[DRY RUN] Would terminate: "
                    f"PID {orphan.pid} - {orphan.name} "
                    f"(age: {orphan.age_seconds:.0f}s, mem: {orphan.memory_mb:.1f}MB)"
                )
                count += 1
            else:
                if self._terminate_process(orphan):
                    count += 1

        if dry_run:
            logger.info(f"[DRY RUN] Would terminate {count} orphans")
        else:
            logger.info(f"✅ Terminated {count} orphan processes")

        return count

    def _matches_pattern(self, cmdline: str) -> bool:
        """
        Check if command line matches any ultravox pattern

        Args:
            cmdline: Command line string

        Returns:
            True if matches any pattern
        """
        for pattern in self.patterns:
            if pattern.search(cmdline):
                return True
        return False

    def _terminate_process(self, orphan: OrphanInfo, timeout: int = 10) -> bool:
        """
        Terminate an orphan process

        Args:
            orphan: OrphanInfo object
            timeout: Timeout for graceful shutdown

        Returns:
            True if terminated successfully
        """
        try:
            proc = psutil.Process(orphan.pid)

            logger.warning(
                f"Terminating orphan: PID {orphan.pid} - {orphan.name}"
            )
            logger.debug(f"  Cmdline: {orphan.cmdline[:100]}")
            logger.debug(
                f"  Age: {orphan.age_seconds:.0f}s, "
                f"Memory: {orphan.memory_mb:.1f}MB"
            )

            # Try graceful shutdown first (SIGTERM)
            proc.terminate()

            try:
                proc.wait(timeout=timeout)
                logger.info(f"✅ Orphan {orphan.pid} terminated gracefully")
                return True

            except psutil.TimeoutExpired:
                # Force kill (SIGKILL)
                logger.warning(
                    f"⚠️  Orphan {orphan.pid} didn't stop, force killing"
                )
                proc.kill()

                try:
                    proc.wait(timeout=5)
                    logger.info(f"✅ Orphan {orphan.pid} force killed")
                    return True

                except psutil.TimeoutExpired:
                    logger.error(f"❌ Failed to kill orphan {orphan.pid}")
                    return False

        except psutil.NoSuchProcess:
            logger.debug(f"Orphan {orphan.pid} already gone")
            return True

        except psutil.AccessDenied:
            logger.error(
                f"❌ Access denied to terminate orphan {orphan.pid}"
            )
            return False

        except Exception as e:
            logger.error(f"Error terminating orphan {orphan.pid}: {e}")
            return False

    def get_orphan_count(self) -> int:
        """
        Get current number of orphan processes

        Returns:
            Number of orphans
        """
        return len(self.find_orphans())

    def has_orphans(self) -> bool:
        """
        Check if there are any orphan processes

        Returns:
            True if orphans exist
        """
        return self.get_orphan_count() > 0

    def add_pattern(self, pattern: str) -> None:
        """
        Add a new pattern to match

        Args:
            pattern: Regex pattern string
        """
        self.patterns.append(re.compile(pattern))
        logger.info(f"Added pattern: {pattern}")
