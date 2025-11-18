#!/usr/bin/env python3
"""
Hot-Reload Manager for Configuration Files

Monitors configuration files for changes and automatically reloads them.
Supports .env, settings.yaml, and services_config.yaml.

Features:
- File watching with polling
- Thread-safe reloading
- Callback support for reload events
- Configurable polling interval

Usage:
    from src.core.config import get_config

    config = get_config()
    config.enable_hot_reload(interval=60)  # Reload every 60s
"""

import os
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

logger = logging.getLogger(__name__)


class HotReloadManager:
    """
    Manages hot-reload of configuration files

    Monitors files for changes (via mtime) and triggers reload when changes are detected.
    Thread-safe with graceful shutdown support.
    """

    def __init__(self, config_manager: Any, interval: int = 60):
        """
        Initialize hot-reload manager

        Args:
            config_manager: ConfigManager instance to reload
            interval: Polling interval in seconds (default: 60)
        """
        self.config_manager = config_manager
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Files to monitor
        self.monitored_files: List[Path] = []
        self.file_mtimes: Dict[Path, float] = {}

        # Callbacks to execute on reload
        self.reload_callbacks: List[Callable] = []

        # Discover files to monitor
        self._discover_files()

    def _discover_files(self) -> None:
        """Discover configuration files to monitor"""
        project_root = Path(__file__).parent.parent.parent

        # .env file
        env_file = project_root / ".env"
        if env_file.exists():
            self.monitored_files.append(env_file)

        # settings.yaml
        settings_file = self._find_settings_file()
        if settings_file and settings_file.exists():
            self.monitored_files.append(settings_file)

        # services_config.yaml
        services_config = project_root / "services_config.yaml"
        if services_config.exists():
            self.monitored_files.append(services_config)

        # Initialize mtimes
        for file_path in self.monitored_files:
            self.file_mtimes[file_path] = file_path.stat().st_mtime

        logger.info(f"🔍 Monitoring {len(self.monitored_files)} configuration files for changes")
        for file_path in self.monitored_files:
            logger.debug(f"   - {file_path}")

    def _find_settings_file(self) -> Optional[Path]:
        """Find settings.yaml file"""
        env_path = os.environ.get('ULTRAVOX_SETTINGS_FILE')
        if env_path and Path(env_path).exists():
            return Path(env_path)

        project_root = Path(__file__).parent.parent.parent
        search_paths = [
            Path('./settings.yaml'),
            Path('./config/settings.yaml'),
            project_root / 'settings.yaml',
            Path.home() / '.ultravox' / 'settings.yaml'
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def start(self) -> None:
        """Start hot-reload monitoring thread"""
        if self.running:
            logger.warning("⚠️  Hot-reload already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        logger.info(f"🔄 Hot-reload monitoring started (interval: {self.interval}s)")

    def stop(self) -> None:
        """Stop hot-reload monitoring thread"""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=self.interval + 5)

        logger.info("🔄 Hot-reload monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in separate thread)"""
        while self.running:
            try:
                # Check for file changes
                changed_files = self._check_for_changes()

                if changed_files:
                    logger.info(f"🔄 Configuration changes detected in: {[str(f) for f in changed_files]}")
                    self._trigger_reload(changed_files)

            except Exception as e:
                logger.error(f"❌ Error in hot-reload monitoring: {e}", exc_info=True)

            # Sleep for interval
            time.sleep(self.interval)

    def _check_for_changes(self) -> List[Path]:
        """
        Check if any monitored files have changed

        Returns:
            List of changed file paths
        """
        changed_files = []

        for file_path in self.monitored_files:
            if not file_path.exists():
                # File was deleted
                logger.warning(f"⚠️  Configuration file deleted: {file_path}")
                continue

            current_mtime = file_path.stat().st_mtime
            previous_mtime = self.file_mtimes.get(file_path, 0)

            if current_mtime > previous_mtime:
                changed_files.append(file_path)
                self.file_mtimes[file_path] = current_mtime

        return changed_files

    def _trigger_reload(self, changed_files: List[Path]) -> None:
        """
        Trigger configuration reload

        Args:
            changed_files: List of files that changed
        """
        try:
            # Reload configuration manager
            logger.info("🔄 Reloading configuration...")
            self.config_manager.reload()

            # Execute callbacks
            for callback in self.reload_callbacks:
                try:
                    callback(changed_files)
                except Exception as e:
                    logger.error(f"❌ Error in reload callback: {e}", exc_info=True)

            logger.info("✅ Configuration reloaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to reload configuration: {e}", exc_info=True)

    def add_reload_callback(self, callback: Callable[[List[Path]], None]) -> None:
        """
        Add a callback to be executed after reload

        Args:
            callback: Function that takes a list of changed file paths
        """
        self.reload_callbacks.append(callback)
        logger.debug(f"➕ Added reload callback: {callback.__name__}")

    def remove_reload_callback(self, callback: Callable[[List[Path]], None]) -> None:
        """
        Remove a reload callback

        Args:
            callback: Callback function to remove
        """
        if callback in self.reload_callbacks:
            self.reload_callbacks.remove(callback)
            logger.debug(f"➖ Removed reload callback: {callback.__name__}")


# ============================================================================
# Convenience functions
# ============================================================================

def watch_config_changes(
    config_manager: Any,
    interval: int = 60,
    callback: Optional[Callable[[List[Path]], None]] = None
) -> HotReloadManager:
    """
    Convenience function to start watching config changes

    Args:
        config_manager: ConfigManager instance
        interval: Polling interval in seconds
        callback: Optional callback on reload

    Returns:
        HotReloadManager instance

    Example:
        from src.core.config import get_config, watch_config_changes

        def on_reload(changed_files):
            print(f"Config reloaded! Changed: {changed_files}")

        config = get_config()
        watcher = watch_config_changes(config, interval=30, callback=on_reload)
    """
    manager = HotReloadManager(config_manager, interval)

    if callback:
        manager.add_reload_callback(callback)

    manager.start()

    return manager
