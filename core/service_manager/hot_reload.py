#!/usr/bin/env python3
"""
Hot Reload with Watchdog - Automatic service restart on code changes

Uses Watchdog to monitor file changes and restart affected services automatically.

Features:
- Monitors service directories for .py file changes
- Identifies which service changed
- Restarts only affected service (3-5s instead of 90s full restart)
- Debouncing to avoid multiple restarts
- Graceful restart with dependency handling

🔥 HOT RELOAD TEST - This change should trigger main process reload
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, Set, Optional, Callable, TYPE_CHECKING

# Lazy import to avoid import errors if watchdog is not installed or _ctypes is missing
if TYPE_CHECKING:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent

logger = logging.getLogger(__name__)

# Check if watchdog is available
_WATCHDOG_AVAILABLE = False
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    _WATCHDOG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Watchdog not available: {e}")
    logger.warning("   Hot reload will be disabled")
    # Create stub classes
    class FileSystemEventHandler:
        pass
    class FileSystemEvent:
        pass
except Exception as e:
    logger.error(f"❌ Error importing watchdog: {e}")
    logger.error("   Hot reload will be disabled")
    # Create stub classes
    class FileSystemEventHandler:
        pass
    class FileSystemEvent:
        pass


class ServiceFileChangeHandler(FileSystemEventHandler):
    """
    Watchdog event handler for service file changes

    Monitors .py files in service directories and triggers service restarts
    """

    def __init__(
        self,
        restart_callback: Callable[[str], None],
        service_manager_reload_callback: Optional[Callable[[], None]] = None,
        debounce_seconds: float = 2.0
    ):
        """
        Initialize file change handler

        Args:
            restart_callback: Async function to restart a service (service_name)
            service_manager_reload_callback: Async function to reload Service Manager (no args)
            debounce_seconds: Wait time before triggering restart (to batch changes)
        """
        super().__init__()
        self.restart_callback = restart_callback
        self.service_manager_reload_callback = service_manager_reload_callback
        self.debounce_seconds = debounce_seconds

        # Debouncing state
        self._pending_restarts: Dict[str, float] = {}  # service_name -> last_change_time
        self._pending_manager_reload: Optional[float] = None  # timestamp for Service Manager reload
        self._debounce_task: Optional[asyncio.Task] = None

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events"""
        if event.is_directory:
            return

        # Only handle .py files
        if not event.src_path.endswith('.py'):
            return

        # Ignore __pycache__ and .pyc files
        if '__pycache__' in event.src_path or event.src_path.endswith('.pyc'):
            return

        # Detect which service changed
        service_name = self._detect_service_from_path(event.src_path)

        if service_name:
            logger.info(f"🔄 File changed: {event.src_path}")
            logger.info(f"   Target: {service_name}")

            # Check if this is a main process file (Service Manager + internal services)
            if service_name == "main":
                # Mark main process for reload
                self._pending_manager_reload = time.time()
                logger.info(f"   ⚠️  Main process reload will be triggered after {self.debounce_seconds}s")
            else:
                # Add to pending service restarts with current timestamp
                self._pending_restarts[service_name] = time.time()

            # Schedule debounced restart
            self._schedule_debounced_restart()

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events"""
        # Treat creation like modification
        self.on_modified(event)

    def _detect_service_from_path(self, file_path: str) -> Optional[str]:
        """
        Detect which service a file belongs to, or if it's a main process file

        Strategy:
        1. If file is in src/services/{service_name}/ → return service_name (individual reload)
        2. Otherwise → return "main" (full process reload)

        Args:
            file_path: Absolute path to changed file

        Returns:
            Service name (e.g., "orchestrator", "llm") for individual reload,
            "main" for main process reload, or None to ignore
        """
        try:
            path = Path(file_path)
            parts = path.parts

            # FIRST: Check if file is in src/services/{service_name}/
            # These get individual service hot-reload
            if 'services' in parts:
                # Find index of 'services'
                idx = parts.index('services')

                # Service name is the next part
                if idx + 1 < len(parts):
                    service_name = parts[idx + 1]

                    # Validate service name (should be a directory with service.py)
                    service_dir = path.parents[len(parts) - idx - 2]
                    service_file = service_dir / 'service.py'

                    if service_file.exists():
                        logger.info(f"   Detected service file change → individual reload")
                        return service_name

            # EVERYTHING ELSE: Trigger main process reload
            # This includes:
            # - src/core/service_manager/ (Service Manager code)
            # - config/ (configuration files)
            # - Root level files (main.sh, etc.)
            # - Any other project files
            logger.info(f"   Detected project file change → main process reload")
            logger.info(f"   This will restart Service Manager + all internal services")
            return "main"

        except Exception as e:
            logger.error(f"❌ Error detecting service from path {file_path}: {e}")
            return None

    def _schedule_debounced_restart(self):
        """
        Schedule debounced restart check

        Creates a task that waits for debounce_seconds and then restarts services

        Note: Uses asyncio.create_task() as hot reload is async.
        For observability, check hot reload status via GET /hot-reload/status
        """
        # Cancel previous debounce task if exists
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        # Create new debounce task
        # Note: We use asyncio.create_task here because hot reload is inherently async
        # and needs to run in the event loop (not thread pool)
        self._debounce_task = asyncio.create_task(self._debounced_restart())

        # Log for observability
        logger.debug(f"🔄 Hot reload debounce task scheduled")
        logger.debug(f"   Pending restarts: {len(self._pending_restarts)}")
        logger.debug(f"   Pending main reload: {self._pending_manager_reload is not None}")

    async def _debounced_restart(self):
        """
        Wait for debounce period and restart services

        This batches multiple rapid changes into a single restart
        """
        try:
            # Wait for debounce period
            await asyncio.sleep(self.debounce_seconds)

            # Get current time
            now = time.time()

            # Check if main process reload is pending and stable
            if self._pending_manager_reload and (now - self._pending_manager_reload >= self.debounce_seconds):
                logger.info(f"🔄 Main process code changed - triggering full reload via systemd")
                logger.info(f"   This will restart Service Manager + all internal services")

                if self.service_manager_reload_callback:
                    try:
                        await self.service_manager_reload_callback()
                    except Exception as e:
                        logger.error(f"❌ Failed to reload main process: {e}")

                # Clear pending manager reload
                self._pending_manager_reload = None

                # Clear all pending service restarts (main process reload will handle them)
                self._pending_restarts.clear()

                # Don't process individual service restarts if we're reloading the whole main process
                return

            # Find services that haven't changed in debounce_seconds (stable)
            services_to_restart: Set[str] = set()

            for service_name, last_change_time in list(self._pending_restarts.items()):
                if now - last_change_time >= self.debounce_seconds:
                    services_to_restart.add(service_name)
                    del self._pending_restarts[service_name]

            # Restart stable services
            if services_to_restart:
                logger.info(f"🔄 Restarting {len(services_to_restart)} service(s): {', '.join(services_to_restart)}")

                for service_name in services_to_restart:
                    try:
                        await self.restart_callback(service_name)
                    except Exception as e:
                        logger.error(f"❌ Failed to restart {service_name}: {e}")

            # If there are still pending restarts, schedule another check
            if self._pending_restarts or self._pending_manager_reload:
                self._schedule_debounced_restart()

        except asyncio.CancelledError:
            logger.debug("Debounce task cancelled")
        except Exception as e:
            logger.error(f"❌ Error in debounced restart: {e}")


class HotReloadManager:
    """
    Hot Reload Manager using Watchdog

    Monitors service directories and main process code, restarts on changes

    The "main" process includes:
    - Service Manager (orchestration, API endpoints, management)
    - All internal services running in-process (session, scenarios, orchestrator, etc.)
    """

    def __init__(
        self,
        project_root: Path,
        restart_callback: Callable[[str], None],
        service_manager_reload_callback: Optional[Callable[[], None]] = None,
        debounce_seconds: float = 2.0
    ):
        """
        Initialize Hot Reload Manager

        Args:
            project_root: Root directory of the project (watches entire project)
            restart_callback: Async function to restart a service
            service_manager_reload_callback: Async function to reload main process (Service Manager + internal services)
            debounce_seconds: Debounce time for restart (default: 2s)
        """
        self.project_root = project_root
        self.restart_callback = restart_callback
        self.service_manager_reload_callback = service_manager_reload_callback
        self.debounce_seconds = debounce_seconds

        # Watchdog observer
        self.observer: Optional[Observer] = None

        # Event handler
        self.event_handler = ServiceFileChangeHandler(
            restart_callback=restart_callback,
            service_manager_reload_callback=service_manager_reload_callback,
            debounce_seconds=debounce_seconds
        )

        logger.info(f"🔥 HotReloadManager initialized")
        logger.info(f"   Project root: {project_root}")
        logger.info(f"   Watch strategy: Service files → individual reload, All other files → main process reload")
        logger.info(f"   Debounce: {debounce_seconds}s")

    def start(self):
        """
        Start watching for file changes across entire project
        """
        logger.info("🔥 Starting Hot Reload Manager...")

        # Check if watchdog is available
        if not _WATCHDOG_AVAILABLE:
            logger.warning("⚠️ Hot reload disabled - watchdog not available")
            logger.warning("   To enable hot reload, install watchdog with proper _ctypes support:")
            logger.warning("   pip install watchdog")
            return

        try:
            # Create observer
            self.observer = Observer()

            # Watch ENTIRE project directory recursively
            # - Files in src/services/{service_name}/ → individual service reload
            # - All other files → main process reload (Service Manager + internal services)
            self.observer.schedule(
                self.event_handler,
                str(self.project_root),
                recursive=True
            )
            logger.info(f"   📁 Watching: {self.project_root} (entire project)")

            # Start observer
            self.observer.start()

            logger.info(f"✅ Hot reload watching started")
            logger.info(f"   • Service file changes (src/services/*/) → individual service reload (3-5s)")
            logger.info(f"   • All other file changes → main process reload via systemd (10-15s)")
            logger.info(f"   • Main process = Service Manager + all internal services")
            logger.info(f"🏁 Hot Reload Manager startup finished")

        except Exception as e:
            logger.error(f"❌ Failed to start hot reload: {e}")
            raise

    def stop(self):
        """
        Stop watching for file changes
        """
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5)
                logger.info(f"🛑 Hot reload watching stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping hot reload: {e}")

    def get_status(self) -> Dict:
        """
        Get hot reload status (observability endpoint)

        Returns:
            Status dict with detailed information about hot reload state
        """
        # Get debounce task status
        debounce_task_status = "idle"
        if self.event_handler._debounce_task:
            if self.event_handler._debounce_task.done():
                debounce_task_status = "completed"
            elif self.event_handler._debounce_task.cancelled():
                debounce_task_status = "cancelled"
            else:
                debounce_task_status = "running"

        return {
            "enabled": self.observer is not None and self.observer.is_alive(),
            "watching": str(self.project_root),
            "debounce_seconds": self.debounce_seconds,
            "pending_restarts": len(self.event_handler._pending_restarts),
            "pending_services": list(self.event_handler._pending_restarts.keys()),
            "pending_main_reload": self.event_handler._pending_manager_reload is not None,
            "debounce_task_status": debounce_task_status,
            "background_task_info": {
                "type": "asyncio.Task",
                "note": "Hot reload uses asyncio for file watching (async I/O)",
                "observability": "Check this endpoint for status"
            }
        }


# Example usage in Service Manager:
#
# # In src/core/service_manager/main.py
#
# class ServiceManager:
#     def __init__(self):
#         # ... existing code ...
#
#         # Initialize Hot Reload Manager
#         project_root = Path.cwd()  # Watch entire project
#         self.hot_reload = HotReloadManager(
#             project_root=project_root,
#             restart_callback=self.restart_service,
#             service_manager_reload_callback=self.reload_main_process,
#             debounce_seconds=2.0
#         )
#
#     async def start(self):
#         # ... existing startup code ...
#
#         # Start hot reload
#         if os.getenv("ENABLE_HOT_RELOAD", "true").lower() == "true":
#             self.hot_reload.start()
#             logger.info("🔥 Hot reload enabled - watching entire project")
#         else:
#             logger.info("⏭️  Hot reload disabled (set ENABLE_HOT_RELOAD=true to enable)")
#
#     async def restart_service(self, service_name: str):
#         """
#         Restart a single service (called by hot reload)
#
#         Args:
#             service_name: Service to restart (e.g., "orchestrator")
#         """
#         logger.info(f"🔄 Restarting service: {service_name}")
#
#         try:
#             # 1. Stop service
#             if service_name in self.services:
#                 service = self.services[service_name]
#                 await service.stop()
#                 logger.info(f"   ✅ Stopped: {service_name}")
#
#             # 2. Clear Python module cache
#             self._clear_module_cache(service_name)
#             logger.info(f"   ✅ Cleared module cache")
#
#             # 3. Reload service class
#             service_class = self._load_service_class(service_name)
#             logger.info(f"   ✅ Reloaded service class")
#
#             # 4. Create new service instance
#             service = service_class(config=self.configs[service_name])
#             self.services[service_name] = service
#
#             # 5. Start service
#             await service.start()
#             logger.info(f"   ✅ Started: {service_name}")
#
#             logger.info(f"✅ Service restarted successfully: {service_name}")
#
#         except Exception as e:
#             logger.error(f"❌ Failed to restart {service_name}: {e}")
#             import traceback
#             traceback.print_exc()
#
#     def _clear_module_cache(self, service_name: str):
#         """
#         Clear Python module cache for a service
#
#         Args:
#             service_name: Service name (e.g., "orchestrator")
#         """
#         import sys
#
#         # Find all modules that belong to this service
#         modules_to_remove = []
#
#         for module_name in sys.modules.keys():
#             # Check if module belongs to this service
#             # Example: src.services.orchestrator.service
#             if f"services.{service_name}" in module_name:
#                 modules_to_remove.append(module_name)
#
#         # Remove modules from cache
#         for module_name in modules_to_remove:
#             del sys.modules[module_name]
#             logger.debug(f"   Removed from cache: {module_name}")
