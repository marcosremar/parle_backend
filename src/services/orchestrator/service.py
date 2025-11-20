"""
Orchestrator Service
Exposes ConversationOrchestrator as a service
Coordinates conversation flow across LLM, TTS, STT services

Can run in two modes (determined by Service Manager):
- Internal: In-process within Service Manager
- External: As standalone HTTP server
"""

import sys
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Logging and Metrics (after adding to path)
from loguru import logger

from .utils.base_service import BaseService
from src.services.database.database_client import UserDatabase
from src.services.orchestrator.orchestrator_engine import ConversationOrchestrator
from src.core.shared.models.config_models import OrchestratorConfig

# Context system (NEW)
from src.services.orchestrator.utils.context import ServiceContext

try:
    from .database_config import DATABASE_CONFIG, ACTIVE_USER_CACHE_SIZE, ACTIVE_USER_TTL_MINUTES
except ImportError:
    from database_config import DATABASE_CONFIG, ACTIVE_USER_CACHE_SIZE, ACTIVE_USER_TTL_MINUTES

class OrchestratorService(BaseService):
    """
    Orchestrator Service

    Provides conversation orchestration as an internal service.
    Wraps ConversationOrchestrator to run in-process within Service Manager.

    Uses ServiceContext for dependency injection (REQUIRED)
    """

    def __init__(self, config: Dict[str, Any] = None, context: Optional[ServiceContext] = None) -> None:
        """
        Initialize Orchestrator Service

        Args:
            config: Service configuration (optional)
            context: ServiceContext for dependency injection (optional, recommended)
        """
        # Pass context to BaseService (DI support)
        super().__init__(context=context, config=config)

        # Log DI status
        if self.context:
            self.logger.info("🎯 Orchestrator Service initialized with ServiceContext (DI enabled)")
            self.logger.info(f"   - Logger: ✅ injected (scoped)")
            self.logger.info(f"   - Communication: ✅ injected ({type(self.comm).__name__})")
            self.logger.info(f"   - GPU Manager: ✅ injected")
            self.logger.info(f"   - Metrics: ✅ injected")
            self.logger.info(f"   - Settings: ✅ injected (SettingsService)")
        else:
            self.logger.warning("⚠️  Orchestrator Service initialized without ServiceContext (legacy mode)")

        # Load configuration from SettingsService (via DI)
        self.orchestrator_config: Optional[OrchestratorConfig] = None
        if self.settings:
            try:
                self.orchestrator_config = OrchestratorConfig.from_settings(self.settings)
                self.logger.info(f"✅ Orchestrator Configuration loaded via SettingsService")
            except Exception as e:
                self.logger.error(f"Failed to load Orchestrator config from SettingsService: {e}")
                self.orchestrator_config = OrchestratorConfig()  # Use defaults

        self.orchestrator: Optional[ConversationOrchestrator] = None
        # Active user database cache (warm start)
        self.active_users: Dict[str, UserDatabase] = {}
        self.last_access: Dict[str, float] = {}
        # Background health check task (for observability)
        self.health_check_task: Optional[asyncio.Task] = None
        self.health_check_status: str = "not_started"
        self.health_check_results: Optional[Dict] = None

        # Active processing tasks (for barge-in cancellation support)
        self.active_tasks: Dict[str, asyncio.Task] = {}  # session_id -> task
        self.cancellation_stats = {
            "total_cancellations": 0,
            "cancelled_sessions": []
        }

    async def initialize(self) -> bool:
        """Initialize orchestrator engine with optional background health checks"""
        try:
            import asyncio

            self.logger.info("🎯 Initializing Orchestrator Service...")

            # Get configuration from SettingsService (via OrchestratorConfig)
            startup_mode = self.orchestrator_config.startup_mode if self.orchestrator_config else False
            in_process_mode = self.orchestrator_config.in_process_mode if self.orchestrator_config else False

            if in_process_mode:
                self.logger.info("⚡ IN-PROCESS MODE enabled (ultra-low latency)")
            else:
                self.logger.info("🌐 HTTP MODE enabled (with failover)")

            # Create orchestrator with in-process mode setting
            self.orchestrator = ConversationOrchestrator(in_process_mode=in_process_mode)

            # If we have Communication Manager from DI, pass it to orchestrator
            if self.comm:
                self.logger.info("📡 Passing Communication Manager from DI to orchestrator")
                # Note: ConversationOrchestrator will use Communication Manager for service calls
                self.orchestrator.comm_manager = self.comm

            # Initialize orchestrator (this will call _health_check_services internally)
            # We need to temporarily set STARTUP_MODE for the orchestrator too
            if startup_mode:
                self.logger.info("🏁 Startup mode: Initializing without health checks (background mode)")
                os.environ["ORCHESTRATOR_SKIP_HEALTH_CHECKS"] = "true"

            await self.orchestrator.initialize()

            if startup_mode:
                os.environ.pop("ORCHESTRATOR_SKIP_HEALTH_CHECKS", None)

            self.initialized = True

            # Run health checks in background if in startup mode
            if startup_mode:
                self.logger.info("📡 Scheduling background health checks...")
                self.health_check_task = asyncio.create_task(self._background_health_checks())
                self.logger.info("   Background task created: asyncio.Task (async I/O)")
                self.logger.info("   Check status: GET /services/orchestrator/health-check-status")

            mode_summary = "IN-PROCESS" if self.orchestrator.in_process_mode else "HTTP"
            comm_status = "DI-injected" if self.comm else "self-managed"
            self.logger.info(f"✅ Orchestrator Service initialized successfully")
            self.logger.info(f"   - Mode: {mode_summary}")
            self.logger.info(f"   - Communication: {comm_status}")
            if startup_mode:
                self.logger.info(f"   - Health checks: Background mode (non-blocking)")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Orchestrator Service: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _background_health_checks(self) -> None:
        """Run health checks in background (non-blocking)"""
        import asyncio
        try:
            self.health_check_status = "waiting"
            # Wait a bit for other services to finish starting
            await asyncio.sleep(2)

            self.health_check_status = "running"
            self.logger.info("🔍 Running background health checks...")
            health_status = await self.orchestrator._health_check_services()

            healthy_count = sum(1 for v in health_status.values() if v)
            total_count = len(health_status)

            self.health_check_status = "completed"
            self.health_check_results = {
                "healthy_count": healthy_count,
                "total_count": total_count,
                "services": health_status,
                "all_healthy": healthy_count == total_count
            }

            self.logger.info(f"✅ Background health checks complete: {healthy_count}/{total_count} services healthy")

        except Exception as e:
            self.health_check_status = "failed"
            self.health_check_results = {"error": str(e)}
            self.logger.warning(f"⚠️  Background health checks failed (non-critical): {e}")

    def get_health_check_status(self) -> Dict[str, Any]:
        """Get background health check task status (for observability)"""
        task_status = "idle"
        if self.health_check_task:
            if self.health_check_task.done():
                if self.health_check_task.cancelled():
                    task_status = "cancelled"
                elif self.health_check_task.exception():
                    task_status = "failed"
                else:
                    task_status = "completed"
            else:
                task_status = "running"

        return {
            "health_check_status": self.health_check_status,
            "task_status": task_status,
            "results": self.health_check_results,
            "background_task_info": {
                "type": "asyncio.Task",
                "note": "Health checks use asyncio for non-blocking service validation",
                "observability": "Check this endpoint for status"
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "service": "orchestrator",
            "status": "healthy" if self.initialized else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "orchestrator_ready": self.orchestrator is not None,
            "stats": self.orchestrator.stats if self.orchestrator else {}
        }

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self.logger.info("🛑 Shutting down Orchestrator Service...")
        if self.orchestrator:
            await self.orchestrator.cleanup()
        self.logger.info("✅ Orchestrator Service shut down successfully")

    def _setup_router(self) -> None:
        """Setup FastAPI routes using the new modular structure"""
        # Import and attach the router created by the routes module
        from .routes import create_router
        router = create_router(self)

        # Include all routes from the modular router
        self.router.include_router(router)

    def register_processing_task(self, session_id: str, task: asyncio.Task) -> None:
        """
        Register an active processing task for potential cancellation

        Args:
            session_id: Session identifier
            task: Asyncio task performing processing (LLM/TTS)
        """
        self.active_tasks[session_id] = task
        self.logger.debug(f"📋 Registered processing task for session {session_id}")

    def unregister_processing_task(self, session_id: str) -> None:
        """
        Unregister processing task when completed

        Args:
            session_id: Session identifier
        """
        if session_id in self.active_tasks:
            del self.active_tasks[session_id]
            self.logger.debug(f"✅ Unregistered processing task for session {session_id}")

    async def cancel_processing(self, session_id: str) -> Dict[str, Any]:
        """
        Cancel active processing for a session (LLM/TTS)

        Args:
            session_id: Session to cancel

        Returns:
            Dict with cancellation result
        """
        if session_id not in self.active_tasks:
            self.logger.warning(f"⚠️  No active processing found for session {session_id}")
            return {
                "success": False,
                "session_id": session_id,
                "error": "No active processing found for this session"
            }

        task = self.active_tasks[session_id]

        if task.done():
            self.logger.info(f"⚠️  Task already completed for session {session_id}")
            del self.active_tasks[session_id]
            return {
                "success": False,
                "session_id": session_id,
                "error": "Processing already completed"
            }

        # Cancel the task
        self.logger.info(f"🛑 Cancelling processing for session {session_id}")
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            self.logger.info(f"✅ Processing cancelled successfully for session {session_id}")

        # Update stats
        self.cancellation_stats["total_cancellations"] += 1
        self.cancellation_stats["cancelled_sessions"].append({
            "session_id": session_id,
            "cancelled_at": datetime.now().isoformat()
        })

        # Keep only last 100 cancellations
        if len(self.cancellation_stats["cancelled_sessions"]) > 100:
            self.cancellation_stats["cancelled_sessions"].pop(0)

        # Clean up
        if session_id in self.active_tasks:
            del self.active_tasks[session_id]

        return {
            "success": True,
            "session_id": session_id,
            "message": "Processing cancelled successfully"
        }

    async def _get_or_preload_user_db(self, user_id: str) -> Optional[UserDatabase]:
        """
        Get or preload user database with eager loading strategy

        Args:
            user_id: User identifier

        Returns:
            UserDatabase instance or None
        """
        import time
        from src.services.database.database_metrics import record_cache_hit, record_cache_miss, record_cache_eviction

        start_time = time.time()

        # Check if already in cache
        if user_id in self.active_users:
            self.last_access[user_id] = time.time()
            latency_ms = (time.time() - start_time) * 1000
            record_cache_hit(latency_ms)
            return self.active_users[user_id]

        # Cache miss
        latency_ms = (time.time() - start_time) * 1000
        record_cache_miss(latency_ms)

        # Evict least recently used if cache is full
        if len(self.active_users) >= ACTIVE_USER_CACHE_SIZE:
            lru_user = min(self.last_access, key=self.last_access.get)
            del self.active_users[lru_user]
            del self.last_access[lru_user]
            record_cache_eviction()
            logger.info(f"Evicted user {lru_user} from cache (LRU)")

        # Preload user database (eager strategy)
        try:
            db = UserDatabase(user_id, **DATABASE_CONFIG)

            # Preload data into memory (eager)
            await db.get_info()

            # Cache for future requests
            self.active_users[user_id] = db
            self.last_access[user_id] = time.time()

            logger.info(f"Pre-loaded user {user_id} into cache (eager strategy)")
            return db

        except Exception as e:
            logger.error(f"Failed to preload user database for {user_id}: {e}")
            return None
