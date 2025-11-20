#!/usr/bin/env python3
"""
GlobalContext - Singleton per machine

Manages resources shared across ALL processes:
- GPU Manager (with SharedGPUState)
- Metrics Collector
- Service Registry
- Profile Manager
"""

import threading
import logging
import asyncio
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GlobalContext:
    """
    Singleton per machine
    Gerencia recursos compartilhados entre TODOS os processos

    Usage:
        # In ServiceManager initialization:
        global_ctx = GlobalContext.get_instance(profile_name="gpu-machine")
        await global_ctx.initialize()

        # In services:
        gpu = global_ctx.gpu_manager
        metrics = global_ctx.metrics
    """

    _instance: Optional['GlobalContext'] = None
    _lock = threading.Lock()
    _initialized = False

    def __init__(self, profile_name: str = "development") -> None:
        """
        Initialize GlobalContext

        Args:
            profile_name: Profile to load (gpu-machine, testing, production)
        """
        if GlobalContext._instance is not None:
            raise RuntimeError(
                "GlobalContext is a singleton. Use GlobalContext.get_instance()"
            )

        self.profile_name = profile_name
        self.profile = None

        # Shared resources (initialized in initialize())
        self.gpu_manager = None
        self.metrics = None
        self.service_registry = None

        logger.info(f"🌍 GlobalContext created with profile: {profile_name}")

    @classmethod
    def get_instance(
        cls,
        profile_name: Optional[str] = None
    ) -> 'GlobalContext':
        """
        Get or create GlobalContext singleton

        Args:
            profile_name: Profile name (only used on first call)

        Returns:
            GlobalContext instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    profile = profile_name or "development"
                    cls._instance = GlobalContext(profile)
                    logger.info(f"✅ GlobalContext instance created: {profile}")

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset singleton (for testing or hot reload)

        WARNING: Only use during controlled shutdown/restart
        """
        with cls._lock:
            if cls._instance:
                logger.warning("🔄 Resetting GlobalContext singleton")
                cls._instance = None
                cls._initialized = False

    async def initialize(self):
        """
        Eager initialization (Decision #5: Hybrid - infra eager)

        Initializes:
        - Profile Manager → loads profile configuration
        - GPU Manager → detects GPU, creates SharedGPUState
        - Metrics Collector → connects to Prometheus/StatsD
        """
        if GlobalContext._initialized:
            logger.info("ℹ️ GlobalContext already initialized")
            return

        logger.info("🚀 Initializing GlobalContext (eager)...")

        # 1. Load Profile
        await self._load_profile()

        # 2. Initialize GPU Manager (EAGER)
        await self._initialize_gpu_manager()

        # 3. Initialize Metrics (EAGER)
        await self._initialize_metrics()

        # 4. Initialize Service Registry
        await self._initialize_service_registry()

        GlobalContext._initialized = True
        logger.info("✅ GlobalContext initialization complete")

    async def _load_profile(self):
        """Load profile configuration"""
        try:
            # Import ProfileManager
            from ..managers.profile_manager import get_profile_manager

            profile_manager = get_profile_manager()
            self.profile = profile_manager.get_profile(self.profile_name)

            logger.info(
                f"📋 Profile loaded: {self.profile_name}\n"
                f"   Description: {self.profile.description if self.profile else 'N/A'}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to load profile {self.profile_name}: {e}")
            # Use default profile
            self.profile = {"name": self.profile_name, "services": []}

    async def _initialize_gpu_manager(self):
        """Initialize GPU Manager with SharedGPUState"""
        try:
            # Import GPU Manager and SharedGPUState
            from .shared_state import SharedGPUState
            from ..managers.gpu_memory_manager import get_gpu_manager

            # Create shared state
            shared_state = SharedGPUState()

            # Get GPU Manager singleton
            self.gpu_manager = get_gpu_manager()

            logger.info("✅ GPU Manager initialized with SharedGPUState")

        except Exception as e:
            logger.error(f"❌ GPU Manager initialization failed: {e}")
            # Create fallback - no GPU manager
            self.gpu_manager = None
            logger.warning("⚠️  Continuing without GPU Manager")

    async def _initialize_metrics(self):
        """Initialize Metrics Collector"""
        try:
            # Import MetricsCollector (if exists)
            try:
                from ..metrics_collector import MetricsCollector
                self.metrics = MetricsCollector()
                await self.metrics.initialize()
                logger.info("✅ Metrics Collector initialized")
            except ImportError:
                logger.warning("⚠️ MetricsCollector not found, skipping")
                self.metrics = None

        except Exception as e:
            logger.error(f"❌ Metrics initialization failed: {e}")
            self.metrics = None

    async def _initialize_service_registry(self):
        """Initialize Service Registry"""
        try:
            # Simple in-memory registry for now
            self.service_registry = {}
            logger.info("✅ Service Registry initialized")

        except Exception as e:
            logger.error(f"❌ Service Registry initialization failed: {e}")
            self.service_registry = {}

    def register_service(self, service_name: str, service_info: dict) -> None:
        """
        Register a service in the global registry

        Args:
            service_name: Service name (e.g., "llm", "tts")
            service_info: Service metadata (pid, port, status, etc.)
        """
        if self.service_registry is not None:
            self.service_registry[service_name] = service_info
            logger.info(f"📝 Service registered: {service_name}")

    def unregister_service(self, service_name: str) -> None:
        """Unregister a service"""
        if self.service_registry is not None and service_name in self.service_registry:
            del self.service_registry[service_name]
            logger.info(f"🗑️ Service unregistered: {service_name}")

    def get_service(self, service_name: str) -> Optional[dict]:
        """Get service info from registry"""
        if self.service_registry is not None:
            return self.service_registry.get(service_name)
        return None

    async def shutdown(self):
        """
        Shutdown GlobalContext

        Called during ServiceManager shutdown
        """
        logger.info("🛑 Shutting down GlobalContext...")

        # Shutdown GPU Manager
        if self.gpu_manager:
            try:
                # Release all GPU allocations
                if hasattr(self.gpu_manager, 'shared_state') and self.gpu_manager.shared_state:
                    self.gpu_manager.shared_state.clear_all()
                    logger.info("   GPU allocations cleared")
            except Exception as e:
                logger.error(f"   GPU Manager shutdown error: {e}")

        # Shutdown Metrics
        if self.metrics:
            try:
                await self.metrics.shutdown()
                logger.info("   Metrics shutdown complete")
            except Exception as e:
                logger.error(f"   Metrics shutdown error: {e}")

        logger.info("✅ GlobalContext shutdown complete")

    def get_status(self) -> dict:
        """Get GlobalContext status"""
        return {
            "profile": self.profile_name,
            "initialized": GlobalContext._initialized,
            "gpu_manager": "initialized" if self.gpu_manager else "not available",
            "metrics": "initialized" if self.metrics else "not available",
            "services_count": len(self.service_registry) if self.service_registry else 0,
            "services": list(self.service_registry.keys()) if self.service_registry else []
        }
