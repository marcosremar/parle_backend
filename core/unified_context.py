#!/usr/bin/env python3
"""
Unified Service Context (v5.3)

Consolidates 3-layer context system into single ServiceContext:
- OLD: GlobalContext → ProcessContext → ServiceContext (1104 lines)
- NEW: ServiceContext (unified, ~400 lines)

This is a dependency injection container that provides services with everything they need:
- Logger (scoped per service, using unified logging system)
- Communication Manager (automatic protocol selection)
- SettingsService (centralized configuration management)
- GPU Manager (optional, for GPU services)
- Metrics Collector (optional)
- Telemetry (OpenTelemetry integration)
- Configuration (hierarchical: global + service overrides)

Logging Changes (v5.3):
- Uses src.core.core_logging.setup_logging() for service loggers
- Consistent logging configuration across all services
- Automatic log rotation, retention, and compression
- Separate error logs and JSON logs for aggregators
- base_logger is kept for internal/utility logging only

Usage:
    # Create context for a service
    context = ServiceContext.create(
        service_name="llm",
        comm=comm_manager,
        profile="gpu-dev"
    )

    # Use in BaseService
    class LLMService(BaseService):
        def __init__(self, context: ServiceContext):
            super().__init__(context=context)
            # Now you have:
            # - self.logger (Unified logger with consistent formatting)
            # - self.comm (Communication Manager)
            # - self.settings (SettingsService singleton)
            # - self.gpu (GPU Manager, if available)
            # - self.metrics (Metrics Collector, if available)
            # - self.telemetry (OpenTelemetry integration)
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

# NEW: Use unified logging system instead of direct loguru import
from src.core.core_logging import get_logger, setup_logging

# For backward compatibility and non-service logging
from loguru import logger as base_logger

# Singleton managers (lazy-loaded)
_gpu_manager = None
_metrics_collector = None
_telemetry_instances = {}  # Per-service telemetry instances
_service_registry = {}
_telemetry_configured = False  # Flag to ensure configure_telemetry is called only once


@dataclass
class ResourceLimits:
    """Resource limits for a service/process"""
    max_cpu_percent: int = 80
    max_ram_mb: int = 4096
    max_gpu_mb: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'max_cpu_percent': self.max_cpu_percent,
            'max_ram_mb': self.max_ram_mb,
            'max_gpu_mb': self.max_gpu_mb
        }


@dataclass
class ServiceContext:
    """
    Unified Service Context - Single-layer dependency injection

    Replaces: GlobalContext, ProcessContext, ServiceContext (old 3-layer system)

    Provides dependency injection for:
    - Logger (Loguru, scoped per service)
    - Communication Manager (automatic protocol selection)
    - GPU Manager (optional, lazy-loaded)
    - Metrics Collector (optional, lazy-loaded)
    - Configuration (hierarchical)
    - Resource Limits
    """

    # Identity
    service_name: str
    process_id: str = "service_manager"

    # Core dependencies (injected automatically)
    logger: Any = None  # Loguru logger
    comm: Any = None  # ServiceCommunicationManager
    settings: Optional[Any] = None  # SettingsService (singleton)

    # Optional dependencies (lazy-loaded)
    gpu: Optional[Any] = None
    metrics: Optional[Any] = None
    telemetry: Optional[Any] = None  # UnifiedTelemetry (OpenTelemetry integration)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Profile and execution info
    profile: str = "dev-local"
    execution_mode: str = "module"  # "module", "internal", "external"

    # Resource limits
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    @classmethod
    def create(
        cls,
        service_name: str,
        comm: Any,
        config: Optional[Dict] = None,
        gpu: Optional[Any] = None,
        metrics: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        profile: str = "dev-local",
        execution_mode: str = "module",
        process_id: str = "service_manager",
        limits: Optional[Dict] = None
    ) -> 'ServiceContext':
        """
        Factory method to create ServiceContext with DI

        Args:
            service_name: Service identifier (e.g., "llm", "tts")
            comm: ServiceCommunicationManager instance (REQUIRED)
            config: Service configuration dict
            gpu: GPU Manager (or None for auto-load)
            metrics: Metrics Collector (or None for auto-load)
            profile: Profile name (dev-local, main-dev, gpu-prod, etc.)
            execution_mode: How service runs (module, internal, external)
            process_id: Process identifier
            limits: Resource limits dict

        Returns:
            ServiceContext instance with all dependencies injected
        """
        # Create scoped logger using unified logging system
        # This ensures consistent logging configuration across all services
        scoped_logger = setup_logging(
            service_name=service_name,
            level=os.getenv("LOG_LEVEL", "INFO")
        )

        # Get SettingsService singleton
        try:
            from src.core.settings_service import SettingsService
            settings_service = SettingsService.get_instance()
        except Exception as e:
            scoped_logger.debug(f"Could not load SettingsService: {e}")
            settings_service = None

        # Load configuration (hierarchical: settings.yaml + service config)
        merged_config = cls._load_config(service_name, config or {})

        # Create resource limits
        resource_limits = ResourceLimits(**limits) if limits else ResourceLimits()

        # Auto-load GPU Manager if not provided (lazy)
        if gpu is None and execution_mode in ("module", "internal"):
            gpu = cls._get_gpu_manager()

        # Auto-load Metrics Collector if not provided (lazy)
        if metrics is None:
            metrics = cls._get_metrics_collector()

        # Auto-load OpenTelemetry if not provided (lazy)
        if telemetry is None:
            telemetry = cls._get_telemetry(service_name)

        context = cls(
            service_name=service_name,
            process_id=process_id,
            logger=scoped_logger,
            comm=comm,
            settings=settings_service,
            gpu=gpu,
            metrics=metrics,
            telemetry=telemetry,
            config=merged_config,
            profile=profile,
            execution_mode=execution_mode,
            limits=resource_limits
        )

        # Register service in global registry
        cls._register_service(service_name, context)

        context.logger.info(
            f"📦 ServiceContext created",
            service=service_name,
            profile=profile,
            mode=execution_mode,
            settings="available" if settings_service else "not available",
            gpu="available" if gpu else "not available",
            metrics="available" if metrics else "not available",
            telemetry="available" if telemetry else "not available"
        )

        return context

    @staticmethod
    def _load_config(service_name: str, base_config: Dict) -> Dict:
        """
        Load hierarchical configuration

        Priority: settings.yaml (global) < service config < base_config

        Args:
            service_name: Service name
            base_config: Base configuration (highest priority)

        Returns:
            Merged configuration dict
        """
        merged = {}

        # 1. Load from global settings (lowest priority)
        try:
            from src.core.settings import get_settings
            settings = get_settings()
            # Check if there's a service-specific section
            service_settings = getattr(settings, service_name, None)
            if service_settings:
                merged.update(service_settings.model_dump())
        except Exception as e:
            base_logger.debug(f"Could not load global settings: {e}")

        # 2. Load service-specific config file (medium priority)
        service_config = ServiceContext._load_service_config_file(service_name)
        merged.update(service_config)

        # 3. Apply base_config (highest priority)
        merged.update(base_config)

        return merged

    @staticmethod
    def _load_service_config_file(service_name: str) -> Dict:
        """
        Load service-specific config.yaml file

        Looks in:
        1. src/services/{service_name}/config.yaml
        2. config/services/{service_name}.yaml

        Returns:
            Service configuration dict
        """
        import yaml

        # Try service directory first
        service_dir = Path(__file__).parent.parent / "services" / service_name
        config_path = service_dir / "config.yaml"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                base_logger.warning(f"Failed to load {config_path}: {e}")

        # Try global config directory
        global_config_path = Path(__file__).parent.parent.parent / "config" / "services" / f"{service_name}.yaml"

        if global_config_path.exists():
            try:
                with open(global_config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                base_logger.warning(f"Failed to load {global_config_path}: {e}")

        return {}

    @staticmethod
    def _get_gpu_manager():
        """
        Lazy-load GPU Manager (singleton)

        Returns:
            GPU Manager instance or None if not available
        """
        global _gpu_manager

        if _gpu_manager is None:
            try:
                from src.core.managers.gpu_memory_manager import get_gpu_manager
                _gpu_manager = get_gpu_manager()
                base_logger.info("✅ GPU Manager lazy-loaded")
            except Exception as e:
                base_logger.warning(f"⚠️ GPU Manager not available: {e}")
                _gpu_manager = None

        return _gpu_manager

    @staticmethod
    def _get_metrics_collector():
        """
        Lazy-load Metrics Collector (singleton)

        Returns:
            Metrics Collector instance or None if not available
        """
        global _metrics_collector

        if _metrics_collector is None:
            try:
                from src.core.metrics_collector import MetricsCollector
                _metrics_collector = MetricsCollector()
                base_logger.info("✅ Metrics Collector lazy-loaded")
            except ImportError:
                base_logger.debug("Metrics Collector not found, skipping")
                _metrics_collector = None
            except Exception as e:
                base_logger.warning(f"⚠️ Metrics Collector initialization failed: {e}")
                _metrics_collector = None

        return _metrics_collector

    @staticmethod
    def _get_telemetry(service_name: str):
        """
        Lazy-load OpenTelemetry (per-service instance)

        Each service gets its own telemetry instance with scoped logger/tracer.

        Args:
            service_name: Service name for telemetry scope

        Returns:
            UnifiedTelemetry instance or None if not available
        """
        global _telemetry_instances, _telemetry_configured

        # Configure telemetry globally on first use
        if not _telemetry_configured:
            try:
                from src.core.observability import configure_telemetry
                configure_telemetry(
                    service_name="ultravox-pipeline",
                    service_version="1.0.0",
                    environment=os.getenv("ULTRAVOX_PROFILE", "dev-local"),
                    enable_console=True,  # Enable console exporter for development
                    enable_prometheus=False  # Disable Prometheus (exporter not installed)
                )
                _telemetry_configured = True
                base_logger.info("✅ OpenTelemetry configured globally")
            except ImportError:
                base_logger.debug("OpenTelemetry SDK not installed, telemetry disabled")
                _telemetry_configured = True  # Mark as configured to avoid retrying
            except Exception as e:
                base_logger.warning(f"⚠️ OpenTelemetry configuration failed: {e}")
                _telemetry_configured = True  # Mark as configured to avoid retrying

        if service_name not in _telemetry_instances:
            try:
                from src.core.observability import get_telemetry
                telemetry = get_telemetry(service_name)
                _telemetry_instances[service_name] = telemetry
                base_logger.info(f"✅ OpenTelemetry lazy-loaded for {service_name}")
            except ImportError:
                base_logger.debug(f"OpenTelemetry not available for {service_name}")
                _telemetry_instances[service_name] = None
            except Exception as e:
                base_logger.warning(f"⚠️ OpenTelemetry initialization failed for {service_name}: {e}")
                _telemetry_instances[service_name] = None

        return _telemetry_instances[service_name]

    @staticmethod
    def _register_service(service_name: str, context: 'ServiceContext'):
        """Register service in global registry"""
        _service_registry[service_name] = {
            'context': context,
            'process_id': context.process_id,
            'profile': context.profile,
            'execution_mode': context.execution_mode,
            'pid': os.getpid()
        }

    @staticmethod
    def get_service_info(service_name: str) -> Optional[Dict]:
        """Get service info from global registry"""
        return _service_registry.get(service_name)

    @staticmethod
    def get_all_services() -> Dict[str, Dict]:
        """Get all registered services"""
        return _service_registry.copy()

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key (supports dot notation: "llm.model_name")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if '.' in key:
            # Support dot notation: "llm.model_name"
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        return default
                else:
                    return default
            return value
        else:
            return self.config.get(key, default)

    async def shutdown(self):
        """
        Shutdown service context and cleanup resources
        """
        self.logger.info("🛑 Shutting down ServiceContext", service=self.service_name)

        # Release GPU allocation if any
        if self.gpu and hasattr(self.gpu, 'release'):
            try:
                self.gpu.release(self.service_name)
                self.logger.info("   GPU released")
            except Exception as e:
                self.logger.error(f"   GPU release error: {e}")

        # Flush metrics
        if self.metrics and hasattr(self.metrics, 'flush'):
            try:
                await self.metrics.flush()
                self.logger.info("   Metrics flushed")
            except Exception as e:
                self.logger.error(f"   Metrics flush error: {e}")

        # Unregister from global registry
        if self.service_name in _service_registry:
            del _service_registry[self.service_name]

        self.logger.info("✅ ServiceContext shutdown complete", service=self.service_name)

    def get_status(self) -> Dict[str, Any]:
        """
        Get service context status

        Returns:
            Status dict with all context information
        """
        return {
            'service_name': self.service_name,
            'process_id': self.process_id,
            'pid': os.getpid(),
            'profile': self.profile,
            'execution_mode': self.execution_mode,
            'gpu': 'available' if self.gpu else 'not available',
            'communication': type(self.comm).__name__ if self.comm else None,
            'metrics': 'available' if self.metrics else 'not available',
            'telemetry': 'available' if self.telemetry else 'not available',
            'config_keys': list(self.config.keys()),
            'limits': self.limits.to_dict()
        }


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def create_service_context(
    service_name: str,
    comm: Any,
    **kwargs
) -> ServiceContext:
    """
    Create ServiceContext (backward compatible function)

    This function provides backward compatibility with old code that
    used GlobalContext/ProcessContext/ServiceContext three-layer system.

    Args:
        service_name: Service identifier
        comm: ServiceCommunicationManager instance
        **kwargs: Additional context parameters

    Returns:
        ServiceContext instance
    """
    return ServiceContext.create(service_name=service_name, comm=comm, **kwargs)


def get_service_registry() -> Dict[str, Dict]:
    """
    Get global service registry (backward compatible)

    Returns:
        Dict of all registered services
    """
    return _service_registry.copy()


async def shutdown_all_contexts():
    """
    Shutdown all service contexts (for graceful shutdown)
    """
    base_logger.info(f"🛑 Shutting down {len(_service_registry)} service contexts...")

    for service_name, service_info in list(_service_registry.items()):
        try:
            context = service_info.get('context')
            if context:
                await context.shutdown()
        except Exception as e:
            base_logger.error(f"Error shutting down {service_name}: {e}")

    # Clear global managers
    global _gpu_manager, _metrics_collector
    _gpu_manager = None
    _metrics_collector = None

    base_logger.info("✅ All contexts shutdown complete")


# ============================================================================
# DEPRECATED - Old 3-layer system compatibility
# ============================================================================

class GlobalContext:
    """DEPRECATED: Use ServiceContext instead"""

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "GlobalContext is deprecated. Use ServiceContext.create() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        base_logger.warning("⚠️ GlobalContext is deprecated, use ServiceContext")

    @classmethod
    def get_instance(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class ProcessContext:
    """DEPRECATED: Use ServiceContext instead"""

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "ProcessContext is deprecated. Use ServiceContext.create() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        base_logger.warning("⚠️ ProcessContext is deprecated, use ServiceContext")


# Alias for backward compatibility with BaseService
UnifiedServiceContext = ServiceContext
