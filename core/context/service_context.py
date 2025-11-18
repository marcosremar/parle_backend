#!/usr/bin/env python3
"""
ServiceContext - One per service

Provides dependency injection for services:
- GPU Manager (from GlobalContext)
- Metrics Collector (from GlobalContext)
- Communication (from ProcessContext)
- Logger (scoped per service)
- Config (hierarchical: global + service overrides)
"""

import logging
from typing import Optional, Any, Dict
from pathlib import Path
import yaml

# Import SettingsService for dependency injection
from src.core.settings_service import SettingsService

logger = logging.getLogger(__name__)


class LoggerFactory:
    """Factory for creating scoped loggers"""

    @staticmethod
    def create(service_name: str, level: int = logging.INFO) -> logging.Logger:
        """
        Create a logger scoped to a service

        Args:
            service_name: Service identifier
            level: Logging level

        Returns:
            Configured logger instance
        """
        service_logger = logging.getLogger(f"services.{service_name}")
        service_logger.setLevel(level)

        # Add handler if not already present
        if not service_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - [{service_name}] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            service_logger.addHandler(handler)

        return service_logger


class ServiceContext:
    """
    One per service
    Injeta dependências no serviço

    Usage:
        # With full dependency injection:
        context = ServiceContext(
            service_name="llm",
            process_context=process_ctx
        )

        # With mocks (for testing):
        mock_gpu = MockGPUManager()
        context = ServiceContext(
            service_name="llm",
            process_context=process_ctx,
            gpu_manager=mock_gpu  # Inject mock
        )

        # In service:
        class LLMService:
            def __init__(self, context: Optional[ServiceContext] = None):
                if context:
                    self.gpu = context.gpu
                    self.logger = context.logger
                    self.comm = context.communication
                else:
                    # Backward compatibility (deprecated)
                    self.gpu = get_gpu_manager()
                    self.logger = logging.getLogger("llm")
                    self.comm = get_communication_manager()
    """

    def __init__(
        self,
        service_name: str,
        process_context,
        # Dependency injection (optional - for mocks)
        gpu_manager: Optional[Any] = None,
        communication: Optional[Any] = None,
        metrics: Optional[Any] = None,
        logger_instance: Optional[logging.Logger] = None,
        config: Optional[Dict] = None,
        settings_service: Optional[SettingsService] = None
    ) -> None:
        """
        Initialize ServiceContext

        Args:
            service_name: Service identifier (e.g., "llm", "tts")
            process_context: ProcessContext instance
            gpu_manager: GPU Manager instance (or mock)
            communication: Communication strategy (or mock)
            metrics: Metrics collector (or mock)
            logger_instance: Logger instance (or None for auto-create)
            config: Service config override (or None for auto-load)
            settings_service: SettingsService instance (or None for auto-create)
        """
        self.service_name = service_name
        self.process_context = process_context

        # Use injected dependencies or get from contexts (Decision #8: Constructor injection)
        self.gpu = gpu_manager or process_context.global_context.gpu_manager
        self.metrics = metrics or process_context.global_context.metrics
        self.communication = communication or process_context.communication

        # Service-specific resources
        self.logger = logger_instance or LoggerFactory.create(service_name)
        self.config = config or self._load_hierarchical_config()
        self.settings = settings_service or SettingsService.get_instance()

        self.logger.info(
            f"📦 ServiceContext created: {service_name}\n"
            f"   GPU: {'injected' if gpu_manager else 'from GlobalContext'}\n"
            f"   Communication: {type(self.communication).__name__ if self.communication else 'None'}\n"
            f"   Settings: {'injected' if settings_service else 'from singleton'}\n"
            f"   Config keys: {list(self.config.keys()) if self.config else []}"
        )

    def _load_hierarchical_config(self) -> dict:
        """
        Load hierarchical configuration (Decision #4: Hierarchical config)

        Merge strategy:
        1. Load global config from profile
        2. Load service-specific config
        3. Service config overrides global config

        Returns:
            Merged configuration dict
        """
        try:
            # Global config from profile
            global_config = {}
            if self.process_context.global_context.profile:
                # Profile is a dataclass, not a dict - get service_overrides for this service
                profile = self.process_context.global_context.profile
                if self.service_name in profile.service_overrides:
                    global_config = profile.service_overrides.get(self.service_name, {})

            # Service-specific config
            service_config = self._load_service_config()

            # Merge: service overrides global
            merged_config = {**global_config, **service_config}

            self.logger.debug(
                f"📋 Config loaded: "
                f"{len(global_config)} global + {len(service_config)} service = "
                f"{len(merged_config)} merged"
            )

            return merged_config

        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {e}")
            return {}

    def _load_service_config(self) -> dict:
        """
        Load service-specific configuration

        Looks for config in:
        1. src/services/{service_name}/config.yaml
        2. config/services/{service_name}.yaml

        Returns:
            Service configuration dict
        """
        # Try service directory first
        service_dir = Path(__file__).parent.parent.parent / "services" / self.service_name
        config_path = service_dir / "config.yaml"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    self.logger.debug(f"📄 Loaded config from: {config_path}")
                    return config or {}
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {config_path}: {e}")

        # Try global config directory
        global_config_path = Path(__file__).parent.parent.parent.parent / "config" / "services" / f"{self.service_name}.yaml"

        if global_config_path.exists():
            try:
                with open(global_config_path) as f:
                    config = yaml.safe_load(f)
                    self.logger.debug(f"📄 Loaded config from: {global_config_path}")
                    return config or {}
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {global_config_path}: {e}")

        # No config file found - return empty
        self.logger.debug(f"ℹ️ No config file found for {self.service_name}")
        return {}

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    async def shutdown(self) -> None:
        """Shutdown service context"""
        self.logger.info(f"🛑 Shutting down ServiceContext: {self.service_name}")

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

        self.logger.info(f"✅ ServiceContext shutdown complete: {self.service_name}")

    def get_status(self) -> dict:
        """Get service context status"""
        return {
            "service_name": self.service_name,
            "process_id": self.process_context.process_id,
            "gpu": "available" if self.gpu else "not available",
            "communication": type(self.communication).__name__ if self.communication else None,
            "metrics": "available" if self.metrics else "not available",
            "config_keys": list(self.config.keys()) if self.config else []
        }
