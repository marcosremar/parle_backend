#!/usr/bin/env python3
"""
StandaloneContext - For standalone servers (no Service Manager)

Provides minimal dependency injection for services running independently:
- Logger (scoped per service)
- Config (loaded from service config files)
- No GPU Manager, Metrics, or Communication (standalone mode)
"""

import logging
from typing import Any, Dict, Optional
from pathlib import Path
import yaml

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


class StandaloneContext:
    """
    Context for standalone servers (no Service Manager)

    Provides minimal dependency injection:
    - Logger (scoped per service)
    - Config (from service config files)
    - No GPU Manager (standalone mode)
    - No Metrics (standalone mode)
    - No Communication Manager (uses HTTP clients directly)

    Usage:
        context = StandaloneContext(service_name="orchestrator")
        service = OrchestratorService(config={...}, context=context)
    """

    def __init__(
        self,
        service_name: str,
        config: Optional[Dict[str, Any]] = None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize StandaloneContext

        Args:
            service_name: Service identifier (e.g., "orchestrator", "external_llm")
            config: Service config override (or None for auto-load)
            logger_instance: Logger instance (or None for auto-create)
        """
        self.service_name = service_name

        # Service-specific resources
        self.logger = logger_instance or LoggerFactory.create(service_name)
        self.config = config or self._load_service_config()

        # Standalone mode - no shared resources
        self.gpu = None
        self.metrics = None
        self.communication = None

        self.logger.info(
            f"📦 StandaloneContext created: {service_name}\n"
            f"   Mode: Standalone (no Service Manager)\n"
            f"   Config keys: {list(self.config.keys()) if self.config else []}"
        )

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

    async def shutdown(self):
        """Shutdown service context"""
        self.logger.info(f"🛑 Shutting down StandaloneContext: {self.service_name}")
        self.logger.info(f"✅ StandaloneContext shutdown complete: {self.service_name}")

    def get_status(self) -> dict:
        """Get service context status"""
        return {
            "service_name": self.service_name,
            "mode": "standalone",
            "gpu": None,
            "communication": None,
            "metrics": None,
            "config_keys": list(self.config.keys()) if self.config else []
        }
