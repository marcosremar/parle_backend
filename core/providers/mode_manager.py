"""
Mode Manager - Easy switching between dev, local, and production modes
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .factory import ProviderFactory

logger = logging.getLogger(__name__)


class ModeManager:
    """
    Manages environment modes (dev, local, production)
    Each mode has its own set of providers configured
    """

    def __init__(self, settings_path: str = "settings.yaml"):
        """
        Initialize Mode Manager

        Args:
            settings_path: Path to settings.yaml file
        """
        self.settings_path = Path(settings_path)
        self.settings = self._load_settings()
        self.current_mode = self._get_current_mode()
        self.providers = None

        logger.info(f"🎯 Mode Manager initialized in '{self.current_mode}' mode")

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from YAML file"""
        try:
            with open(self.settings_path, 'r') as f:
                settings = yaml.safe_load(f)

            # Replace environment variables
            self._replace_env_vars(settings)
            return settings

        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            raise

    def _replace_env_vars(self, obj):
        """Recursively replace ${VAR} with environment variables"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    obj[key] = os.getenv(env_var, value)
                else:
                    self._replace_env_vars(value)
        elif isinstance(obj, list):
            for item in obj:
                self._replace_env_vars(item)

    def _get_current_mode(self) -> str:
        """
        Get current mode from environment or settings

        Priority:
        1. ULTRAVOX_MODE environment variable
        2. mode in settings.yaml
        3. Default to 'dev'
        """
        # Check environment variable first
        env_mode = os.getenv("ULTRAVOX_MODE")
        if env_mode:
            logger.info(f"Mode set from environment: {env_mode}")
            return self._resolve_alias(env_mode)

        # Use settings mode
        settings_mode = self.settings.get("mode", "dev")
        return self._resolve_alias(settings_mode)

    def _resolve_alias(self, mode: str) -> str:
        """Resolve mode aliases (e.g., 'test' -> 'dev')"""
        aliases = self.settings.get("mode_aliases", {})
        return aliases.get(mode, mode)

    def get_mode_config(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific mode

        Args:
            mode: Mode name (if None, uses current mode)

        Returns:
            Mode configuration dictionary
        """
        mode = mode or self.current_mode
        mode = self._resolve_alias(mode)

        modes_config = self.settings.get("modes", {})
        if mode not in modes_config:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(modes_config.keys())}")

        return modes_config[mode]

    def get_providers(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Get providers for a specific mode

        Args:
            mode: Mode name (if None, uses current mode)

        Returns:
            Dictionary with 'llm', 'tts', 'stt' provider instances
        """
        if mode is None and self.providers is not None:
            return self.providers

        mode_config = self.get_mode_config(mode)
        providers_config = {"providers": mode_config.get("providers", {})}

        # Create providers using factory
        self.providers = ProviderFactory.create_from_settings(providers_config)

        return self.providers

    def switch_mode(self, new_mode: str) -> Dict[str, Any]:
        """
        Switch to a different mode

        Args:
            new_mode: Name of the mode to switch to

        Returns:
            New providers dictionary
        """
        new_mode = self._resolve_alias(new_mode)

        if new_mode not in self.settings.get("modes", {}):
            raise ValueError(f"Invalid mode: {new_mode}")

        logger.info(f"🔄 Switching from '{self.current_mode}' to '{new_mode}' mode")

        # Clean up old providers if they exist
        if self.providers:
            for provider in self.providers.values():
                if hasattr(provider, 'cleanup'):
                    provider.cleanup()

        self.current_mode = new_mode
        self.providers = None  # Reset providers

        return self.get_providers()

    def list_modes(self) -> Dict[str, str]:
        """
        List all available modes with descriptions

        Returns:
            Dictionary of mode names and descriptions
        """
        modes = {}
        modes_config = self.settings.get("modes", {})

        for mode_name, mode_config in modes_config.items():
            description = mode_config.get("description", "No description")
            modes[mode_name] = description

        # Add aliases
        aliases = self.settings.get("mode_aliases", {})
        for alias, target in aliases.items():
            if target in modes:
                modes[f"{alias} (→{target})"] = modes[target]

        return modes

    def get_mode_info(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a mode

        Args:
            mode: Mode name (if None, uses current mode)

        Returns:
            Detailed mode information
        """
        mode = mode or self.current_mode
        mode_config = self.get_mode_config(mode)

        info = {
            "name": mode,
            "description": mode_config.get("description", ""),
            "providers": {}
        }

        # Get provider details
        for provider_type in ["llm", "tts", "stt"]:
            if provider_type in mode_config.get("providers", {}):
                provider_config = mode_config["providers"][provider_type]
                info["providers"][provider_type] = {
                    "type": provider_config.get("provider"),
                    "model": provider_config.get("model", "N/A"),
                    "details": {k: v for k, v in provider_config.items()
                              if k not in ["api_key", "subscription_key"]}
                }

        return info

    def __repr__(self) -> str:
        return f"ModeManager(mode='{self.current_mode}')"


# Singleton instance
_mode_manager = None


def get_mode_manager(settings_path: str = "settings.yaml") -> ModeManager:
    """
    Get or create the global ModeManager instance

    Args:
        settings_path: Path to settings file

    Returns:
        ModeManager instance
    """
    global _mode_manager
    if _mode_manager is None:
        _mode_manager = ModeManager(settings_path)
    return _mode_manager