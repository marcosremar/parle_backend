#!/usr/bin/env python3
"""
SettingsService - Centralized Configuration Management with Dependency Injection

⚠️  DEPRECATED as of v5.2 - Use src.core.config instead!

This module is deprecated. Please migrate to the new unified configuration system:

    # OLD (deprecated):
    from src.core.settings_service import SettingsService
    settings = SettingsService.get_instance()

    # NEW (recommended):
    from src.core.config import get_config
    config = get_config()
    value = config.get_str("KEY", required=False)

See docs/CONFIG_MIGRATION_GUIDE.md for migration instructions.

Legacy Documentation:
---------------------
Provides a single source of truth for all application configuration:
- Loads from .env file (lowest priority)
- Loads from settings.yaml (medium priority)
- Loads from service-specific config files (highest priority)
- Loads from environment variables (overrides everything)

Usage (DEPRECATED):
    # In services with DI:
    class MyService(BaseService):
        async def initialize(self) -> bool:
            # Access settings via ServiceContext
            logs_dir = self.context.settings.get_str(
                "LOGS_DIR",
                default=f"{Path.home()}/.cache/ultravox-pipeline/logs"
            )

            hf_token = self.context.settings.get_str("HF_TOKEN", required=False)

            # Type-safe getters
            max_workers = self.context.settings.get_int("MAX_WORKERS", default=6)
            debug_mode = self.context.settings.get_bool("DEBUG", default=False)
"""

import os
import logging
import threading
from typing import Optional, Any, Dict, List
from pathlib import Path
from functools import lru_cache
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class SettingsService:
    """
    Centralized settings service for dependency injection

    Thread-safe singleton that manages all configuration loading
    and access for the application.
    """

    _instance: Optional['SettingsService'] = None
    _lock = threading.Lock()

    def __init__(self, env_file: Optional[Path] = None, settings_file: Optional[Path] = None):
        """
        Initialize SettingsService

        Args:
            env_file: Path to .env file (defaults to project root)
            settings_file: Path to settings.yaml (defaults to project root)
        """
        # Determine project root
        project_root = Path(__file__).parent.parent.parent

        self.env_file = env_file or (project_root / ".env")
        self.settings_file = settings_file or (project_root / "settings.yaml")

        # Cache for loaded settings
        self._cache: Dict[str, Any] = {}
        self._loaded = False

        logger.info(
            f"🔧 SettingsService initialized\n"
            f"   .env: {self.env_file}\n"
            f"   settings.yaml: {self.settings_file}"
        )

    @classmethod
    def get_instance(
        cls,
        env_file: Optional[Path] = None,
        settings_file: Optional[Path] = None
    ) -> 'SettingsService':
        """
        Get or create SettingsService singleton

        Args:
            env_file: Path to .env file
            settings_file: Path to settings.yaml

        Returns:
            SettingsService instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SettingsService(env_file, settings_file)
                    cls._instance.load_all()

        return cls._instance

    def load_all(self) -> None:
        """
        Load all configuration from all sources

        Priority order (highest to lowest):
        1. Environment variables (ULTRAVOX_* or exact match)
        2. settings.yaml
        3. .env file
        """
        if self._loaded:
            return

        try:
            # Step 1: Load .env file
            self._load_env_file()

            # Step 2: Load settings.yaml
            self._load_settings_yaml()

            # Step 3: Load environment variable overrides
            self._load_env_overrides()

            self._loaded = True
            logger.info(f"✅ SettingsService loaded {len(self._cache)} configuration values")

        except Exception as e:
            logger.error(f"❌ Failed to load settings: {e}")
            raise

    def _load_env_file(self) -> None:
        """Load configuration from .env file"""
        if not self.env_file.exists():
            logger.warning(f"⚠️  .env file not found: {self.env_file}")
            return

        try:
            load_dotenv(self.env_file, override=False)
            # Also cache the values
            for key, value in os.environ.items():
                self._cache[key] = value
            logger.debug(f"📄 Loaded .env from: {self.env_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load .env: {e}")

    def _load_settings_yaml(self) -> None:
        """Load configuration from settings.yaml"""
        if not self.settings_file.exists():
            logger.debug(f"ℹ️  settings.yaml not found: {self.settings_file}")
            return

        try:
            with open(self.settings_file) as f:
                settings = yaml.safe_load(f) or {}

            # Flatten and cache YAML settings
            self._flatten_dict(settings, prefix="")
            logger.debug(f"📄 Loaded settings.yaml from: {self.settings_file}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to load settings.yaml: {e}")

    def _flatten_dict(self, d: Dict, prefix: str = "") -> None:
        """
        Flatten nested dictionary into cache

        Args:
            d: Dictionary to flatten
            prefix: Key prefix for nested values
        """
        for key, value in d.items():
            full_key = f"{prefix}{key}".upper() if not prefix else f"{prefix}_{key}".upper()

            if isinstance(value, dict):
                self._flatten_dict(value, full_key)
            else:
                self._cache[full_key] = value

    def _load_env_overrides(self) -> None:
        """Load and override with environment variables"""
        # Check for ULTRAVOX_* prefixed variables
        for key, value in os.environ.items():
            if key.startswith("ULTRAVOX_"):
                # Store without prefix
                setting_key = key.replace("ULTRAVOX_", "", 1)
                self._cache[setting_key] = value
            else:
                # Also store original keys (they may have been set by .env)
                self._cache[key] = value

    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get configuration value with automatic type inference

        Args:
            key: Configuration key (case-insensitive)
            default: Default value if not found
            required: If True, raise KeyError if not found

        Returns:
            Configuration value or default

        Raises:
            KeyError: If required=True and key not found
        """
        key_upper = key.upper()

        if key_upper not in self._cache:
            if required:
                raise KeyError(f"Required configuration not found: {key}")
            return default

        return self._cache[key_upper]

    def get_str(self, key: str, default: str = "", required: bool = False) -> str:
        """
        Get string configuration value

        Args:
            key: Configuration key
            default: Default string value
            required: If True, raise KeyError if not found

        Returns:
            String configuration value
        """
        value = self.get(key, default, required)
        return str(value) if value is not None else default

    def get_int(self, key: str, default: int = 0, required: bool = False) -> int:
        """
        Get integer configuration value

        Args:
            key: Configuration key
            default: Default integer value
            required: If True, raise KeyError if not found

        Returns:
            Integer configuration value
        """
        value = self.get(key, default, required)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️  Could not convert {key}={value} to int, using default={default}")
            return default

    def get_float(self, key: str, default: float = 0.0, required: bool = False) -> float:
        """
        Get float configuration value

        Args:
            key: Configuration key
            default: Default float value
            required: If True, raise KeyError if not found

        Returns:
            Float configuration value
        """
        value = self.get(key, default, required)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️  Could not convert {key}={value} to float, using default={default}")
            return default

    def get_bool(self, key: str, default: bool = False, required: bool = False) -> bool:
        """
        Get boolean configuration value

        Args:
            key: Configuration key
            default: Default boolean value
            required: If True, raise KeyError if not found

        Returns:
            Boolean configuration value
        """
        value = self.get(key, default, required)
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        # String parsing
        return str(value).lower() in ("true", "1", "yes", "on")

    def get_list(self, key: str, default: Optional[List] = None, required: bool = False) -> List:
        """
        Get list configuration value (comma-separated string or YAML list)

        Args:
            key: Configuration key
            default: Default list value
            required: If True, raise KeyError if not found

        Returns:
            List configuration value
        """
        value = self.get(key, default, required)

        if value is None:
            return default or []

        if isinstance(value, list):
            return value

        # Parse comma-separated string
        return [item.strip() for item in str(value).split(",")]

    def get_dict(self, key: str, default: Optional[Dict] = None, required: bool = False) -> Dict:
        """
        Get dictionary configuration value

        Args:
            key: Configuration key
            default: Default dict value
            required: If True, raise KeyError if not found

        Returns:
            Dictionary configuration value
        """
        value = self.get(key, default, required)

        if value is None:
            return default or {}

        if isinstance(value, dict):
            return value

        # Log warning if trying to parse non-dict
        logger.warning(f"⚠️  Expected dict for {key}, got {type(value).__name__}")
        return default or {}

    def get_path(self, key: str, default: Optional[Path] = None, required: bool = False) -> Path:
        """
        Get Path configuration value (with variable expansion)

        Args:
            key: Configuration key
            default: Default Path value
            required: If True, raise KeyError if not found

        Returns:
            Path configuration value with expanded variables
        """
        value = self.get_str(key, required=required)

        if not value:
            return default or Path(".")

        # Expand environment variables in path
        expanded = os.path.expandvars(value)
        expanded = os.path.expanduser(expanded)

        return Path(expanded)

    def has(self, key: str) -> bool:
        """
        Check if configuration key exists

        Args:
            key: Configuration key

        Returns:
            True if key exists, False otherwise
        """
        return key.upper() in self._cache

    def all(self) -> Dict[str, Any]:
        """
        Get all configuration values (read-only)

        Returns:
            Dictionary of all configuration values
        """
        return dict(self._cache)

    def reload(self) -> None:
        """
        Reload all configuration from sources
        Useful for testing or dynamic reloads
        """
        self._cache.clear()
        self._loaded = False
        self.load_all()

    # ========================================================================
    # Port Resolution - Integration with PORT_MATRIX
    # ========================================================================

    def get_service_port(self, service_name: str, default: Optional[int] = None) -> int:
        """
        Get port for a service with fallback chain

        Resolution order:
        1. Environment variable: {SERVICE_NAME}_PORT (highest priority)
        2. PORT_MATRIX from service_config.py
        3. Default parameter
        4. Raise KeyError if required

        Args:
            service_name: Service name (e.g., "api_gateway", "external_stt")
            default: Default port if not found

        Returns:
            Port number

        Raises:
            KeyError: If port not found and no default provided

        Example:
            port = settings.get_service_port("api_gateway")
            # Looks for: API_GATEWAY_PORT env var → PORT_MATRIX → error
        """
        # Try environment variable first
        env_var = f"{service_name.upper()}_PORT"
        port_str = self.get_str(env_var, required=False)
        if port_str:
            try:
                return int(port_str)
            except (ValueError, TypeError):
                logger.warning(f"⚠️  Invalid port in {env_var}={port_str}, trying PORT_MATRIX")

        # Try PORT_MATRIX
        try:
            from src.config.service_config import ServiceType, ServiceRegistry

            # Map service name to ServiceType enum
            service_type_map = {
                "api_gateway": ServiceType.API_GATEWAY,
                "orchestrator": ServiceType.ORCHESTRATOR,
                "session": ServiceType.SESSION_SERVICE,
                "user": ServiceType.USER_SERVICE,
                "external_stt": ServiceType.EXTERNAL_STT,
                "external_llm": ServiceType.EXTERNAL_LLM,
                "external_ultravox": ServiceType.EXTERNAL_ULTRAVOX,
                "llm": ServiceType.LLM_SERVICE,
                "tts": ServiceType.TTS_SERVICE,
                "stt": ServiceType.STT_SERVICE,
                "conversation_store": ServiceType.CONVERSATION_SERVICE,
                "database": ServiceType.USER_SERVICE,  # TODO: Add DATABASE service type
                "websocket": ServiceType.WEBSOCKET_GATEWAY,
                "webrtc": ServiceType.WEBRTC_GATEWAY,
                "webrtc_signaling": ServiceType.WEBRTC_SIGNALING,
                "rest_polling": ServiceType.REST_POLLING,
                "scenarios": ServiceType.SCENARIOS_SERVICE,
            }

            service_type = service_type_map.get(service_name.lower())
            if service_type:
                return ServiceRegistry.get_service(service_type).get_port()

        except Exception as e:
            logger.debug(f"Could not get port from PORT_MATRIX for {service_name}: {e}")

        # Use default
        if default is not None:
            return default

        # No port found
        raise KeyError(f"Port not found for service '{service_name}'")

    def get_service_url(
        self,
        service_name: str,
        protocol: str = "http",
        host: Optional[str] = None
    ) -> str:
        """
        Build service URL from port resolution

        Args:
            service_name: Service name
            protocol: Protocol (http, https, ws, wss)
            host: Host (defaults to localhost)

        Returns:
            Complete service URL

        Example:
            url = settings.get_service_url("api_gateway")
            # Returns: "http://localhost:8800"

            url = settings.get_service_url("api_gateway", protocol="https", host="api.example.com")
            # Returns: "https://api.example.com:8800"
        """
        port = self.get_service_port(service_name)
        host = host or self.get_str("SERVICE_HOST", default="localhost")
        return f"{protocol}://{host}:{port}"

    # ========================================================================
    # API Key Validation
    # ========================================================================

    def validate_required_keys(self, keys: List[str]) -> Dict[str, str]:
        """
        Validate that required API keys are present

        Args:
            keys: List of required environment variable names

        Returns:
            Dictionary of {key: value} for valid keys

        Raises:
            ValueError: If any required key is missing

        Example:
            try:
                creds = settings.validate_required_keys(["GROQ_API_KEY", "HF_TOKEN"])
                groq_key = creds["GROQ_API_KEY"]
            except ValueError as e:
                logger.error(f"Missing credentials: {e}")
                sys.exit(1)
        """
        missing_keys = []
        valid_keys = {}

        for key in keys:
            value = self.get_str(key, required=False)
            if not value or len(value.strip()) == 0:
                missing_keys.append(key)
            else:
                valid_keys[key] = value.strip()

        if missing_keys:
            error_msg = (
                f"Missing required API keys: {', '.join(missing_keys)}\n"
                "Please set these environment variables in your .env file:\n"
            )
            for key in missing_keys:
                error_msg += f"  - {key}=your-api-key-here\n"

            raise ValueError(error_msg)

        return valid_keys

    def get_api_key(
        self,
        key_name: str,
        required: bool = False,
        alternatives: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Get API key with support for alternatives

        Args:
            key_name: Primary key name
            required: If True, raise ValueError if not found
            alternatives: List of alternative key names to try

        Returns:
            API key value or None

        Raises:
            ValueError: If required=True and key not found

        Example:
            # Try HF_TOKEN, then HUGGINGFACE_TOKEN
            hf_token = settings.get_api_key(
                "HF_TOKEN",
                required=True,
                alternatives=["HUGGINGFACE_TOKEN"]
            )
        """
        # Try primary key
        value = self.get_str(key_name, required=False)
        if value:
            return value

        # Try alternatives
        if alternatives:
            for alt_key in alternatives:
                value = self.get_str(alt_key, required=False)
                if value:
                    logger.debug(f"Using alternative key {alt_key} for {key_name}")
                    return value

        # Not found
        if required:
            alt_msg = f" (tried alternatives: {', '.join(alternatives)})" if alternatives else ""
            raise ValueError(f"Required API key not found: {key_name}{alt_msg}")

        return None
