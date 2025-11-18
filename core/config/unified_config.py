#!/usr/bin/env python3
"""
Unified Configuration Manager - Single Source of Truth

Unifies settings.py and settings_service.py into one cohesive system.

Features:
- Pydantic BaseSettings for type-safe configuration
- Thread-safe singleton pattern
- Loads from .env, settings.yaml, services_config.yaml
- Type-safe getters (get_str, get_int, get_bool, get_path)
- Hot-reload support
- Backward compatible with SettingsService

Usage:
    from src.core.config import get_config

    # Get singleton instance
    config = get_config()

    # Get service-specific config (type-safe)
    from src.core.config import TTSConfig
    tts_config = config.get_service_config("external_tts", TTSConfig)

    # Get raw values with type conversion
    hf_token = config.get_str("HF_TOKEN", required=False)
    max_workers = config.get_int("MAX_WORKERS", default=6)
    debug = config.get_bool("DEBUG", default=False)
    logs_dir = config.get_path("LOGS_DIR")

    # Enable hot-reload (polls for changes every 60s)
    config.enable_hot_reload(interval=60)
"""

import os
import yaml
import threading
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List, Type, TypeVar
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from .loader import load_services_config, get_service_ports

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class WorkerSettings(BaseModel):
    """Worker configuration settings"""
    max_workers: int = Field(6, ge=1, le=32, description="Maximum number of workers")
    concurrent_per_worker: int = Field(2, ge=1, le=10)
    timeout: int = Field(30, ge=1, le=300, description="Worker timeout in seconds")
    warmup_on_init: bool = True
    queue_size: int = Field(100, ge=1)


class PipelineSettings(BaseModel):
    """Pipeline configuration settings"""
    default_language: str = Field("Portuguese", description="Default language for TTS/STT")
    default_voice: str = Field("af_bella", description="Default TTS voice")
    device: str = Field("cuda", pattern="^(cuda|cpu)$")
    debug: bool = False


class MemorySettings(BaseModel):
    """Memory and context settings"""
    max_context_messages: int = Field(10, ge=1, le=100)
    context_window_size: int = Field(4, ge=1, le=20)
    enable_long_term_memory: bool = True
    enable_embeddings_search: bool = True
    cache_dir: str = "./data/memory_cache"
    enable_persistence: bool = True
    session_ttl: int = Field(30, ge=1, description="Session TTL in minutes")


class ServerSettings(BaseModel):
    """Server configuration settings"""
    host: str = Field("0.0.0.0", description="Server bind address")
    port: int = Field(8088, ge=1024, le=65535)
    websocket_enabled: bool = True
    cors_enabled: bool = True


class SessionsSettings(BaseModel):
    """Session service configuration settings"""
    redis_url: str = Field("redis://localhost:6379", description="Redis connection URL")
    redis_db: int = Field(0, ge=0, le=15)
    session_ttl: int = Field(1800, ge=60, description="Session TTL in seconds")
    max_active_sessions: int = Field(500, ge=1)
    cleanup_interval: int = Field(300, ge=60, description="Cleanup interval in seconds")
    default_voice: str = "FEMALE_BR_1"
    default_language: str = "pt"
    max_age_hours: int = Field(24, ge=1)


class ScenariosSettings(BaseModel):
    """Scenarios service configuration settings"""
    database_path: str = "./data/scenarios.db"
    cache_ttl: int = Field(3600, ge=0)
    max_cached_scenarios: int = Field(100, ge=1)
    templates_dir: str = "./src/services/scenarios/templates"
    max_scenarios: int = Field(1000, ge=1)
    max_name_length: int = Field(100, ge=1, le=500)
    max_prompt_length: int = Field(5000, ge=100, le=50000)


class PipelineFailoverSettings(BaseModel):
    """Pipeline failover configuration settings"""
    enabled: bool = True
    primary_llm_url: str = "http://localhost:8100"
    primary_timeout: int = Field(10, ge=1, le=60)
    fallback_llm_url: str = "http://localhost:8110"
    fallback_timeout: int = Field(15, ge=1, le=60)
    failure_threshold: int = Field(3, ge=1, le=10)
    recovery_timeout: int = Field(30, ge=10, le=300)
    half_open_max_calls: int = Field(1, ge=1)
    preserve_session: bool = True
    preserve_memory: bool = True
    preserve_scenario: bool = True


class TransportSettings(BaseModel):
    """Transport services configuration settings"""
    webrtc_signaling_port: int = Field(8090, ge=1024, le=65535)
    socketio_port: int = Field(8304, ge=1024, le=65535)
    rest_polling_port: int = Field(8303, ge=1024, le=65535)
    metrics_port: int = Field(8600, ge=1024, le=65535)


class ServiceEndpointsSettings(BaseModel):
    """Service endpoints configuration"""
    endpoints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def get_base_url(self, service_name: str) -> Optional[str]:
        """Get base URL for a service"""
        if not self.endpoints or service_name not in self.endpoints:
            return None

        service_config = self.endpoints[service_name]
        host = service_config.get('host', 'localhost')
        port = service_config.get('port')

        if not port:
            return None

        return f"http://{host}:{port}"

    def get_endpoint(self, service_name: str, endpoint_name: str, **kwargs) -> Optional[str]:
        """
        Get full endpoint URL for a service

        Args:
            service_name: Service identifier (e.g., 'ultravox', 'tts')
            endpoint_name: Endpoint name (e.g., 'audio', 'health')
            **kwargs: Variables to format endpoint path (e.g., session_id='123')

        Returns:
            Full URL or None if not found
        """
        if not self.endpoints or service_name not in self.endpoints:
            return None

        service_config = self.endpoints[service_name]
        endpoints = service_config.get('endpoints', {})

        if endpoint_name not in endpoints:
            return None

        base_url = self.get_base_url(service_name)
        if not base_url:
            return None

        endpoint_path = endpoints[endpoint_name]

        # Format path variables (e.g., {session_id} → 123)
        if kwargs:
            endpoint_path = endpoint_path.format(**kwargs)

        return f"{base_url}{endpoint_path}"


class ConfigManager(BaseSettings):
    """
    Unified Configuration Manager (v5.0)

    Single source of truth for all configuration.
    Unifies settings.py + settings_service.py functionality.

    Loads from (in priority order):
    1. .env file (lowest priority)
    2. settings.yaml
    3. services_config.yaml
    4. ULTRAVOX_* environment variables (highest priority)

    Thread-safe singleton with type-safe getters.
    """

    # Nested settings from settings.py
    workers: WorkerSettings = Field(default_factory=WorkerSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    sessions: SessionsSettings = Field(default_factory=SessionsSettings)
    scenarios: ScenariosSettings = Field(default_factory=ScenariosSettings)
    pipeline_failover: PipelineFailoverSettings = Field(default_factory=PipelineFailoverSettings)
    transport: TransportSettings = Field(default_factory=TransportSettings)
    service_endpoints: ServiceEndpointsSettings = Field(default_factory=ServiceEndpointsSettings)

    # Internal state
    _cache: Dict[str, Any] = {}
    _services_config: Dict[str, Any] = {}
    _config_file: Optional[Path] = None
    _loaded: bool = False

    # Singleton state
    _instance: Optional['ConfigManager'] = None
    _lock: threading.Lock = threading.Lock()

    model_config = SettingsConfigDict(
        env_prefix='ULTRAVOX_',
        env_nested_delimiter='__',
        case_sensitive=False,
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        arbitrary_types_allowed=True,
    )

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        """Initialize (only once due to singleton)"""
        if not self._loaded:
            super().__init__(**kwargs)
            self._load_all()
            self._loaded = True

    @model_validator(mode='before')
    @classmethod
    def load_yaml_config(cls, values):
        """Load settings from YAML file before validation"""
        import re

        def expand_env_var(value) -> Any:
            """Expand ${VAR:-default} syntax"""
            if not isinstance(value, str):
                return value

            pattern = r'\$\{([^:}]+)(?::-([^}]*))?\}'

            def replace_match(match) -> Any:
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                return os.environ.get(var_name, default_value)

            return re.sub(pattern, replace_match, value)

        def expand_dict(d):
            """Recursively expand environment variables in dict"""
            if isinstance(d, dict):
                return {k: expand_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [expand_dict(item) for item in d]
            elif isinstance(d, str):
                return expand_env_var(d)
            else:
                return d

        # Find and load settings.yaml
        config_file = cls._find_config_file()
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}

                yaml_config = expand_dict(yaml_config)
                values['_config_file'] = config_file

                # Merge YAML into values
                for section, section_config in yaml_config.items():
                    if section in values:
                        if isinstance(values[section], dict) and isinstance(section_config, dict):
                            values[section].update(section_config)
                        else:
                            values[section] = section_config
                    else:
                        values[section] = section_config

                logger.info(f"✅ Loaded settings.yaml from {config_file}")
            except Exception as e:
                logger.error(f"❌ Failed to load settings.yaml: {e}")

        return values

    @staticmethod
    def _find_config_file() -> Optional[Path]:
        """Find the settings.yaml file"""
        env_path = os.environ.get('ULTRAVOX_SETTINGS_FILE')
        if env_path and Path(env_path).exists():
            return Path(env_path)

        project_root = Path(__file__).resolve().parent.parent.parent
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

    def _load_all(self) -> None:
        """Load all configuration from all sources"""
        try:
            # Load .env file
            self._load_env_file()

            # Load services_config.yaml
            self._load_services_config()

            # Build cache from environment variables
            self._load_env_cache()

            logger.info(f"✅ ConfigManager loaded {len(self._cache)} configuration values")

        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise

    def _load_env_file(self) -> None:
        """Load .env file"""
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"

        if env_file.exists():
            load_dotenv(env_file, override=False)
            logger.debug(f"📄 Loaded .env from: {env_file}")
        else:
            logger.warning(f"⚠️  .env file not found: {env_file}")

    def _load_services_config(self) -> None:
        """Load services_config.yaml"""
        try:
            self._services_config = load_services_config()
            logger.debug("📄 Loaded services_config.yaml")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load services_config.yaml: {e}")
            self._services_config = {}

    def _load_env_cache(self) -> None:
        """Build cache from environment variables"""
        for key, value in os.environ.items():
            if key.startswith("ULTRAVOX_"):
                setting_key = key.replace("ULTRAVOX_", "", 1)
                self._cache[setting_key] = value
            else:
                self._cache[key] = value

    # ========================================================================
    # Type-safe getters (from SettingsService)
    # ========================================================================

    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key (case-insensitive)
            default: Default value if not found
            required: If True, raise KeyError if not found

        Returns:
            Configuration value or default
        """
        key_upper = key.upper()

        if key_upper not in self._cache:
            if required:
                raise KeyError(f"Required configuration not found: {key}")
            return default

        return self._cache[key_upper]

    def get_str(self, key: str, default: str = "", required: bool = False) -> str:
        """Get string configuration value"""
        value = self.get(key, default, required)
        return str(value) if value is not None else default

    def get_int(self, key: str, default: int = 0, required: bool = False) -> int:
        """Get integer configuration value"""
        value = self.get(key, default, required)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️  Could not convert {key}={value} to int, using default={default}")
            return default

    def get_float(self, key: str, default: float = 0.0, required: bool = False) -> float:
        """Get float configuration value"""
        value = self.get(key, default, required)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️  Could not convert {key}={value} to float, using default={default}")
            return default

    def get_bool(self, key: str, default: bool = False, required: bool = False) -> bool:
        """Get boolean configuration value"""
        value = self.get(key, default, required)
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        return str(value).lower() in ("true", "1", "yes", "on")

    def get_list(self, key: str, default: Optional[List] = None, required: bool = False) -> List:
        """Get list configuration value (comma-separated string or YAML list)"""
        value = self.get(key, default, required)

        if value is None:
            return default or []

        if isinstance(value, list):
            return value

        return [item.strip() for item in str(value).split(",")]

    def get_dict(self, key: str, default: Optional[Dict] = None, required: bool = False) -> Dict:
        """Get dictionary configuration value"""
        value = self.get(key, default, required)

        if value is None:
            return default or {}

        if isinstance(value, dict):
            return value

        logger.warning(f"⚠️  Expected dict for {key}, got {type(value).__name__}")
        return default or {}

    def get_path(self, key: str, default: Optional[Path] = None, required: bool = False) -> Path:
        """Get Path configuration value (with variable expansion)"""
        value = self.get_str(key, required=required)

        if not value:
            return default or Path(".")

        expanded = os.path.expandvars(value)
        expanded = os.path.expanduser(expanded)

        return Path(expanded)

    # ========================================================================
    # Service-specific configuration
    # ========================================================================

    def get_service_config(self, service_id: str, config_class: Type[T]) -> T:
        """
        Get type-safe service configuration

        Args:
            service_id: Service identifier (e.g., 'external_tts')
            config_class: Pydantic model class (e.g., TTSConfig)

        Returns:
            Instantiated config model

        Example:
            from src.core.config import get_config, TTSConfig

            config = get_config()
            tts_config = config.get_service_config("external_tts", TTSConfig)
            print(tts_config.hf_token)
        """
        if not hasattr(config_class, 'from_settings'):
            raise ValueError(f"{config_class.__name__} must have a from_settings() classmethod")

        return config_class.from_settings(self)

    def get_service_ports(self) -> Dict[str, int]:
        """Get all service ports from services_config.yaml"""
        return get_service_ports()

    def get_service_info(self, service_id: str) -> Dict[str, Any]:
        """Get service information from services_config.yaml"""
        services = self._services_config.get('services', {})
        return services.get(service_id, {})

    # ========================================================================
    # Hot-reload support
    # ========================================================================

    def enable_hot_reload(self, interval: int = 60) -> None:
        """
        Enable hot-reload of configuration (polls for changes)

        Args:
            interval: Polling interval in seconds
        """
        from .hot_reload import HotReloadManager

        self._hot_reload = HotReloadManager(self, interval)
        self._hot_reload.start()
        logger.info(f"🔄 Hot-reload enabled (interval: {interval}s)")

    def disable_hot_reload(self) -> None:
        """Disable hot-reload"""
        if hasattr(self, '_hot_reload'):
            self._hot_reload.stop()
            logger.info("🔄 Hot-reload disabled")

    def reload(self) -> None:
        """Reload all configuration from sources"""
        self._cache.clear()
        self._services_config.clear()
        self._load_all()
        logger.info("🔄 Configuration reloaded")

    # ========================================================================
    # Utility methods
    # ========================================================================

    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        return key.upper() in self._cache

    def all(self) -> Dict[str, Any]:
        """Get all configuration values (read-only)"""
        return dict(self._cache)

    def print_summary(self) -> None:
        """Print a summary of current settings"""
        print("\n" + "="*60)
        print("📋 ULTRAVOX CONFIGURATION MANAGER (v5.0)")
        print("="*60)
        print(f"Workers: {self.workers.max_workers} (concurrent: {self.workers.concurrent_per_worker})")
        print(f"Device: {self.pipeline.device}")
        print(f"Language: {self.pipeline.default_language}")
        print(f"Voice: {self.pipeline.default_voice}")
        print(f"Memory: {self.memory.max_context_messages} messages")
        print(f"Server: {self.server.host}:{self.server.port}")
        print(f"Sessions: Redis={self.sessions.redis_url}, TTL={self.sessions.session_ttl}s")
        print(f"Config file: {self._config_file}")
        print(f"Services loaded: {len(self._services_config.get('services', {}))}")
        print(f"Cache entries: {len(self._cache)}")
        print("="*60)


# ============================================================================
# Singleton access
# ============================================================================

@lru_cache()
def get_config() -> ConfigManager:
    """
    Get the singleton ConfigManager instance

    Returns:
        ConfigManager instance with all configuration loaded
    """
    return ConfigManager()


def reload_config() -> ConfigManager:
    """
    Reload configuration from all sources

    Returns:
        Fresh ConfigManager instance
    """
    get_config.cache_clear()
    config = get_config()
    config.reload()
    return config
