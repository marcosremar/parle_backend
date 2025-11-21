#!/usr/bin/env python3
"""
Unified Settings Manager for Ultravox Pipeline (v4.0)

⚠️  DEPRECATED as of v5.2 - Use src.core.config instead!

This module is deprecated. Please migrate to the new unified configuration system:

    # OLD (deprecated):
    from src.core.settings import get_settings
    settings = get_settings()

    # NEW (recommended):
    from src.core.config import get_config
    config = get_config()

See docs/CONFIG_MIGRATION_GUIDE.md for migration instructions.

Legacy Documentation:
---------------------
Type-safe configuration using Pydantic with support for .env, YAML, and environment variables

Priority: .env < settings.yaml < ULTRAVOX_* env vars

Migration from v3.x:
- All dataclass-based settings replaced with single UltravoxSettings
- Backward compatible: Old getter functions still work with deprecation warnings
- New usage: settings = get_settings(); print(settings.sessions.redis_url)
"""

import os
import yaml
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

from pydantic import ValidationError, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
# Import exceptions from orchestrator (shared by all services)
try:
    from src.services.orchestrator.utils.exceptions import UltravoxError, wrap_exception
except ImportError:
    # Fallback if orchestrator not available
    class UltravoxError(Exception):
        pass
    def wrap_exception(e, service_name=None, operation=None):
        return e


class WorkerSettings(BaseSettings):
    """Worker configuration settings"""
    model_config = SettingsConfigDict(extra='ignore')

    max_workers: int = Field(6, ge=1, le=32, description="Maximum number of workers")
    concurrent_per_worker: int = Field(2, ge=1, le=10)
    timeout: int = Field(30, ge=1, le=300, description="Worker timeout in seconds")
    warmup_on_init: bool = True
    queue_size: int = Field(100, ge=1)

    @field_validator('max_workers')
    @classmethod
    def validate_max_workers(cls, v) -> Any:
        if v > 16:
            warnings.warn(f"max_workers={v} is high. Consider reducing for better performance.")
        return v


class PipelineSettings(BaseSettings):
    """Pipeline configuration settings"""
    model_config = SettingsConfigDict(extra='ignore')

    default_language: str = Field("Portuguese", description="Default language for TTS/STT")
    default_voice: str = Field("af_bella", description="Default TTS voice")
    device: str = Field("cuda", pattern="^(cuda|cpu)$")
    debug: bool = False

    @field_validator('device')
    @classmethod
    def validate_device(cls, v) -> str:
        if v == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.warn("CUDA requested but not available, falling back to CPU")
                    return 'cpu'
            except ImportError:
                warnings.warn("PyTorch not installed, falling back to CPU")
                return 'cpu'
        return v


class MemorySettings(BaseSettings):
    """Memory and context settings"""
    model_config = SettingsConfigDict(extra='ignore')

    max_context_messages: int = Field(10, ge=1, le=100)
    context_window_size: int = Field(4, ge=1, le=20)
    enable_long_term_memory: bool = True
    enable_embeddings_search: bool = True
    cache_dir: str = "./data/memory_cache"
    enable_persistence: bool = True
    session_ttl: int = Field(30, ge=1, description="Session TTL in minutes")


class ServerSettings(BaseSettings):
    """Server configuration settings"""
    model_config = SettingsConfigDict(extra='ignore')

    host: str = Field("0.0.0.0", description="Server bind address")
    port: int = Field(8088, ge=1024, le=65535)
    websocket_enabled: bool = True
    cors_enabled: bool = True


class SessionsSettings(BaseSettings):
    """Session service configuration settings"""
    model_config = SettingsConfigDict(extra='ignore')

    redis_url: str = Field("redis://localhost:6379", description="Redis connection URL")
    redis_db: int = Field(0, ge=0, le=15)
    session_ttl: int = Field(1800, ge=60, description="Session TTL in seconds")
    max_active_sessions: int = Field(500, ge=1)
    cleanup_interval: int = Field(300, ge=60, description="Cleanup interval in seconds")
    default_voice: str = "FEMALE_BR_1"
    default_language: str = "pt"
    max_age_hours: int = Field(24, ge=1)

    @field_validator('session_ttl')
    @classmethod
    def validate_session_ttl(cls, v) -> Any:
        if v < 300:
            warnings.warn(f"session_ttl={v}s is very short. Sessions may expire too quickly.")
        return v


class ScenariosSettings(BaseSettings):
    """Scenarios service configuration settings"""
    model_config = SettingsConfigDict(extra='ignore')

    database_path: str = "./data/scenarios.db"
    cache_ttl: int = Field(3600, ge=0)
    max_cached_scenarios: int = Field(100, ge=1)
    templates_dir: str = "./src/services/scenarios/templates"
    max_scenarios: int = Field(1000, ge=1)
    max_name_length: int = Field(100, ge=1, le=500)
    max_prompt_length: int = Field(5000, ge=100, le=50000)


class PipelineFailoverSettings(BaseSettings):
    """Pipeline failover configuration settings"""
    model_config = SettingsConfigDict(extra='ignore')

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


class TransportSettings(BaseSettings):
    """Transport services configuration settings"""
    model_config = SettingsConfigDict(extra='ignore')

    webrtc_signaling_port: int = Field(8090, ge=1024, le=65535)
    socketio_port: int = Field(8304, ge=1024, le=65535)
    rest_polling_port: int = Field(8303, ge=1024, le=65535)
    metrics_port: int = Field(8600, ge=1024, le=65535)

    @field_validator('socketio_port', 'rest_polling_port', 'metrics_port')
    @classmethod
    def validate_unique_ports(cls, v, info) -> Any:
        """Ensure all transport ports are unique"""
        if info.data:
            used_ports = {info.data.get('webrtc_signaling_port')}
            if v in used_ports:
                raise ValueError(f"Port {v} is already used by another transport service")
        return v


class ServiceEndpointsSettings(BaseSettings):
    """Service endpoints configuration"""
    model_config = SettingsConfigDict(extra='ignore')

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


class UltravoxSettings(BaseSettings):
    """
    Unified Settings for Ultravox Pipeline (v4.0)

    Single source of truth for all configuration.
    Loads from (in priority order):
    1. .env file (lowest priority)
    2. settings.yaml
    3. ULTRAVOX_* environment variables (highest priority)

    Usage:
        from src.core.settings import get_settings

        settings = get_settings()
        print(settings.sessions.redis_url)
        print(settings.pipeline.device)
    """

    # Nested settings
    workers: WorkerSettings = Field(default_factory=WorkerSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    sessions: SessionsSettings = Field(default_factory=SessionsSettings)
    scenarios: ScenariosSettings = Field(default_factory=ScenariosSettings)
    pipeline_failover: PipelineFailoverSettings = Field(default_factory=PipelineFailoverSettings)
    transport: TransportSettings = Field(default_factory=TransportSettings)
    service_endpoints: ServiceEndpointsSettings = Field(default_factory=ServiceEndpointsSettings)

    # Raw config for custom access
    _raw_config: Dict[str, Any] = {}
    _config_file: Optional[Path] = None

    model_config = SettingsConfigDict(
        env_prefix='ULTRAVOX_',
        env_nested_delimiter='__',
        case_sensitive=False,
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',  # Ignore unknown fields for forward compatibility
        arbitrary_types_allowed=True  # Allow arbitrary types (Path, etc.)
    )

    @model_validator(mode='before')
    @classmethod
    def load_yaml_config(cls, values) -> Any:
        """Load settings from YAML file before validation"""
        import re

        def expand_env_var(value) -> Any:
            """Expand ${VAR:-default} syntax"""
            if not isinstance(value, str):
                return value

            # Pattern for ${VAR:-default}
            pattern = r'\$\{([^:}]+)(?::-([^}]*))?\}'

            def replace_match(match) -> Any:
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                return os.environ.get(var_name, default_value)

            return re.sub(pattern, replace_match, value)

        def expand_dict(d) -> Any:
            """Recursively expand environment variables in dict"""
            if isinstance(d, dict):
                return {k: expand_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [expand_dict(item) for item in d]
            elif isinstance(d, str):
                return expand_env_var(d)
            else:
                return d

        # Find config file
        config_file = cls._find_config_file()

        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}

                # Expand environment variables
                yaml_config = expand_dict(yaml_config)

                # Store raw config
                values['_raw_config'] = yaml_config
                values['_config_file'] = config_file

                # Merge YAML into values (env vars will override later)
                for section, section_config in yaml_config.items():
                    if section in values:
                        # Merge with existing values
                        if isinstance(values[section], dict) and isinstance(section_config, dict):
                            values[section].update(section_config)
                        else:
                            values[section] = section_config
                    else:
                        values[section] = section_config

                from loguru import logger
                logger.info(f"✅ Loaded settings from {config_file}")
            except Exception as e:
                from loguru import logger
                logger.error(f"❌ Failed to load settings from {config_file}: {e}")

        return values

    @staticmethod
    def _find_config_file() -> Optional[Path]:
        """Find the settings file in various locations"""
        # Check environment variable
        env_path = os.environ.get('ULTRAVOX_SETTINGS_FILE')
        if env_path and Path(env_path).exists():
            return Path(env_path)

        # Check common locations
        project_root = Path(__file__).resolve().parent.parent
        search_paths = [
            Path('./settings.yaml'),
            Path('./config/settings.yaml'),
            project_root / 'settings.yaml',
            Path.home() / '.ultravox' / 'settings.yaml'
        ]

        for path in search_paths:
            if path.exists():
                return path

        return Path('./settings.yaml')  # Default (may not exist)

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a setting value by path (backward compatibility)
        Example: settings.get('workers.max_workers', 6)
        """
        keys = path.split('.')
        value = self._raw_config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def print_summary(self) -> Any:
        """Print a summary of current settings"""
        print("\n" + "="*60)
        print("📋 ULTRAVOX PIPELINE SETTINGS (v4.0)")
        print("="*60)
        print(f"Workers: {self.workers.max_workers} (concurrent: {self.workers.concurrent_per_worker})")
        print(f"Device: {self.pipeline.device}")
        print(f"Language: {self.pipeline.default_language}")
        print(f"Voice: {self.pipeline.default_voice}")
        print(f"Memory: {self.memory.max_context_messages} messages")
        print(f"Long-term memory: {self.memory.enable_long_term_memory}")
        print(f"Embeddings search: {self.memory.enable_embeddings_search}")
        print(f"Server: {self.server.host}:{self.server.port}")
        print(f"Sessions: Redis={self.sessions.redis_url}, TTL={self.sessions.session_ttl}s")
        print(f"Config file: {self._config_file}")
        print("="*60)


# Singleton instance
@lru_cache()
def get_settings() -> UltravoxSettings:
    """
    Get the singleton settings instance

    Returns:
        UltravoxSettings instance with all configuration loaded
    """
    return UltravoxSettings()


# ==============================================================================
# BACKWARD COMPATIBILITY - DEPRECATED FUNCTIONS
# These functions are kept for backward compatibility with v3.x code
# They will be removed in v5.0
# ==============================================================================

def _deprecation_warning(func_name: str, new_usage: str):
    """Show deprecation warning"""
    warnings.warn(
        f"{func_name} is deprecated and will be removed in v5.0. "
        f"Use: {new_usage}",
        DeprecationWarning,
        stacklevel=3
    )


def reload_settings() -> get_settings:
    """Reload settings from file and environment (DEPRECATED)"""
    _deprecation_warning(
        "reload_settings()",
        "get_settings.cache_clear(); settings = get_settings()"
    )
    get_settings.cache_clear()
    return get_settings()


def get_worker_settings() -> WorkerSettings:
    """Get worker settings (DEPRECATED)"""
    _deprecation_warning(
        "get_worker_settings()",
        "get_settings().workers"
    )
    return get_settings().workers


def get_pipeline_settings() -> PipelineSettings:
    """Get pipeline settings (DEPRECATED)"""
    _deprecation_warning(
        "get_pipeline_settings()",
        "get_settings().pipeline"
    )
    return get_settings().pipeline


def get_memory_settings() -> MemorySettings:
    """Get memory settings (DEPRECATED)"""
    _deprecation_warning(
        "get_memory_settings()",
        "get_settings().memory"
    )
    return get_settings().memory


def get_server_settings() -> ServerSettings:
    """Get server settings (DEPRECATED)"""
    _deprecation_warning(
        "get_server_settings()",
        "get_settings().server"
    )
    return get_settings().server


def get_sessions_settings() -> SessionsSettings:
    """Get sessions service settings (DEPRECATED)"""
    _deprecation_warning(
        "get_sessions_settings()",
        "get_settings().sessions"
    )
    return get_settings().sessions


def get_scenarios_settings() -> ScenariosSettings:
    """Get scenarios service settings (DEPRECATED)"""
    _deprecation_warning(
        "get_scenarios_settings()",
        "get_settings().scenarios"
    )
    return get_settings().scenarios


def get_pipeline_failover_settings() -> PipelineFailoverSettings:
    """Get pipeline failover settings (DEPRECATED)"""
    _deprecation_warning(
        "get_pipeline_failover_settings()",
        "get_settings().pipeline_failover"
    )
    return get_settings().pipeline_failover


def get_transport_settings() -> TransportSettings:
    """Get transport services settings (DEPRECATED)"""
    _deprecation_warning(
        "get_transport_settings()",
        "get_settings().transport"
    )
    return get_settings().transport


def get_service_endpoints_settings() -> ServiceEndpointsSettings:
    """Get service endpoints settings (DEPRECATED)"""
    _deprecation_warning(
        "get_service_endpoints_settings()",
        "get_settings().service_endpoints"
    )
    return get_settings().service_endpoints


def get_config(path: str, default: Any = None) -> Any:
    """Get a configuration value (DEPRECATED)"""
    _deprecation_warning(
        "get_config()",
        "get_settings().get()"
    )
    return get_settings().get(path, default)


# For backward compatibility with old Settings class
Settings = UltravoxSettings
