#!/usr/bin/env python3
"""
Unified Configuration Management System for Ultravox Pipeline

Single source of truth for all configuration:
- Type-safe Pydantic models
- Centralized loading from .env, settings.yaml, services_config.yaml
- Hot-reload support
- Validation and error handling

Usage:
    # Get singleton config manager
    from src.core.config import get_config

    config = get_config()

    # Get service-specific config
    tts_config = config.get_service_config("external_tts", TTSConfig)

    # Get raw value
    hf_token = config.get_str("HF_TOKEN", required=False)

    # Enable hot-reload
    config.enable_hot_reload(interval=60)
"""

from .unified_config import (
    ConfigManager,
    get_config,
    reload_config,
)

from .models import (
    # Service Configs
    TTSConfig,
    LLMConfig,
    ExternalSTTConfig,
    OrchestratorConfig,
    DatabaseConfig,
    RunPodLLMConfig,
    StorageConfig,
    STTConfig,
    PortConfig,
    UserConfig,
    ScenariosConfig,
    WebRTCConfig,
    WebRTCSignalingConfig,
    SkyPilotConfig,
    MetricsConfig,
)

from .loader import (
    load_services_config,
    get_service_ports,
    get_service_config,
    get_profile_services,
)

__all__ = [
    # Main API
    "ConfigManager",
    "get_config",
    "reload_config",
    # Service Configs
    "TTSConfig",
    "LLMConfig",
    "ExternalSTTConfig",
    "OrchestratorConfig",
    "DatabaseConfig",
    "RunPodLLMConfig",
    "StorageConfig",
    "STTConfig",
    "PortConfig",
    "UserConfig",
    "ScenariosConfig",
    "WebRTCConfig",
    "WebRTCSignalingConfig",
    "SkyPilotConfig",
    "MetricsConfig",
    # Loader functions
    "load_services_config",
    "get_service_ports",
    "get_service_config",
    "get_profile_services",
]
