#!/usr/bin/env python3
"""
Centralized Configuration Models for all services

Uses Pydantic for type-safe, validated configuration.
All services read configuration through SettingsService and instantiate
their respective config models.

Usage:
    # In a service:
    class MyService(BaseService):
        async def initialize(self) -> bool:
            config = TTSConfig.from_settings(self.settings)
            api_key = config.hf_token  # Type-safe, validated
"""

from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
import logging

logger = logging.getLogger(__name__)


class TTSConfig(BaseModel):
    """Configuration for TTS services (external_tts, tts, etc.)"""

    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token"
    )
    hf_api_key: Optional[str] = Field(
        default=None,
        description="Alternative HuggingFace API key"
    )
    huggingface_api_key: Optional[str] = Field(
        default=None,
        description="Another alternative HuggingFace API key name"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache/ultravox-pipeline/logs",
        description="Directory for service logs"
    )
    default_voice: str = Field(
        default="af_heart",
        description="Default TTS voice ID"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def effective_token(self) -> Optional[str]:
        """Get the effective token with fallback priority"""
        return self.hf_token or self.hf_api_key or self.huggingface_api_key

    @classmethod
    def from_settings(cls, settings) -> "TTSConfig":
        """Create TTSConfig from SettingsService"""
        return cls(
            hf_token=settings.get_str("HF_TOKEN", required=False),
            hf_api_key=settings.get_str("HF_API_KEY", required=False),
            huggingface_api_key=settings.get_str("HUGGINGFACE_API_KEY", required=False),
            logs_dir=settings.get_path("LOGS_DIR"),
            default_voice=settings.get_str("EXTERNAL_TTS_DEFAULT_VOICE", default="af_heart"),
            debug=settings.get_bool("DEBUG", default=False),
        )


class OrchestratorConfig(BaseModel):
    """Configuration for Orchestrator service"""

    startup_mode: bool = Field(
        default=False,
        description="Skip health checks on startup"
    )
    in_process_mode: bool = Field(
        default=False,
        description="Run services in-process vs HTTP"
    )
    skip_health_checks: bool = Field(
        default=False,
        description="Skip all health checks"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "OrchestratorConfig":
        """Create OrchestratorConfig from SettingsService"""
        return cls(
            startup_mode=settings.get_bool("STARTUP_MODE", default=False),
            in_process_mode=settings.get_bool("ORCHESTRATOR_IN_PROCESS", default=False),
            skip_health_checks=settings.get_bool("ORCHESTRATOR_SKIP_HEALTH_CHECKS", default=False),
            debug=settings.get_bool("DEBUG", default=False),
        )


class RunPodLLMConfig(BaseModel):
    """Configuration for RunPod LLM service"""

    api_key: Optional[str] = Field(
        default=None,
        description="RunPod API key"
    )
    endpoint_id: Optional[str] = Field(
        default=None,
        description="RunPod endpoint ID"
    )
    qwen_endpoint_id: Optional[str] = Field(
        default=None,
        description="RunPod Qwen endpoint ID"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @field_validator('api_key')
    def validate_api_key(cls, v):
        """Warn if API key is not set"""
        if not v:
            logger.warning("⚠️  RunPod API key not configured")
        return v

    @classmethod
    def from_settings(cls, settings) -> "RunPodLLMConfig":
        """Create RunPodLLMConfig from SettingsService"""
        return cls(
            api_key=settings.get_str("RUNPOD_API_KEY", required=False),
            endpoint_id=settings.get_str("RUNPOD_ENDPOINT_ID", required=False),
            qwen_endpoint_id=settings.get_str("RUNPOD_QWEN_ENDPOINT_ID", required=False),
            debug=settings.get_bool("DEBUG", default=False),
        )


class DatabaseConfig(BaseModel):
    """Configuration for Database service (Redis)"""

    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database index (0-15)"
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password (if required)"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @field_validator('redis_url')
    def validate_redis_url(cls, v):
        """Validate Redis URL format"""
        if not v.startswith(('redis://', 'rediss://')):
            raise ValueError("Redis URL must start with 'redis://' or 'rediss://'")
        return v

    @classmethod
    def from_settings(cls, settings) -> "DatabaseConfig":
        """Create DatabaseConfig from SettingsService"""
        return cls(
            redis_url=settings.get_str("REDIS_URL", default="redis://localhost:6379"),
            redis_db=settings.get_int("REDIS_DB", default=0),
            redis_password=settings.get_str("REDIS_PASSWORD", required=False),
            debug=settings.get_bool("DEBUG", default=False),
        )


class LLMConfig(BaseModel):
    """Configuration for LLM services (llm, external_llm, etc.)"""

    default_model: str = Field(
        default="groq/llama-3.1-8b-instant",
        description="Default LLM model to use"
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=32000,
        description="Maximum tokens for LLM responses"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature (creativity)"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "LLMConfig":
        """Create LLMConfig from SettingsService"""
        return cls(
            default_model=settings.get_str("DEFAULT_LLM_MODEL", default="groq/llama-3.1-8b-instant"),
            groq_api_key=settings.get_str("GROQ_API_KEY", required=False),
            openai_api_key=settings.get_str("OPENAI_API_KEY", required=False),
            max_tokens=settings.get_int("LLM_MAX_TOKENS", default=1024),
            temperature=settings.get_float("LLM_TEMPERATURE", default=0.7),
            debug=settings.get_bool("DEBUG", default=False),
        )


class StorageConfig(BaseModel):
    """Configuration for Storage services"""

    base_path: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ultravox-pipeline" / "storage",
        description="Base storage directory"
    )
    files_path: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ultravox-pipeline" / "storage" / "files",
        description="Files storage directory"
    )
    audio_path: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ultravox-pipeline" / "storage" / "audio",
        description="Audio storage directory"
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        description="Maximum file size in MB"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_settings(cls, settings) -> "StorageConfig":
        """Create StorageConfig from SettingsService"""
        # Use portable cache directory as default
        default_storage = Path.home() / ".cache" / "ultravox-pipeline" / "storage"
        return cls(
            base_path=settings.get_path("STORAGE_BASE_PATH", default=default_storage),
            files_path=settings.get_path("STORAGE_FILES_PATH", default=default_storage / "files"),
            audio_path=settings.get_path("STORAGE_AUDIO_PATH", default=default_storage / "audio"),
            max_file_size_mb=settings.get_int("STORAGE_MAX_FILE_SIZE_MB", default=100),
            debug=settings.get_bool("DEBUG", default=False),
        )


class STTConfig(BaseModel):
    """Configuration for STT (Speech-to-Text) services"""

    provider: str = Field(
        default="openai",
        description="STT provider (openai, google, etc.)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for STT provider"
    )
    language: str = Field(
        default="en-US",
        description="Default language for STT"
    )
    model: str = Field(
        default="whisper-1",
        description="STT model name"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "STTConfig":
        """Create STTConfig from SettingsService"""
        return cls(
            provider=settings.get_str("STT_PROVIDER", default="openai"),
            api_key=settings.get_str("STT_API_KEY", required=False),
            language=settings.get_str("STT_LANGUAGE", default="en-US"),
            model=settings.get_str("STT_MODEL", default="whisper-1"),
            debug=settings.get_bool("DEBUG", default=False),
        )


class PortConfig(BaseModel):
    """Configuration for service ports"""

    websocket_port: int = Field(default=8022, description="WebSocket port")
    rest_polling_port: int = Field(default=8106, description="REST polling port")
    webrtc_port: int = Field(default=8020, description="WebRTC port")
    webrtc_signaling_port: int = Field(default=8021, description="WebRTC signaling port")
    api_gateway_port: int = Field(default=8010, description="API Gateway port")
    conversation_store_port: int = Field(default=8200, description="Conversation Store port")
    metrics_port: int = Field(default=9090, description="Metrics port")

    model_config = ConfigDict()

    @field_validator('websocket_port', 'rest_polling_port', 'webrtc_port',
               'webrtc_signaling_port', 'api_gateway_port', 'conversation_store_port', 'metrics_port')
    def validate_port(cls, v):
        """Validate port number"""
        if not (1024 <= v <= 65535):
            raise ValueError(f"Port must be between 1024 and 65535, got {v}")
        return v

    @classmethod
    def from_settings(cls, settings) -> "PortConfig":
        """Create PortConfig from SettingsService"""
        return cls(
            websocket_port=settings.get_int("WEBSOCKET_PORT", default=8022),
            rest_polling_port=settings.get_int("REST_POLLING_PORT", default=8106),
            webrtc_port=settings.get_int("WEBRTC_PORT", default=8020),
            webrtc_signaling_port=settings.get_int("WEBRTC_SIGNALING_PORT", default=8021),
            api_gateway_port=settings.get_int("API_GATEWAY_PORT", default=8010),
            conversation_store_port=settings.get_int("CONVERSATION_STORE_PORT", default=8200),
            metrics_port=settings.get_int("METRICS_PORT", default=9090),
        )


class ExternalSTTConfig(BaseModel):
    """Configuration for External STT service"""

    provider: str = Field(
        default="groq",
        description="STT provider (groq, openai)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for STT provider"
    )
    language: str = Field(
        default="pt",
        description="Default language code"
    )
    response_format: str = Field(
        default="text",
        description="Response format (text/json/verbose_json)"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "ExternalSTTConfig":
        """Create ExternalSTTConfig from SettingsService"""
        return cls(
            provider=settings.get_str("EXTERNAL_STT_PROVIDER", default="groq"),
            api_key=settings.get_str("EXTERNAL_STT_API_KEY", required=False),
            language=settings.get_str("EXTERNAL_STT_LANGUAGE", default="pt"),
            response_format=settings.get_str("EXTERNAL_STT_RESPONSE_FORMAT", default="text"),
            debug=settings.get_bool("DEBUG", default=False),
        )


class UserConfig(BaseModel):
    """Configuration for User service"""

    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "UserConfig":
        """Create UserConfig from SettingsService"""
        return cls(
            debug=settings.get_bool("DEBUG", default=False),
        )


class ScenariosConfig(BaseModel):
    """Configuration for Scenarios service"""

    cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "ScenariosConfig":
        """Create ScenariosConfig from SettingsService"""
        return cls(
            cache_ttl=settings.get_int("SCENARIOS_CACHE_TTL", default=300),
            debug=settings.get_bool("DEBUG", default=False),
        )


class WebRTCConfig(BaseModel):
    """Configuration for WebRTC service"""

    stun_servers: List[str] = Field(
        default_factory=lambda: ["stun:stun.l.google.com:19302"],
        description="STUN servers for NAT traversal"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "WebRTCConfig":
        """Create WebRTCConfig from SettingsService"""
        stun_str = settings.get_str("WEBRTC_STUN_SERVERS", default="stun:stun.l.google.com:19302")
        stun_servers = [s.strip() for s in stun_str.split(",") if s.strip()]

        return cls(
            stun_servers=stun_servers or ["stun:stun.l.google.com:19302"],
            debug=settings.get_bool("DEBUG", default=False),
        )


class WebRTCSignalingConfig(BaseModel):
    """Configuration for WebRTC Signaling service"""

    ice_servers: List[str] = Field(
        default_factory=lambda: ["stun:stun.l.google.com:19302"],
        description="ICE servers for connection establishment"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "WebRTCSignalingConfig":
        """Create WebRTCSignalingConfig from SettingsService"""
        ice_str = settings.get_str("WEBRTC_ICE_SERVERS", default="stun:stun.l.google.com:19302")
        ice_servers = [s.strip() for s in ice_str.split(",") if s.strip()]

        return cls(
            ice_servers=ice_servers or ["stun:stun.l.google.com:19302"],
            debug=settings.get_bool("DEBUG", default=False),
        )


class SkyPilotConfig(BaseModel):
    """Configuration for SkyPilot service"""

    enabled_providers: List[str] = Field(
        default_factory=lambda: [],
        description="Enabled cloud providers (vastai, runpod, lambda, gcp)"
    )
    gcs_cache_enabled: bool = Field(
        default=False,
        description="Enable GCS caching"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "SkyPilotConfig":
        """Create SkyPilotConfig from SettingsService"""
        providers_str = settings.get_str("SKYPILOT_ENABLED_PROVIDERS", default="")
        providers = [p.strip() for p in providers_str.split(",") if p.strip()]

        return cls(
            enabled_providers=providers,
            gcs_cache_enabled=settings.get_bool("SKYPILOT_GCS_CACHE_ENABLED", default=False),
            debug=settings.get_bool("DEBUG", default=False),
        )


class MetricsConfig(BaseModel):
    """Configuration for Metrics Testing service"""

    language: str = Field(
        default="pt-BR",
        description="Default language for metrics testing"
    )
    kokoro_voice: str = Field(
        default="Dora",
        description="Default Kokoro TTS voice"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    model_config = ConfigDict()

    @classmethod
    def from_settings(cls, settings) -> "MetricsConfig":
        """Create MetricsConfig from SettingsService"""
        return cls(
            language=settings.get_str("METRICS_LANGUAGE", default="pt-BR"),
            kokoro_voice=settings.get_str("METRICS_KOKORO_VOICE", default="Dora"),
            debug=settings.get_bool("DEBUG", default=False),
        )
