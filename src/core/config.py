"""
Unified Configuration System for Monolith Modular Architecture
Centralizes all configuration loading from .env and settings.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")
    
    url: str = Field(default="sqlite:///./data/parle.db", description="Database URL")
    pool_size: int = Field(default=10, ge=1, le=50)
    max_overflow: int = Field(default=20, ge=0)
    echo: bool = Field(default=False, description="SQLAlchemy echo mode")


class RedisConfig(BaseSettings):
    """Redis configuration"""
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")
    
    url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = None


class LLMConfig(BaseSettings):
    """LLM service configuration"""
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")
    
    provider: str = Field(default="groq", description="LLM provider (groq, openai, litellm)")
    model: str = Field(default="llama-3.1-8b-instant", description="Model name")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: float = Field(default=30.0, ge=1.0)


class STTConfig(BaseSettings):
    """STT service configuration"""
    model_config = SettingsConfigDict(env_prefix="STT_", extra="ignore")
    
    provider: str = Field(default="groq", description="STT provider (groq, whisper)")
    model: str = Field(default="whisper-large-v3", description="Model name")
    api_key: Optional[str] = None
    language: str = Field(default="pt", description="Default language")
    timeout: float = Field(default=15.0, ge=1.0)


class TTSConfig(BaseSettings):
    """TTS service configuration"""
    model_config = SettingsConfigDict(env_prefix="TTS_", extra="ignore")
    
    provider: str = Field(default="elevenlabs", description="TTS provider (elevenlabs, huggingface)")
    voice_id: str = Field(default="Rachel", description="Default voice ID")
    api_key: Optional[str] = None
    language: str = Field(default="pt", description="Default language")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    timeout: float = Field(default=20.0, ge=1.0)


class AuthConfig(BaseSettings):
    """Authentication configuration"""
    model_config = SettingsConfigDict(env_prefix="AUTH_", extra="ignore")
    
    jwt_secret_key: str = Field(default="", description="JWT secret key (REQUIRED in production)")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=1, ge=1, le=24, description="JWT expiration in hours (max 24h)")
    jwt_refresh_expiration_hours: int = Field(default=168, ge=24, le=720, description="Refresh token expiration in hours")
    password_min_length: int = Field(default=8, ge=6, le=128)
    
    @model_validator(mode='after')
    def validate_jwt_secret(self):
        """Validate JWT secret key in production"""
        # Only validate in production, allow empty/default in development
        # This will be checked at AppConfig level
        return self


class ServerConfig(BaseSettings):
    """Server configuration"""
    model_config = SettingsConfigDict(env_prefix="SERVER_", extra="ignore")
    
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of workers")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    log_level: str = Field(default="INFO", description="Log level")
    allowed_origins: List[str] = Field(default_factory=lambda: [], description="Allowed CORS origins (empty = all in dev)")
    max_upload_size_mb: int = Field(default=10, ge=1, le=100, description="Maximum upload size in MB")


class MonolithConfig(BaseSettings):
    """Monolith-specific configuration"""
    model_config = SettingsConfigDict(env_prefix="MONOLITH_", extra="ignore")
    
    mode: str = Field(default="monolith", description="Execution mode (monolith, microservices)")
    use_direct_calls: bool = Field(default=True, description="Use direct Python calls instead of HTTP")
    shared_database: bool = Field(default=True, description="Use shared database connection pool")
    shared_redis: bool = Field(default=True, description="Use shared Redis connection")


class AppConfig(BaseSettings):
    """Main application configuration"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment (development, production)")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Project info
    project_name: str = Field(default="Parle Backend", description="Project name")
    version: str = Field(default="1.0.0", description="Version")
    
    # Sub-configs
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    monolith: MonolithConfig = Field(default_factory=MonolithConfig)
    
    # Service URLs (for backward compatibility, will be removed in monolith mode)
    stt_service_url: Optional[str] = Field(default=None, description="STT service URL (legacy)")
    tts_service_url: Optional[str] = Field(default=None, description="TTS service URL (legacy)")
    llm_service_url: Optional[str] = Field(default=None, description="LLM service URL (legacy)")
    orchestrator_service_url: Optional[str] = Field(default=None, description="Orchestrator service URL (legacy)")
    
    @model_validator(mode='before')
    @classmethod
    def load_yaml_config(cls, values: Any) -> Any:
        """
        Load configuration from settings.yaml file.
        
        Priority order (highest to lowest):
        1. Environment variables (handled by Pydantic)
        2. settings.yaml values
        3. Field defaults
        
        Args:
            values: Initial values dict from Pydantic
            
        Returns:
            Updated values dict with YAML config merged
        """
        if not isinstance(values, dict):
            return values
        
        # Load YAML config if file exists
        if CONFIG_YAML_PATH.exists():
            try:
                with open(CONFIG_YAML_PATH, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                
                logger.debug(f"📄 Loading configuration from {CONFIG_YAML_PATH}")
                
                # Helper function to merge nested dicts
                def merge_dict(base: Dict, update: Dict) -> Dict:
                    """Recursively merge update dict into base dict"""
                    result = base.copy()
                    for key, value in update.items():
                        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                            result[key] = merge_dict(result[key], value)
                        else:
                            result[key] = value
                    return result
                
                # Merge YAML config into values
                # YAML values are lower priority than env vars (which are already in values)
                values = merge_dict(yaml_config, values)
                
                logger.info(f"✅ Configuration loaded from {CONFIG_YAML_PATH}")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load YAML config from {CONFIG_YAML_PATH}: {e}")
                logger.debug("   Continuing with environment variables and defaults only")
        
        return values
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # System always uses direct module calls (monolithic modular architecture)
        self.monolith.mode = "monolith"
        self.monolith.use_direct_calls = True
    
    @model_validator(mode='after')
    def validate_production_config(self):
        """Validate production configuration requirements"""
        if self.environment == "production":
            # Validate JWT secret key
            if not self.auth.jwt_secret_key or self.auth.jwt_secret_key == "your-secret-key-change-in-production":
                raise ValueError(
                    "AUTH_JWT_SECRET_KEY must be set in production. "
                    "Please set a strong secret key in your environment variables."
                )
            
            # Validate CORS origins
            if not self.server.allowed_origins:
                logger.warning(
                    "⚠️  SERVER_ALLOWED_ORIGINS not set in production. "
                    "This allows requests from any origin. Consider setting allowed origins."
                )
            
            # Validate API keys for production services
            if not self.llm.api_key:
                logger.warning("⚠️  LLM_API_KEY not set - LLM service may not work")
            if not self.stt.api_key:
                logger.warning("⚠️  STT_API_KEY not set - STT service may not work")
            if not self.tts.api_key:
                logger.warning("⚠️  TTS_API_KEY not set - TTS service may not work")
            
            # Validate server configuration
            if self.server.max_upload_size_mb > 100:
                logger.warning(f"⚠️  SERVER_MAX_UPLOAD_SIZE_MB ({self.server.max_upload_size_mb}MB) is very large. Consider reducing for security.")
            
            # Validate database URL format
            if self.database.url and not self.database.url.startswith(("sqlite:///", "postgresql://", "mysql://")):
                logger.warning(f"⚠️  DB_URL format may be invalid: {self.database.url}")
        
        return self


@lru_cache()
def get_config() -> AppConfig:
    """
    Get application configuration (singleton)
    
    Returns:
        AppConfig instance
    """
    try:
        config = AppConfig()
        logger.info("✅ Configuration loaded successfully")
        logger.debug(f"   Environment: {config.environment}")
        logger.debug(f"   Monolith mode: {config.monolith.mode}")
        logger.debug(f"   Direct calls: {config.monolith.use_direct_calls}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        # Return default config on error
        return AppConfig()


# Backward compatibility alias
def get_settings() -> AppConfig:
    """Backward compatibility alias for get_config()"""
    return get_config()
