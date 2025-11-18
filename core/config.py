"""
Configuration classes for the unified pipeline

⚠️  DEPRECATED as of v5.2 - Use src.core.config instead!

This module is deprecated. Please migrate to the new unified configuration system:

    # OLD (deprecated):
    from src.core.config import UnifiedConfig
    config = UnifiedConfig.from_env()

    # NEW (recommended):
    from src.core.config import get_config
    config = get_config()

See docs/CONFIG_MIGRATION_GUIDE.md for migration instructions.

Legacy Documentation:
---------------------
Uses dataclasses for clean configuration management
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
from pathlib import Path

# Load environment configuration


@dataclass
class UltravoxConfig:
    """Configuration for Ultravox audio processor"""
    
    # Model settings
    model_path: str = "fixie-ai/ultravox-v0_6-llama-3_1-8b"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    
    # vLLM settings
    gpu_memory_utilization: float = 0.10  # 10% of GPU memory
    max_model_len: int = 4096
    enforce_eager: bool = True
    enable_prefix_caching: bool = False
    trust_remote_code: bool = True
    
    # Generation settings
    temperature: float = 0.3
    max_tokens: int = 50
    repetition_penalty: float = 1.1
    stop_tokens: List[str] = field(default_factory=lambda: [".", "!", "?", "\n\n"])
    
    # Audio settings
    required_sample_rate: int = 16000  # Ultravox requires 16kHz
    
    # Prompt settings
    default_prompt: str = "Você é um assistente útil. <|audio|>\nResponda em português de forma natural:"
    
    # Memory settings
    use_memory: bool = True
    max_context_messages: int = 10
    
    # HuggingFace token (optional)
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv('HF_TOKEN'))


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech"""
    
    # Engine selection
    engine: str = "gtts"  # Options: "gtts", "piper", "kokoro"
    
    # Audio settings
    sample_rate: int = 24000  # Output sample rate
    channels: int = 1  # Mono
    
    # Language settings
    language: str = "pt"
    region: str = "BR"
    
    # Voice settings
    voice_id: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    
    # gTTS specific
    gtts_slow: bool = False
    gtts_lang: str = "pt"
    
    # Piper specific (if using)
    piper_model_path: Optional[str] = None
    piper_config_path: Optional[str] = None
    
    # Cache settings
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("TTS_CACHE_DIR")))
    max_cache_size_mb: int = 100


@dataclass
class MemoryConfig:
    """Configuration for memory storage"""
    
    # Storage backend
    backend: str = "inmemory"  # Options: "inmemory", "redis", "sqlite"
    
    # Session management
    max_sessions: int = 100
    max_messages_per_session: int = 100
    session_timeout_seconds: int = 1800  # 30 minutes
    
    # Redis settings (if using Redis)
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv('REDIS_URL', 'redis://localhost:6379'))
    redis_db: int = 0
    redis_key_prefix: str = "ultravox:"
    
    # SQLite settings (if using SQLite)
    sqlite_path: Path = field(default_factory=lambda: Path(os.getenv("ULTRAVOX_MEMORY_DB_PATH")))
    
    # Memory pruning
    auto_cleanup: bool = True
    cleanup_interval_seconds: int = 300  # 5 minutes


@dataclass
class WebRTCConfig:
    """Configuration for WebRTC handling"""
    
    # Connection settings
    stun_servers: List[str] = field(default_factory=lambda: [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302"
    ])
    
    turn_servers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Audio settings
    audio_codec: str = "opus"  # Options: "opus", "pcm"
    audio_channels: int = 1
    audio_sample_rate: int = 16000  # Input sample rate
    
    # Connection management
    max_connections: int = 50
    connection_timeout_seconds: int = 60
    ice_gathering_timeout_seconds: int = 10
    
    # Data channel settings
    data_channel_name: str = "audio"
    ordered: bool = True
    max_retransmits: Optional[int] = None
    
    # Stats collection
    collect_stats: bool = True
    stats_interval_seconds: int = 5


@dataclass
class ServerConfig:
    """Main server configuration"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    
    # SSL/TLS (optional)
    ssl_cert: Optional[Path] = None
    ssl_key: Optional[Path] = None
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Static files
    serve_static: bool = True
    static_dir: Path = field(default_factory=lambda: Path("./web-interface"))
    
    # API settings
    api_prefix: str = "/api"
    health_check_path: str = "/health"
    metrics_path: str = "/metrics"
    
    # Performance settings
    max_workers: int = 10
    request_timeout_seconds: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[Path] = None
    
    # Monitoring
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    # Rate limiting
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_period_seconds: int = 60


@dataclass
class UnifiedConfig:
    """Complete configuration for the unified pipeline"""
    
    ultravox: UltravoxConfig = field(default_factory=UltravoxConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    webrtc: WebRTCConfig = field(default_factory=WebRTCConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    @classmethod
    def from_env(cls) -> 'UnifiedConfig':
        """
        Create configuration from environment variables
        Override defaults with env vars if present
        """
        config = cls()
        
        # Override with environment variables
        if os.getenv('ULTRAVOX_MODEL_PATH'):
            config.ultravox.model_path = os.getenv('ULTRAVOX_MODEL_PATH')
            
        if os.getenv('TTS_ENGINE'):
            config.tts.engine = os.getenv('TTS_ENGINE')
            
        if os.getenv('MEMORY_BACKEND'):
            config.memory.backend = os.getenv('MEMORY_BACKEND')
            
        if os.getenv('SERVER_PORT'):
            config.server.port = int(os.getenv('SERVER_PORT'))
            
        if os.getenv('SERVER_HOST'):
            config.server.host = os.getenv('SERVER_HOST')
            
        return config
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'UnifiedConfig':
        """
        Load configuration from JSON or YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            UnifiedConfig instance
        """
        import json
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
        # Create config from dictionary
        config = cls()
        
        # Update from loaded data
        if 'ultravox' in data:
            for key, value in data['ultravox'].items():
                if hasattr(config.ultravox, key):
                    setattr(config.ultravox, key, value)
                    
        if 'tts' in data:
            for key, value in data['tts'].items():
                if hasattr(config.tts, key):
                    setattr(config.tts, key, value)
                    
        if 'memory' in data:
            for key, value in data['memory'].items():
                if hasattr(config.memory, key):
                    setattr(config.memory, key, value)
                    
        if 'webrtc' in data:
            for key, value in data['webrtc'].items():
                if hasattr(config.webrtc, key):
                    setattr(config.webrtc, key, value)
                    
        if 'server' in data:
            for key, value in data['server'].items():
                if hasattr(config.server, key):
                    setattr(config.server, key, value)
                    
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)