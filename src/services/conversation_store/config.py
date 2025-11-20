"""
Conversation Store Service Configuration

This file contains ONLY private/deployment-specific settings:
- Redis URLs with credentials
- Storage paths (production vs development)
- API keys
- Secrets

All database architecture settings are in storage.py.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

# Load environment configuration


class Config:
    """Configuration class for Conversation Store service"""

    def __init__(self) -> None:
        """Initialize configuration by loading .env file"""
        self._load_env()
        self._load_config()

    def _load_env(self) -> None:
        """Load environment variables from .env file"""
        # Find .env in project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent  # Go up to project root
        env_file = project_root / ".env"

        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    # Parse KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        # Only set if not already in environment
                        if key and not os.getenv(key):
                            os.environ[key] = value

    def _load_config(self) -> None:
        """Load service-specific configuration from environment variables"""
        # Common configurations
        self.DEBUG = self._get_bool('DEBUG', False)
        self.LOG_LEVEL = self._get_str('LOG_LEVEL', 'INFO')

        # Storage configuration (PRIVATE - deployment-specific)
        self.ENVIRONMENT = self._get_str('ENVIRONMENT', 'development')

        # Redis (private URLs and credentials)
        self.ENABLE_REDIS = self._get_bool('ENABLE_REDIS', False)
        self.REDIS_URL = self._get_str('REDIS_URL', os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.REDIS_PASSWORD = self._get_str('REDIS_PASSWORD')
        self.REDIS_USERNAME = self._get_str('REDIS_USERNAME')
        self.REDIS_SSL = self._get_bool('REDIS_SSL', False)

        # Storage directories (deployment-specific)
        if self.ENVIRONMENT == 'production':
            self.STORAGE_DIR = Path(self._get_str('STORAGE_DIR', os.path.expanduser("~/.cache/ultravox-pipeline/conversation_store")))
        elif self.ENVIRONMENT == 'staging':
            # Use CONVERSATION_STORE_STAGING_PATH from .env (with fallback)
            staging_path = self._get_str('CONVERSATION_STORE_STAGING_PATH', os.path.expanduser("~/.cache/ultravox-pipeline/conversation_store_staging"))
            self.STORAGE_DIR = Path(staging_path)
        else:
            # Use CONVERSATION_STORE_PATH from .env (with fallback)
            store_path = self._get_str('CONVERSATION_STORE_PATH', os.path.expanduser("~/.cache/ultravox-pipeline/conversation_store"))
            self.STORAGE_DIR = Path(store_path)

    def _get_str(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get string value from environment"""
        return os.getenv(key, default)

    def _get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get integer value from environment"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')

    def _get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Get float value from environment"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default


# Global config instance
config = Config()


def get_storage_config() -> Dict[str, Any]:
    """
    Get storage configuration for FastConversationStorage

    Returns deployment-specific overrides for storage.py defaults.
    """
    return {
        "enable_redis": config.ENABLE_REDIS,
        "redis_url": config.REDIS_URL,
        "storage_dir": config.STORAGE_DIR
    }


def get_redis_credentials() -> Dict[str, Optional[Any]]:
    """
    Get Redis credentials (if using authenticated Redis)

    Returns:
        dict: Redis connection parameters
    """
    return {
        "url": config.REDIS_URL,
        "password": config.REDIS_PASSWORD,
        "username": config.REDIS_USERNAME,
        "ssl": config.REDIS_SSL,
    }
