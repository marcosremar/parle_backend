"""
Webrtc Signaling Service Configuration
Centralized configuration loading from .env file
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration class for Webrtc Signaling service"""

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

        # Service-specific configurations
        # Add your service-specific config here
        # Example:
        # self.API_KEY = self._get_str('SERVICE_API_KEY')
        # self.MAX_CONNECTIONS = self._get_int('SERVICE_MAX_CONNECTIONS', 100)

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
