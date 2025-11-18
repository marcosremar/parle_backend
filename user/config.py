"""
Configuration Management for User Service
Supports hierarchical configuration with overrides from external YAML files
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy


# Default configuration embedded in the service
DEFAULT_CONFIG = {
    "service": {
        "name": "user-service",
        "port": 8200,
        "host": "0.0.0.0"
    },
    "session": {
        "expires_in_hours": 24,
        "cleanup_interval_minutes": 60,
        "max_concurrent_sessions_per_user": 5
    },
    "api_keys": {
        "default_expires_in_days": 365,
        "max_keys_per_user": 10,
        "cleanup_interval_minutes": 1440  # 24 hours
    },
    "security": {
        "hash_algorithm": "sha256",
        "min_password_length": 8,
        "require_email_verification": False,
        "allow_registration": True,
        "rate_limit": {
            "enabled": False,
            "max_requests_per_minute": 60
        }
    },
    "storage": {
        "type": "memory",  # memory, redis, postgres
        "backup_enabled": False,
        "backup_interval_minutes": 30
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}


def deep_merge(base: Dict[Any, Any], override: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively merge two dictionaries

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary (base is not modified)
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def load_config() -> Dict[Any, Any]:
    """
    Load configuration with hierarchical override support

    Configuration priority (highest to lowest):
    1. /workspace/ultravox-pipeline/config/user.yaml (external override)
    2. Default configuration (embedded in service)

    Returns:
        Merged configuration dictionary
    """
    # Start with default config
    config = deepcopy(DEFAULT_CONFIG)

    # Check for external override file (dynamic project root detection)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    override_file = project_root / "config" / "user.yaml"

    if override_file.exists():
        try:
            with open(override_file, 'r') as f:
                override_config = yaml.safe_load(f)

            if override_config:
                # Merge override configuration
                config = deep_merge(config, override_config)
                print(f"✅ Loaded configuration overrides from {override_file}")
        except Exception as e:
            print(f"⚠️  Failed to load override config from {override_file}: {e}")
            print("   Using default configuration")

    return config


# Global configuration instance
_config: Dict[Any, Any] = None


def get_config() -> Dict[Any, Any]:
    """
    Get current configuration (lazy-loaded singleton)

    Returns:
        Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Dict[Any, Any]:
    """
    Force reload configuration from files

    Returns:
        Reloaded configuration dictionary
    """
    global _config
    _config = load_config()
    return _config


def get(key_path: str, default: Any = None) -> Any:
    """
    Get configuration value by dot-notation path

    Examples:
        get("session.expires_in_hours")  # Returns 24
        get("security.min_password_length")  # Returns 8
        get("nonexistent.key", "default")  # Returns "default"

    Args:
        key_path: Dot-separated path to configuration key
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    config = get_config()
    keys = key_path.split('.')

    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def print_config():
    """Print current configuration (for debugging)"""
    import json
    config = get_config()
    print("=" * 60)
    print("User Service Configuration")
    print("=" * 60)
    print(json.dumps(config, indent=2))
    print("=" * 60)
