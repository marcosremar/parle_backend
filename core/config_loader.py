"""
Configuration loader utilities for Ultravox Pipeline.

⚠️  DEPRECATED as of v5.2 - Use src.core.config instead!

This module is deprecated. Please migrate to the new unified configuration system:

    # OLD (deprecated):
    from src.core.config_loader import get_service_ports
    ports = get_service_ports()

    # NEW (recommended):
    from src.core.config import get_config
    config = get_config()
    ports = config.get_service_ports()

See docs/CONFIG_MIGRATION_GUIDE.md for migration instructions.

Legacy Documentation:
---------------------
Provides centralized functions to load configuration from YAML files.
Eliminates hardcoded configuration values across the codebase.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from loguru import logger


def load_services_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load services configuration from services_config.yaml.

    This function provides a single source of truth for service configuration,
    eliminating hardcoded values (like ports) scattered across the codebase.

    Args:
        config_path: Path to services_config.yaml (defaults to project root)

    Returns:
        Dict containing full services configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if config_path is None:
        # Default to project root
        config_path = Path(__file__).parent.parent.parent / "services_config.yaml"

    if not config_path.exists():
        logger.error(f"❌ services_config.yaml not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.debug(f"✅ Loaded services configuration from {config_path}")
        return config or {}

    except yaml.YAMLError as e:
        logger.error(f"❌ Failed to parse {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error loading {config_path}: {e}")
        raise


def get_service_ports(config_path: Optional[Path] = None) -> Dict[str, int]:
    """
    Extract service ports from services_config.yaml.

    Returns a mapping of service_name -> port number.
    Used by Communication Manager to avoid hardcoded ports.

    Args:
        config_path: Path to services_config.yaml (defaults to project root)

    Returns:
        Dict mapping service names to port numbers

    Example:
        {
            'llm': 8100,
            'tts': 8101,
            'stt': 8099,
            'external_llm': 8888,
            ...
        }
    """
    try:
        config = load_services_config(config_path)
        ports = {}

        # Extract ports from services section
        services = config.get('services', {})
        for service_id, service_config in services.items():
            if isinstance(service_config, dict) and 'port' in service_config:
                # Use service_id as key (matches Communication Manager expectations)
                ports[service_id] = service_config['port']

        logger.info(f"📊 Loaded {len(ports)} service ports from configuration")
        return ports

    except Exception as e:
        logger.warning(
            f"⚠️  Failed to load service ports from config: {e}. "
            "Falling back to empty dict (Communication Manager will use discovery)"
        )
        return {}


def get_service_config(service_id: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific service.

    Args:
        service_id: Service identifier (e.g., 'llm', 'external_stt')
        config_path: Path to services_config.yaml (defaults to project root)

    Returns:
        Dict containing service configuration

    Example:
        config = get_service_config('external_llm')
        # Returns:
        # {
        #     'name': 'External LLM Service',
        #     'port': 8888,
        #     'enabled': True,
        #     'profile': 'module',
        #     ...
        # }
    """
    try:
        full_config = load_services_config(config_path)
        services = full_config.get('services', {})

        if service_id not in services:
            logger.warning(f"⚠️  Service '{service_id}' not found in configuration")
            return {}

        return services[service_id]

    except Exception as e:
        logger.error(f"❌ Failed to get config for service '{service_id}': {e}")
        return {}


def get_profile_services(profile: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get all services enabled for a specific profile.

    Args:
        profile: Profile name (e.g., 'dev-local', 'gpu-prod')
        config_path: Path to services_config.yaml (defaults to project root)

    Returns:
        Dict of services enabled in this profile

    Example:
        services = get_profile_services('dev-local')
        # Returns only services where profile='dev-local' or profile='*'
    """
    try:
        full_config = load_services_config(config_path)
        services = full_config.get('services', {})

        profile_services = {}
        for service_id, service_config in services.items():
            if isinstance(service_config, dict):
                service_profile = service_config.get('profile', '')

                # Include if exact match or wildcard
                if service_profile == profile or service_profile == '*':
                    profile_services[service_id] = service_config

        logger.info(f"📋 Found {len(profile_services)} services for profile '{profile}'")
        return profile_services

    except Exception as e:
        logger.error(f"❌ Failed to get profile services for '{profile}': {e}")
        return {}
