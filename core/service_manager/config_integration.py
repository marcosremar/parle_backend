"""
Service Manager Configuration Integration

This module demonstrates how to integrate UnifiedConfigAdapter with
the Service Manager for automatic configuration, validation, and port allocation.

Usage in Service Manager:
    from src.core.service_manager.config_integration import get_service_config, get_startup_order

    # Get configuration (auto-validates)
    config = get_service_config()

    # Get enabled services for current profile
    enabled_services = config.get_enabled_services()

    # Get startup order (topological sort)
    startup_order = get_startup_order()

    # Get port for service (auto-allocates if needed)
    port = config.get_service_port("llm", allocate_if_missing=True)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

# Ensure project root in path
if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config.unified_config_adapter import UnifiedConfigAdapter, get_unified_config

logger = logging.getLogger(__name__)


def get_service_config(profile: Optional[str] = None) -> UnifiedConfigAdapter:
    """
    Get service configuration for Service Manager.

    Args:
        profile: Override active profile (default: from env or config)

    Returns:
        UnifiedConfigAdapter instance (validated and ready)

    Example:
        >>> config = get_service_config(profile="dev-local")
        >>> enabled = config.get_enabled_services()
        >>> print(f"Enabled services: {len(enabled)}")
    """
    # Check for profile override from environment
    if profile is None:
        profile = os.getenv("ULTRAVOX_PROFILE")

    # Get unified config (cached singleton)
    config = get_unified_config(profile_name=profile)

    return config


def get_startup_order(profile: Optional[str] = None) -> List[str]:
    """
    Get service startup order (topological sort based on dependencies).

    Args:
        profile: Override active profile

    Returns:
        List of service names in startup order

    Example:
        >>> order = get_startup_order()
        >>> print(f"Starting services: {' → '.join(order[:5])}...")
    """
    config = get_service_config(profile=profile)
    return config.calculate_startup_order()


def get_enabled_services_dict(profile: Optional[str] = None) -> Dict[str, dict]:
    """
    Get enabled services as dictionary (compatible with existing code).

    Args:
        profile: Override active profile

    Returns:
        Dictionary mapping service_name to service info dict

    Example:
        >>> services = get_enabled_services_dict()
        >>> for name, info in services.items():
        ...     print(f"{name}: port={info['port']}")
    """
    config = get_service_config(profile=profile)
    enabled = config.get_enabled_services()

    # Convert to dict format compatible with existing code
    result = {}
    for name, service_def in enabled.items():
        port = config.get_service_port(name, allocate_if_missing=True)
        result[name] = {
            "name": name,
            "port": port,
            "execution_mode": service_def.execution_mode,
            "module_path": service_def.module_path,
            "venv_path": service_def.venv_path,
            "auto_start": service_def.auto_start,
            "startup_priority": service_def.startup_priority,
            "enabled": service_def.enabled,
            "depends_on": service_def.get_dependencies(),
            "description": service_def.description,
        }

    return result


def validate_service_manager_config() -> Dict[str, any]:
    """
    Validate Service Manager configuration and return report.

    Returns:
        Dictionary with validation results

    Example:
        >>> report = validate_service_manager_config()
        >>> if report["validation_passed"]:
        ...     print("✅ Configuration valid!")
        >>> else:
        ...     print(f"❌ Errors: {report['errors']}")
    """
    config = get_service_config()

    # Run all validations
    port_conflicts = config.check_port_conflicts()
    dep_errors = config.validate_dependencies()
    summary = config.get_config_summary()

    # Compile report
    report = {
        **summary,
        "port_conflicts": port_conflicts,
        "dependency_errors": dep_errors,
        "errors": [],
    }

    # Add errors to report
    if port_conflicts:
        for port, services in port_conflicts.items():
            report["errors"].append(f"Port {port}: {', '.join(services)}")

    if dep_errors:
        report["errors"].extend(dep_errors)

    return report


def print_service_manager_status():
    """
    Print Service Manager configuration status (for debugging).

    Example:
        >>> print_service_manager_status()
        ============================================================
        SERVICE MANAGER CONFIGURATION STATUS
        ============================================================
        Profile: dev-local
        Total services: 29
        Enabled services: 16
        ...
    """
    config = get_service_config()
    report = validate_service_manager_config()

    print("\n" + "=" * 60)
    print("SERVICE MANAGER CONFIGURATION STATUS")
    print("=" * 60)

    # Summary
    print(f"\n📊 Configuration Summary:")
    print(f"  Profile: {report['profile']}")
    print(f"  Total services: {report['total_services']}")
    print(f"  Enabled services: {report['enabled_services']}")
    print(f"  Port conflicts: {report['port_conflicts']}")
    print(f"  Dependency errors: {report['dependency_errors']}")

    # Validation status
    if report["validation_passed"]:
        print(f"\n✅ Configuration validation: PASSED")
    else:
        print(f"\n❌ Configuration validation: FAILED")
        print(f"\n  Errors ({len(report['errors'])}):")
        for error in report["errors"]:
            print(f"    - {error}")

    # Startup order
    try:
        startup_order = get_startup_order()
        print(f"\n🚀 Startup Order ({len(startup_order)} services):")
        print(f"  {' → '.join(startup_order[:8])}...")
        if len(startup_order) > 8:
            print(f"  ... and {len(startup_order) - 8} more")
    except Exception as e:
        print(f"\n❌ Failed to calculate startup order: {e}")

    # Enabled services
    enabled = config.get_enabled_services()
    print(f"\n📋 Enabled Services ({len(enabled)}):")
    for name in sorted(enabled.keys())[:10]:
        service = enabled[name]
        port = config.get_service_port(name, allocate_if_missing=False)
        port_str = f"port={port}" if port else "no port"
        print(f"  • {name}: {service.execution_mode} ({port_str})")
    if len(enabled) > 10:
        print(f"  ... and {len(enabled) - 10} more")

    print("\n" + "=" * 60 + "\n")


# Example integration with Service Manager startup
def initialize_service_manager_config(profile: Optional[str] = None) -> UnifiedConfigAdapter:
    """
    Initialize Service Manager configuration.

    This should be called early in Service Manager startup to:
    - Load and validate configuration
    - Detect errors early (fail-fast)
    - Initialize port pool
    - Calculate startup order

    Args:
        profile: Override active profile

    Returns:
        UnifiedConfigAdapter instance

    Raises:
        ValueError: If configuration validation fails

    Example:
        # In Service Manager main.py
        try:
            config = initialize_service_manager_config()
            logger.info("✅ Configuration initialized successfully")
        except ValueError as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            sys.exit(1)
    """
    logger.info("🔧 Initializing Service Manager Configuration...")

    try:
        # Get and validate config
        config = get_service_config(profile=profile)

        # Log validation results
        report = validate_service_manager_config()

        if not report["validation_passed"]:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in report["errors"]
            )
            raise ValueError(error_msg)

        logger.info(f"✅ Configuration validated successfully")
        logger.info(f"  Profile: {report['profile']}")
        logger.info(f"  Enabled services: {report['enabled_services']}")
        logger.info(f"  No port conflicts detected")
        logger.info(f"  All dependencies valid")

        return config

    except Exception as e:
        logger.error(f"❌ Failed to initialize configuration: {e}")
        raise


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    # Print status
    print_service_manager_status()

    # Test initialization
    print("\n🧪 Testing initialization...")
    try:
        config = initialize_service_manager_config()
        print("✅ Initialization successful!")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)
