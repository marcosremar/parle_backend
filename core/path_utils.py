"""
Path utilities for Ultravox Pipeline (DRY v5.2)

Eliminates duplicated project root setup code found in 26+ service.py files.

Before (duplicated in every service.py):
    from pathlib import Path
    import sys
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

After (one import):
    from src.core.path_utils import ensure_project_root_in_path
    ensure_project_root_in_path()

Usage:
    from src.core.path_utils import ensure_project_root_in_path, get_project_root

    # Ensure project root is in sys.path (idempotent - safe to call multiple times)
    ensure_project_root_in_path()

    # Get project root path
    root = get_project_root()
"""

import sys
from pathlib import Path
from typing import Optional


def get_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Get the project root directory.

    Walks up the directory tree from start_path until finding a directory
    containing a marker file (pyproject.toml, .git, or CLAUDE.md).

    Args:
        start_path: Starting path (defaults to this file's location)

    Returns:
        Path to project root

    Example:
        root = get_project_root()
        config_file = root / "config" / "settings.yaml"
    """
    if start_path is None:
        start_path = Path(__file__)

    # Walk up the tree looking for project markers
    current = start_path.resolve()

    # Marker files that indicate project root
    markers = ["pyproject.toml", ".git", "CLAUDE.md", "main.sh"]

    for parent in [current] + list(current.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent

    # Fallback: assume we're in src/core/, so go up 2 levels
    # This handles cases where markers aren't found (e.g., in tests)
    if "src/core" in str(current):
        return current.parent.parent

    # Last resort: return current directory
    return current


def ensure_project_root_in_path(prepend: bool = True) -> Path:
    """
    Ensure project root is in sys.path.

    This is idempotent - safe to call multiple times without duplicating entries.

    Args:
        prepend: If True, add to beginning of sys.path (higher priority)
                 If False, add to end of sys.path (lower priority)

    Returns:
        Path to project root

    Example:
        from src.core.path_utils import ensure_project_root_in_path

        # Add project root to sys.path (if not already present)
        ensure_project_root_in_path()

        # Now you can import from src/ even if running from nested directories
        from src.core.base_service import BaseService
    """
    project_root = get_project_root()
    project_root_str = str(project_root)

    # Check if already in sys.path (avoid duplicates)
    if project_root_str not in sys.path:
        if prepend:
            sys.path.insert(0, project_root_str)
        else:
            sys.path.append(project_root_str)

    return project_root


def get_service_path(service_name: str) -> Path:
    """
    Get path to a service directory.

    Args:
        service_name: Name of the service (e.g., "external_llm", "session")

    Returns:
        Path to service directory

    Example:
        service_path = get_service_path("external_llm")
        config_file = service_path / "config.yaml"
    """
    project_root = get_project_root()
    return project_root / "src" / "services" / service_name


def get_config_path() -> Path:
    """
    Get path to config directory.

    Returns:
        Path to config directory

    Example:
        config_dir = get_config_path()
        services_config = config_dir / "services_config.yaml"
    """
    project_root = get_project_root()
    return project_root / "config"


def get_logs_path() -> Path:
    """
    Get path to logs directory.

    Returns:
        Path to logs directory (creates if doesn't exist)

    Example:
        logs_dir = get_logs_path()
        log_file = logs_dir / "service.log"
    """
    project_root = get_project_root()
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def get_cache_path() -> Path:
    """
    Get path to cache directory.

    Returns:
        Path to cache directory in ~/.cache/ultravox-pipeline/

    Example:
        cache_dir = get_cache_path()
        model_cache = cache_dir / "models"
    """
    from pathlib import Path
    import os

    cache_dir = Path.home() / ".cache" / "ultravox-pipeline"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


__all__ = [
    'get_project_root',
    'ensure_project_root_in_path',
    'get_service_path',
    'get_config_path',
    'get_logs_path',
    'get_cache_path',
]
