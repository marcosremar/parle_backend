#!/usr/bin/env python3
"""
Environment Manager - Centralized Environment Variable Management

Provides utilities for managing environment variables with type conversion,
validation, and fallback support.

Features:
- Type-safe environment variable access
- Fallback to .env file
- Variable expansion (${VAR:-default})
- Required vs optional variables
- Environment variable validation

Usage:
    from src.core.config.env_manager import EnvManager

    env = EnvManager()

    # Get with type conversion
    api_key = env.get_str("GROQ_API_KEY", required=True)
    max_workers = env.get_int("MAX_WORKERS", default=6)
    debug = env.get_bool("DEBUG", default=False)

    # Get with validation
    port = env.get_int("PORT", default=8080, min_value=1024, max_value=65535)

    # Set environment variable
    env.set("MY_VAR", "value")
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict, Union
from dotenv import load_dotenv, set_key

logger = logging.getLogger(__name__)


class EnvManager:
    """
    Centralized environment variable manager

    Provides type-safe access to environment variables with validation
    and fallback to .env file.
    """

    def __init__(self, env_file: Optional[Path] = None, auto_load: bool = True):
        """
        Initialize environment manager

        Args:
            env_file: Path to .env file (defaults to project root)
            auto_load: If True, automatically load .env file
        """
        # Find project root and .env file
        if env_file is None:
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"

        self.env_file = env_file
        self._loaded = False

        if auto_load:
            self.load_env_file()

    def load_env_file(self, override: bool = False) -> bool:
        """
        Load environment variables from .env file

        Args:
            override: If True, override existing environment variables

        Returns:
            True if file loaded successfully, False otherwise
        """
        if not self.env_file.exists():
            logger.warning(f"⚠️  .env file not found: {self.env_file}")
            return False

        try:
            load_dotenv(self.env_file, override=override)
            self._loaded = True
            logger.info(f"✅ Loaded .env file: {self.env_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load .env file: {e}")
            return False

    # ========================================================================
    # Type-safe getters
    # ========================================================================

    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get environment variable

        Args:
            key: Environment variable name
            default: Default value if not found
            required: If True, raise KeyError if not found

        Returns:
            Environment variable value or default
        """
        value = os.environ.get(key)

        if value is None:
            if required:
                raise KeyError(f"Required environment variable not found: {key}")
            return default

        return value

    def get_str(
        self,
        key: str,
        default: str = "",
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> str:
        """
        Get string environment variable with validation

        Args:
            key: Environment variable name
            default: Default value
            required: If True, raise KeyError if not found
            min_length: Minimum string length
            max_length: Maximum string length

        Returns:
            String value
        """
        value = self.get(key, default, required)
        str_value = str(value) if value is not None else default

        # Validate length
        if min_length is not None and len(str_value) < min_length:
            logger.warning(f"⚠️  {key} is shorter than minimum length {min_length}")

        if max_length is not None and len(str_value) > max_length:
            logger.warning(f"⚠️  {key} is longer than maximum length {max_length}")

        return str_value

    def get_int(
        self,
        key: str,
        default: int = 0,
        required: bool = False,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """
        Get integer environment variable with validation

        Args:
            key: Environment variable name
            default: Default value
            required: If True, raise KeyError if not found
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Integer value
        """
        value = self.get(key, default, required)

        if value is None:
            return default

        try:
            int_value = int(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️  Could not convert {key}={value} to int, using default={default}")
            return default

        # Validate range
        if min_value is not None and int_value < min_value:
            logger.warning(f"⚠️  {key}={int_value} is less than minimum {min_value}")

        if max_value is not None and int_value > max_value:
            logger.warning(f"⚠️  {key}={int_value} is greater than maximum {max_value}")

        return int_value

    def get_float(
        self,
        key: str,
        default: float = 0.0,
        required: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """
        Get float environment variable with validation

        Args:
            key: Environment variable name
            default: Default value
            required: If True, raise KeyError if not found
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Float value
        """
        value = self.get(key, default, required)

        if value is None:
            return default

        try:
            float_value = float(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️  Could not convert {key}={value} to float, using default={default}")
            return default

        # Validate range
        if min_value is not None and float_value < min_value:
            logger.warning(f"⚠️  {key}={float_value} is less than minimum {min_value}")

        if max_value is not None and float_value > max_value:
            logger.warning(f"⚠️  {key}={float_value} is greater than maximum {max_value}")

        return float_value

    def get_bool(self, key: str, default: bool = False, required: bool = False) -> bool:
        """
        Get boolean environment variable

        Args:
            key: Environment variable name
            default: Default value
            required: If True, raise KeyError if not found

        Returns:
            Boolean value
        """
        value = self.get(key, default, required)

        if value is None:
            return default

        if isinstance(value, bool):
            return value

        # String parsing (case-insensitive)
        return str(value).lower() in ("true", "1", "yes", "on")

    def get_list(
        self,
        key: str,
        default: Optional[List[str]] = None,
        required: bool = False,
        separator: str = ","
    ) -> List[str]:
        """
        Get list environment variable (comma-separated by default)

        Args:
            key: Environment variable name
            default: Default value
            required: If True, raise KeyError if not found
            separator: Separator character (default: comma)

        Returns:
            List of strings
        """
        value = self.get(key, default, required)

        if value is None:
            return default or []

        if isinstance(value, list):
            return value

        # Parse separated string
        return [item.strip() for item in str(value).split(separator) if item.strip()]

    def get_path(
        self,
        key: str,
        default: Optional[Path] = None,
        required: bool = False,
        must_exist: bool = False
    ) -> Path:
        """
        Get Path environment variable with expansion

        Args:
            key: Environment variable name
            default: Default value
            required: If True, raise KeyError if not found
            must_exist: If True, warn if path doesn't exist

        Returns:
            Path object with expanded variables
        """
        value = self.get_str(key, required=required)

        if not value:
            return default or Path(".")

        # Expand environment variables and user home
        expanded = os.path.expandvars(value)
        expanded = os.path.expanduser(expanded)

        path = Path(expanded)

        if must_exist and not path.exists():
            logger.warning(f"⚠️  Path for {key} does not exist: {path}")

        return path

    # ========================================================================
    # Setters
    # ========================================================================

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """
        Set environment variable

        Args:
            key: Environment variable name
            value: Value to set
            persist: If True, save to .env file
        """
        os.environ[key] = str(value)

        if persist:
            self.save_to_env_file(key, str(value))

    def save_to_env_file(self, key: str, value: str) -> bool:
        """
        Save environment variable to .env file

        Args:
            key: Environment variable name
            value: Value to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            set_key(self.env_file, key, value)
            logger.info(f"💾 Saved {key} to .env file")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save {key} to .env file: {e}")
            return False

    # ========================================================================
    # Utilities
    # ========================================================================

    def has(self, key: str) -> bool:
        """
        Check if environment variable exists

        Args:
            key: Environment variable name

        Returns:
            True if exists, False otherwise
        """
        return key in os.environ

    def delete(self, key: str) -> bool:
        """
        Delete environment variable

        Args:
            key: Environment variable name

        Returns:
            True if deleted, False if not found
        """
        if key in os.environ:
            del os.environ[key]
            return True
        return False

    def get_all(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Get all environment variables (optionally filtered by prefix)

        Args:
            prefix: Optional prefix to filter by (e.g., "ULTRAVOX_")

        Returns:
            Dictionary of environment variables
        """
        if prefix is None:
            return dict(os.environ)

        return {
            key: value
            for key, value in os.environ.items()
            if key.startswith(prefix)
        }

    def expand_value(self, value: str) -> str:
        """
        Expand environment variables in a string

        Supports ${VAR} and ${VAR:-default} syntax.

        Args:
            value: String to expand

        Returns:
            Expanded string
        """
        if '${' not in value:
            return value

        # Pattern for ${VAR:-default}
        pattern = r'\$\{([^:}]+)(?::-([^}]*))?\}'

        def replace_match(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_value)

        return re.sub(pattern, replace_match, value)

    def validate_required(self, required_vars: List[str]) -> List[str]:
        """
        Validate that required environment variables are set

        Args:
            required_vars: List of required variable names

        Returns:
            List of missing variables
        """
        missing = []

        for var in required_vars:
            if var not in os.environ:
                missing.append(var)
                logger.error(f"❌ Required environment variable not set: {var}")

        return missing

    def print_summary(self, prefix: Optional[str] = "ULTRAVOX_") -> None:
        """
        Print summary of environment variables

        Args:
            prefix: Optional prefix to filter by
        """
        vars = self.get_all(prefix)

        print("\n" + "="*60)
        print(f"📋 ENVIRONMENT VARIABLES ({prefix or 'ALL'})")
        print("="*60)

        if not vars:
            print("(none found)")
        else:
            for key, value in sorted(vars.items()):
                # Mask sensitive values
                if any(sensitive in key.upper() for sensitive in ['KEY', 'TOKEN', 'PASSWORD', 'SECRET']):
                    masked_value = value[:8] + "..." if len(value) > 8 else "***"
                    print(f"{key}: {masked_value}")
                else:
                    print(f"{key}: {value}")

        print("="*60)


# ============================================================================
# Singleton access
# ============================================================================

_env_manager: Optional[EnvManager] = None


def get_env_manager() -> EnvManager:
    """Get singleton EnvManager instance"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvManager()
    return _env_manager
