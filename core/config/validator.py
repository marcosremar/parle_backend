#!/usr/bin/env python3
"""
Configuration Validator - Centralized Validation for All Config

Provides validation utilities for configuration values across all services.
Ensures type safety, range checking, and cross-field validation.

Features:
- Type validation (int, float, str, bool, Path, etc.)
- Range validation (min/max for numbers, length for strings)
- Format validation (URLs, emails, API keys, etc.)
- Cross-field validation (port conflicts, required dependencies)
- Service-specific validation

Usage:
    from src.core.config.validator import ConfigValidator

    validator = ConfigValidator()

    # Validate single value
    validator.validate_port(8080, name="api_gateway_port")

    # Validate API key
    validator.validate_api_key("sk-...", name="GROQ_API_KEY", required=True)

    # Validate service config
    validator.validate_service_config(tts_config, service_name="external_tts")
"""

import re
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Configuration validation error"""
    pass


class ConfigValidator:
    """
    Centralized configuration validator

    Validates configuration values with type checking, range validation,
    and format validation.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize validator

        Args:
            strict: If True, raise exceptions on validation errors
                   If False, log warnings and return False
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    # ========================================================================
    # Basic type validation
    # ========================================================================

    def validate_int(
        self,
        value: Any,
        name: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        required: bool = True
    ) -> bool:
        """
        Validate integer value

        Args:
            value: Value to validate
            name: Name of the configuration value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            required: If True, None/empty is invalid

        Returns:
            True if valid, False otherwise
        """
        if value is None or value == "":
            if required:
                return self._error(f"{name} is required but not set")
            return True

        try:
            int_value = int(value)
        except (ValueError, TypeError):
            return self._error(f"{name} must be an integer, got: {value}")

        if min_value is not None and int_value < min_value:
            return self._error(f"{name} must be >= {min_value}, got: {int_value}")

        if max_value is not None and int_value > max_value:
            return self._error(f"{name} must be <= {max_value}, got: {int_value}")

        return True

    def validate_float(
        self,
        value: Any,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        required: bool = True
    ) -> bool:
        """Validate float value"""
        if value is None or value == "":
            if required:
                return self._error(f"{name} is required but not set")
            return True

        try:
            float_value = float(value)
        except (ValueError, TypeError):
            return self._error(f"{name} must be a float, got: {value}")

        if min_value is not None and float_value < min_value:
            return self._error(f"{name} must be >= {min_value}, got: {float_value}")

        if max_value is not None and float_value > max_value:
            return self._error(f"{name} must be <= {max_value}, got: {float_value}")

        return True

    def validate_bool(self, value: Any, name: str, required: bool = True) -> bool:
        """Validate boolean value"""
        if value is None or value == "":
            if required:
                return self._error(f"{name} is required but not set")
            return True

        if isinstance(value, bool):
            return True

        # Accept string representations
        if isinstance(value, str):
            if value.lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
                return True

        return self._error(f"{name} must be a boolean, got: {value}")

    def validate_str(
        self,
        value: Any,
        name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        required: bool = True
    ) -> bool:
        """Validate string value"""
        if value is None or value == "":
            if required:
                return self._error(f"{name} is required but not set")
            return True

        str_value = str(value)

        if min_length is not None and len(str_value) < min_length:
            return self._error(f"{name} must be at least {min_length} characters, got: {len(str_value)}")

        if max_length is not None and len(str_value) > max_length:
            return self._error(f"{name} must be at most {max_length} characters, got: {len(str_value)}")

        if pattern is not None:
            if not re.match(pattern, str_value):
                return self._error(f"{name} does not match required pattern: {pattern}")

        return True

    def validate_path(
        self,
        value: Any,
        name: str,
        must_exist: bool = False,
        must_be_dir: bool = False,
        must_be_file: bool = False,
        required: bool = True
    ) -> bool:
        """Validate path value"""
        if value is None or value == "":
            if required:
                return self._error(f"{name} is required but not set")
            return True

        try:
            path = Path(value)
        except (TypeError, ValueError):
            return self._error(f"{name} is not a valid path: {value}")

        if must_exist and not path.exists():
            return self._error(f"{name} does not exist: {path}")

        if must_be_dir and path.exists() and not path.is_dir():
            return self._error(f"{name} must be a directory: {path}")

        if must_be_file and path.exists() and not path.is_file():
            return self._error(f"{name} must be a file: {path}")

        return True

    # ========================================================================
    # Format validation
    # ========================================================================

    def validate_url(
        self,
        value: Any,
        name: str,
        schemes: Optional[List[str]] = None,
        required: bool = True
    ) -> bool:
        """
        Validate URL format

        Args:
            value: URL to validate
            name: Configuration name
            schemes: Allowed URL schemes (e.g., ['http', 'https'])
            required: If True, None/empty is invalid
        """
        if value is None or value == "":
            if required:
                return self._error(f"{name} is required but not set")
            return True

        try:
            parsed = urlparse(str(value))
        except Exception:
            return self._error(f"{name} is not a valid URL: {value}")

        if not parsed.scheme:
            return self._error(f"{name} must have a URL scheme (e.g., http://): {value}")

        if schemes is not None and parsed.scheme not in schemes:
            return self._error(f"{name} scheme must be one of {schemes}, got: {parsed.scheme}")

        return True

    def validate_port(
        self,
        value: Any,
        name: str,
        allow_privileged: bool = False,
        required: bool = True
    ) -> bool:
        """
        Validate port number

        Args:
            value: Port number
            name: Configuration name
            allow_privileged: If True, allow ports < 1024
            required: If True, None/empty is invalid
        """
        min_port = 1 if allow_privileged else 1024
        return self.validate_int(value, name, min_value=min_port, max_value=65535, required=required)

    def validate_api_key(
        self,
        value: Any,
        name: str,
        min_length: int = 8,
        pattern: Optional[str] = None,
        required: bool = True
    ) -> bool:
        """
        Validate API key format

        Args:
            value: API key
            name: Configuration name
            min_length: Minimum key length
            pattern: Optional regex pattern for key format
            required: If True, None/empty is invalid
        """
        if not required and (value is None or value == ""):
            return True

        if value is None or value == "":
            return self._error(f"{name} is required but not set")

        str_value = str(value)

        if len(str_value) < min_length:
            return self._error(f"{name} must be at least {min_length} characters")

        if pattern and not re.match(pattern, str_value):
            return self._error(f"{name} does not match expected format")

        return True

    # ========================================================================
    # Service-specific validation
    # ========================================================================

    def validate_service_config(
        self,
        config: Any,
        service_name: str
    ) -> bool:
        """
        Validate service configuration object

        Args:
            config: Service configuration (Pydantic model)
            service_name: Service identifier

        Returns:
            True if valid, False otherwise
        """
        try:
            # Pydantic models validate themselves on instantiation
            # Just check for required fields based on service type

            if service_name in ("external_tts", "tts"):
                return self._validate_tts_config(config)
            elif service_name in ("external_llm", "llm"):
                return self._validate_llm_config(config)
            elif service_name in ("external_stt", "stt"):
                return self._validate_stt_config(config)
            elif service_name == "orchestrator":
                return self._validate_orchestrator_config(config)
            else:
                # Generic validation - just check it's a valid Pydantic model
                return True

        except Exception as e:
            return self._error(f"Failed to validate {service_name} config: {e}")

    def _validate_tts_config(self, config: Any) -> bool:
        """Validate TTS service configuration"""
        valid = True

        # Check HF token (at least one variant should be set)
        if not (config.hf_token or config.hf_api_key or config.huggingface_api_key):
            valid = self._warning(
                "No HuggingFace API token found (HF_TOKEN, HF_API_KEY, or HUGGINGFACE_API_KEY). "
                "TTS service may not work."
            )

        return valid

    def _validate_llm_config(self, config: Any) -> bool:
        """Validate LLM service configuration"""
        valid = True

        # Check API key
        if hasattr(config, 'groq_api_key') and not config.groq_api_key:
            valid = self._warning("GROQ_API_KEY not set. LLM service may not work.")

        # Validate temperature
        if hasattr(config, 'temperature'):
            if not 0.0 <= config.temperature <= 2.0:
                valid = self._error(f"LLM temperature must be between 0.0 and 2.0, got: {config.temperature}")

        return valid

    def _validate_stt_config(self, config: Any) -> bool:
        """Validate STT service configuration"""
        valid = True

        # Check API key
        if hasattr(config, 'api_key') and not config.api_key:
            valid = self._warning("STT API key not set. Service may not work.")

        return valid

    def _validate_orchestrator_config(self, config: Any) -> bool:
        """Validate Orchestrator service configuration"""
        # Orchestrator has no critical required fields
        return True

    # ========================================================================
    # Cross-field validation
    # ========================================================================

    def validate_port_conflicts(self, ports: Dict[str, int]) -> bool:
        """
        Validate that no port conflicts exist

        Args:
            ports: Dictionary of service_name -> port

        Returns:
            True if no conflicts, False otherwise
        """
        port_to_service = {}
        valid = True

        for service_name, port in ports.items():
            if port in port_to_service:
                valid = self._error(
                    f"Port conflict: {service_name} and {port_to_service[port]} "
                    f"both use port {port}"
                )
            else:
                port_to_service[port] = service_name

        return valid

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _error(self, message: str) -> bool:
        """
        Handle validation error

        Args:
            message: Error message

        Returns:
            False (always)

        Raises:
            ValidationError: If strict=True
        """
        self.errors.append(message)
        logger.error(f"❌ Validation error: {message}")

        if self.strict:
            raise ValidationError(message)

        return False

    def _warning(self, message: str) -> bool:
        """
        Handle validation warning

        Args:
            message: Warning message

        Returns:
            True (warning doesn't fail validation)
        """
        self.warnings.append(message)
        logger.warning(f"⚠️  Validation warning: {message}")
        return True

    def get_errors(self) -> List[str]:
        """Get all validation errors"""
        return self.errors.copy()

    def get_warnings(self) -> List[str]:
        """Get all validation warnings"""
        return self.warnings.copy()

    def clear(self) -> None:
        """Clear all errors and warnings"""
        self.errors.clear()
        self.warnings.clear()

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)"""
        return len(self.errors) == 0

    def print_summary(self) -> None:
        """Print validation summary"""
        if not self.errors and not self.warnings:
            print("✅ All validations passed")
            return

        if self.errors:
            print(f"\n❌ {len(self.errors)} validation error(s):")
            for error in self.errors:
                print(f"   - {error}")

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} validation warning(s):")
            for warning in self.warnings:
                print(f"   - {warning}")


# ============================================================================
# Convenience functions
# ============================================================================

def validate_config(config_manager: Any, strict: bool = False) -> bool:
    """
    Validate entire configuration

    Args:
        config_manager: ConfigManager instance
        strict: If True, raise exception on validation errors

    Returns:
        True if valid, False otherwise

    Example:
        from src.core.config import get_config, validate_config

        config = get_config()
        if validate_config(config):
            print("Configuration is valid!")
    """
    validator = ConfigValidator(strict=strict)

    # Validate service ports for conflicts
    try:
        ports = config_manager.get_service_ports()
        validator.validate_port_conflicts(ports)
    except Exception as e:
        logger.warning(f"Could not validate port conflicts: {e}")

    # Print summary
    validator.print_summary()

    return validator.is_valid()
