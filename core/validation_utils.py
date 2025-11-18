"""
Shared validators for Pydantic models (DRY v5.2)

Eliminates duplicated validation logic across 4+ services.

Before (duplicated in each models.py):
    @field_validator("audio_data")
    def validate_audio(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Audio data cannot be empty")
        return v

After (reusable validator):
    from src.core.validators import validate_non_empty_string

    class MyModel(BaseModel):
        audio_base64: str

        _validate_audio = validate_non_empty_string("audio_base64")

Usage:
    from src.core.validators import (
        validate_non_empty_string,
        validate_audio_base64,
        validate_session_id,
        validate_language_code
    )

    from pydantic import BaseModel

    class MyRequest(BaseModel):
        session_id: str
        audio_base64: str
        language: str = "en"

        # Apply validators
        _validate_session_id = validate_session_id("session_id")
        _validate_audio = validate_audio_base64("audio_base64")
        _validate_language = validate_language_code("language")
"""

from typing import Any, Callable, Optional
from pydantic import field_validator
import base64
import re


def validate_non_empty_string(field_name: str, max_length: Optional[int] = None) -> Callable:
    """
    Reusable validator for non-empty strings.

    Args:
        field_name: Name of the field to validate
        max_length: Maximum string length (optional)

    Returns:
        Pydantic field_validator decorator

    Example:
        class MyModel(BaseModel):
            text: str

            _validate_text = validate_non_empty_string("text", max_length=1000)
    """
    @field_validator(field_name)
    @classmethod
    def validator(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError(f"{field_name} cannot be empty")

        if max_length and len(v) > max_length:
            raise ValueError(f"{field_name} exceeds maximum length of {max_length}")

        return v

    return validator


def validate_audio_base64(field_name: str) -> Callable:
    """
    Reusable validator for base64-encoded audio data.

    Validates:
    - Non-empty string
    - Valid base64 encoding
    - Minimum size (> 100 bytes)

    Args:
        field_name: Name of the field to validate

    Returns:
        Pydantic field_validator decorator

    Example:
        class AudioRequest(BaseModel):
            audio_base64: str

            _validate_audio = validate_audio_base64("audio_base64")
    """
    @field_validator(field_name)
    @classmethod
    def validator(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError(f"{field_name} cannot be empty")

        try:
            # Validate base64 encoding
            decoded = base64.b64decode(v)

            # Minimum audio size (100 bytes)
            if len(decoded) < 100:
                raise ValueError(f"{field_name} is too small to be valid audio (< 100 bytes)")

        except Exception as e:
            raise ValueError(f"{field_name} is not valid base64-encoded audio: {e}")

        return v

    return validator


def validate_session_id(field_name: str) -> Callable:
    """
    Reusable validator for session IDs.

    Validates:
    - Non-empty string
    - Length between 1-128 characters
    - Alphanumeric + hyphens/underscores only

    Args:
        field_name: Name of the field to validate

    Returns:
        Pydantic field_validator decorator

    Example:
        class SessionRequest(BaseModel):
            session_id: str

            _validate_session_id = validate_session_id("session_id")
    """
    @field_validator(field_name)
    @classmethod
    def validator(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError(f"{field_name} cannot be empty")

        if len(v) > 128:
            raise ValueError(f"{field_name} exceeds maximum length of 128")

        # Only alphanumeric + hyphens/underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f"{field_name} must contain only alphanumeric characters, hyphens, and underscores")

        return v

    return validator


def validate_language_code(field_name: str) -> Callable:
    """
    Reusable validator for ISO 639-1 language codes.

    Validates:
    - 2-letter language code
    - Supported languages: pt, en, es, fr, de, it, ja, ko, zh, ru

    Args:
        field_name: Name of the field to validate

    Returns:
        Pydantic field_validator decorator

    Example:
        class TranscriptionRequest(BaseModel):
            language: str = "en"

            _validate_language = validate_language_code("language")
    """
    @field_validator(field_name)
    @classmethod
    def validator(cls, v: str) -> str:
        valid_languages = {'pt', 'en', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'zh', 'ru'}

        if not v:
            raise ValueError(f"{field_name} cannot be empty")

        v_lower = v.lower()
        if v_lower not in valid_languages:
            raise ValueError(
                f"{field_name} must be one of: {', '.join(sorted(valid_languages))}"
            )

        return v_lower

    return validator


def validate_positive_integer(field_name: str, min_value: int = 1, max_value: Optional[int] = None) -> Callable:
    """
    Reusable validator for positive integers.

    Args:
        field_name: Name of the field to validate
        min_value: Minimum allowed value (default: 1)
        max_value: Maximum allowed value (optional)

    Returns:
        Pydantic field_validator decorator

    Example:
        class Config(BaseModel):
            max_tokens: int

            _validate_max_tokens = validate_positive_integer("max_tokens", min_value=1, max_value=4096)
    """
    @field_validator(field_name)
    @classmethod
    def validator(cls, v: int) -> int:
        if v < min_value:
            raise ValueError(f"{field_name} must be at least {min_value}")

        if max_value is not None and v > max_value:
            raise ValueError(f"{field_name} must not exceed {max_value}")

        return v

    return validator


def validate_float_range(field_name: str, min_value: float = 0.0, max_value: float = 1.0) -> Callable:
    """
    Reusable validator for floats within a range.

    Args:
        field_name: Name of the field to validate
        min_value: Minimum allowed value (default: 0.0)
        max_value: Maximum allowed value (default: 1.0)

    Returns:
        Pydantic field_validator decorator

    Example:
        class Config(BaseModel):
            temperature: float = 0.7

            _validate_temperature = validate_float_range("temperature", min_value=0.0, max_value=2.0)
    """
    @field_validator(field_name)
    @classmethod
    def validator(cls, v: float) -> float:
        if v < min_value:
            raise ValueError(f"{field_name} must be at least {min_value}")

        if v > max_value:
            raise ValueError(f"{field_name} must not exceed {max_value}")

        return v

    return validator


def validate_url(field_name: str, schemes: Optional[list] = None) -> Callable:
    """
    Reusable validator for URLs.

    Args:
        field_name: Name of the field to validate
        schemes: Allowed URL schemes (default: ["http", "https"])

    Returns:
        Pydantic field_validator decorator

    Example:
        class Config(BaseModel):
            api_url: str

            _validate_api_url = validate_url("api_url")
    """
    if schemes is None:
        schemes = ["http", "https"]

    @field_validator(field_name)
    @classmethod
    def validator(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError(f"{field_name} cannot be empty")

        # Basic URL validation
        url_pattern = re.compile(
            r'^(?:(?:' + '|'.join(schemes) + r')://)' # scheme
            r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if not url_pattern.match(v):
            raise ValueError(f"{field_name} is not a valid URL")

        return v

    return validator


def validate_file_path(field_name: str, must_exist: bool = False) -> Callable:
    """
    Reusable validator for file paths.

    Args:
        field_name: Name of the field to validate
        must_exist: If True, validate that file exists

    Returns:
        Pydantic field_validator decorator

    Example:
        class Config(BaseModel):
            model_path: str

            _validate_model_path = validate_file_path("model_path", must_exist=True)
    """
    @field_validator(field_name)
    @classmethod
    def validator(cls, v: str) -> str:
        from pathlib import Path

        if not v or len(v.strip()) == 0:
            raise ValueError(f"{field_name} cannot be empty")

        path = Path(v)

        if must_exist and not path.exists():
            raise ValueError(f"{field_name} does not exist: {v}")

        return v

    return validator


def validate_enum(field_name: str, allowed_values: list) -> Callable:
    """
    Reusable validator for enum-like fields.

    Args:
        field_name: Name of the field to validate
        allowed_values: List of allowed values

    Returns:
        Pydantic field_validator decorator

    Example:
        class Config(BaseModel):
            mode: str

            _validate_mode = validate_enum("mode", ["development", "production"])
    """
    @field_validator(field_name)
    @classmethod
    def validator(cls, v: str) -> str:
        if v not in allowed_values:
            raise ValueError(
                f"{field_name} must be one of: {', '.join(allowed_values)}"
            )

        return v

    return validator


__all__ = [
    'validate_non_empty_string',
    'validate_audio_base64',
    'validate_session_id',
    'validate_language_code',
    'validate_positive_integer',
    'validate_float_range',
    'validate_url',
    'validate_file_path',
    'validate_enum',
]
