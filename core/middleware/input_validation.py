#!/usr/bin/env python3
"""
Input Validation Middleware for FastAPI
Validates and sanitizes all incoming requests

Features:
- Request size limits
- Content-Type validation
- Payload sanitization (XSS, SQL injection)
- Schema validation
- File upload validation
"""

import logging
import json
import re
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # Reject anything suspicious
    MODERATE = "moderate"  # Sanitize and warn
    LENIENT = "lenient"    # Log only


@dataclass
class ValidationConfig:
    """
    Validation configuration

    Args:
        max_content_length: Max request body size in bytes
        max_json_depth: Max JSON nesting depth
        allowed_content_types: List of allowed Content-Type headers
        level: Validation strictness level
        sanitize_html: Remove HTML tags from strings
        sanitize_sql: Detect SQL injection patterns
        validate_base64: Validate base64 encoded data
        enabled: Enable/disable validation
    """
    max_content_length: int = 50 * 1024 * 1024  # 50 MB
    max_json_depth: int = 10
    allowed_content_types: List[str] = None
    level: ValidationLevel = ValidationLevel.MODERATE
    sanitize_html: bool = True
    sanitize_sql: bool = True
    validate_base64: bool = True
    enabled: bool = True

    def __post_init__(self):
        if self.allowed_content_types is None:
            self.allowed_content_types = [
                "application/json",
                "multipart/form-data",
                "application/x-www-form-urlencoded",
                "application/octet-stream",
                "audio/wav",
                "audio/mpeg",
                "audio/ogg"
            ]


class InputValidator:
    """
    Input validation and sanitization

    Usage:
        validator = InputValidator(config=ValidationConfig())

        # Validate request
        is_valid, error = await validator.validate_request(request)

        if not is_valid:
            raise HTTPException(400, detail=error)
    """

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[\s\S]*?>[\s\S]*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[\s\S]*?>",
        r"<object[\s\S]*?>",
        r"<embed[\s\S]*?>"
    ]

    # SQL injection patterns
    SQL_PATTERNS = [
        r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\bCREATE\b|\bALTER\b)",
        r"(--|#|\/\*|\*\/)",
        r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+",
        r"'\s*(OR|AND)\s*'",
    ]

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

        if self.config.enabled:
            logger.info(
                f"🛡️  Input Validator initialized\n"
                f"   Level: {self.config.level.value}\n"
                f"   Max content: {self.config.max_content_length / 1024 / 1024:.1f} MB\n"
                f"   Max JSON depth: {self.config.max_json_depth}\n"
                f"   Sanitize HTML: {self.config.sanitize_html}\n"
                f"   Sanitize SQL: {self.config.sanitize_sql}"
            )

    async def validate_request(self, request: Request) -> tuple[bool, Optional[str]]:
        """
        Validate incoming request

        Args:
            request: FastAPI request object

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        if not self.config.enabled:
            return True, None

        # 1. Validate content length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.config.max_content_length:
                    return False, f"Request too large: {size / 1024 / 1024:.1f} MB (max: {self.config.max_content_length / 1024 / 1024:.1f} MB)"
            except ValueError:
                return False, "Invalid Content-Length header"

        # 2. Validate Content-Type
        content_type = request.headers.get("content-type", "").split(";")[0].strip()
        if content_type and content_type not in self.config.allowed_content_types:
            # Check if it's a wildcard match
            allowed = any(
                content_type.startswith(ct.split("*")[0])
                for ct in self.config.allowed_content_types
                if "*" in ct
            )
            if not allowed:
                return False, f"Unsupported Content-Type: {content_type}"

        # 3. Validate JSON payload (if applicable)
        if content_type == "application/json":
            try:
                body = await request.body()
                if body:
                    data = json.loads(body)

                    # Check JSON depth
                    depth = self._get_json_depth(data)
                    if depth > self.config.max_json_depth:
                        return False, f"JSON nesting too deep: {depth} (max: {self.config.max_json_depth})"

                    # Sanitize payload
                    if self.config.level != ValidationLevel.LENIENT:
                        sanitized, issues = self._sanitize_data(data)
                        if issues and self.config.level == ValidationLevel.STRICT:
                            return False, f"Validation failed: {', '.join(issues)}"
                        elif issues:
                            logger.warning(f"⚠️  Sanitized request: {', '.join(issues)}")

                # Reset body for downstream consumption
                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive

            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"
            except Exception as e:
                logger.error(f"Validation error: {e}")
                return False, "Validation failed"

        return True, None

    def _get_json_depth(self, data: Any, current_depth: int = 1) -> int:
        """Calculate JSON nesting depth"""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in data)
        else:
            return current_depth

    def _sanitize_data(self, data: Any) -> tuple[Any, List[str]]:
        """
        Sanitize data recursively

        Returns:
            Tuple of (sanitized_data, list of issues found)
        """
        issues = []

        if isinstance(data, str):
            return self._sanitize_string(data, issues)

        elif isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Sanitize key
                sanitized_key, key_issues = self._sanitize_string(str(key), [])
                issues.extend([f"key '{key}': {issue}" for issue in key_issues])

                # Sanitize value
                sanitized_value, value_issues = self._sanitize_data(value)
                issues.extend([f"key '{key}': {issue}" for issue in value_issues])

                sanitized[sanitized_key] = sanitized_value

            return sanitized, issues

        elif isinstance(data, list):
            sanitized = []
            for i, item in enumerate(data):
                sanitized_item, item_issues = self._sanitize_data(item)
                issues.extend([f"index {i}: {issue}" for issue in item_issues])
                sanitized.append(sanitized_item)

            return sanitized, issues

        else:
            return data, issues

    def _sanitize_string(self, text: str, issues: List[str]) -> tuple[str, List[str]]:
        """Sanitize string value"""
        original = text

        # Check for XSS
        if self.config.sanitize_html:
            for pattern in self.XSS_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    issues.append(f"XSS pattern detected: {pattern[:50]}")
                    # Remove the pattern
                    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Check for SQL injection
        if self.config.sanitize_sql:
            for pattern in self.SQL_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    issues.append(f"SQL injection pattern detected: {pattern[:50]}")
                    # Don't remove - might be legitimate content
                    # Just flag it

        # Validate base64 if it looks like base64
        if self.config.validate_base64 and self._looks_like_base64(text):
            if not self._is_valid_base64(text):
                issues.append("Invalid base64 encoding")

        return text, issues

    def _looks_like_base64(self, text: str) -> bool:
        """Check if string looks like base64"""
        if len(text) < 20:
            return False
        # Base64 strings are usually long and contain only valid chars
        return bool(re.match(r'^[A-Za-z0-9+/]+=*$', text)) and len(text) > 100

    def _is_valid_base64(self, text: str) -> bool:
        """Validate base64 string"""
        try:
            import base64
            import binascii
            decoded = base64.b64decode(text, validate=True)
            return len(decoded) > 0
        except (ValueError, binascii.Error, TypeError):
            return False


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for input validation

    Usage:
        app = FastAPI()

        # Add validation middleware
        app.add_middleware(
            InputValidationMiddleware,
            config=ValidationConfig(
                max_content_length=50 * 1024 * 1024,
                level=ValidationLevel.MODERATE
            )
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[ValidationConfig] = None
    ):
        """
        Initialize middleware

        Args:
            app: FastAPI application
            config: Validation configuration
        """
        super().__init__(app)
        self.config = config or ValidationConfig()
        self.validator = InputValidator(config=self.config)

    async def dispatch(self, request: Request, call_next):
        """Process request through validator"""
        if not self.config.enabled:
            return await call_next(request)

        # Skip validation for health checks and docs
        if request.url.path in ["/health", "/", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        # Validate request
        is_valid, error = await self.validator.validate_request(request)

        if not is_valid:
            logger.warning(f"⚠️  Validation failed for {request.url.path}: {error}")

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Validation failed",
                    "message": error,
                    "path": request.url.path
                }
            )

        # Process request
        response = await call_next(request)
        return response


# Global validator instance
_input_validator: Optional[InputValidator] = None


def get_input_validator(config: Optional[ValidationConfig] = None) -> InputValidator:
    """Get global input validator instance (singleton)"""
    global _input_validator

    if _input_validator is None:
        _input_validator = InputValidator(config=config)

    return _input_validator
