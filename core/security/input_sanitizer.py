"""
Input Sanitizer with XSS, SQL injection, and CSRF protection.

Features:
- XSS prevention (HTML encoding, script tag removal)
- SQL injection detection and prevention
- CSRF token generation and validation
- Path traversal prevention
- File upload validation (magic numbers, size limits)
- JSON depth limiting
- URL validation and sanitization
- Comprehensive input validation

Author: Ultravox Team
Version: 1.0.0
"""

import base64
import hashlib
import html
import mimetypes
import re
import secrets
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator


class SanitizationLevel(str, Enum):
    """Sanitization strictness levels."""

    STRICT = "strict"  # Reject anything suspicious
    MODERATE = "moderate"  # Sanitize and warn (default)
    LENIENT = "lenient"  # Log only, allow everything


class SanitizationConfig(BaseModel):
    """Input sanitizer configuration."""

    # Sanitization level
    level: SanitizationLevel = Field(
        SanitizationLevel.MODERATE, description="Sanitization strictness"
    )

    # XSS protection
    enable_xss_protection: bool = Field(True, description="Enable XSS protection")
    xss_strip_tags: bool = Field(
        True, description="Strip dangerous HTML tags instead of encoding"
    )
    allowed_html_tags: Set[str] = Field(
        default_factory=lambda: {"b", "i", "u", "em", "strong", "p", "br"},
        description="Allowed HTML tags (if not stripping all)",
    )

    # SQL injection protection
    enable_sql_injection_detection: bool = Field(
        True, description="Enable SQL injection detection"
    )

    # CSRF protection
    enable_csrf_protection: bool = Field(True, description="Enable CSRF protection")
    csrf_token_length: int = Field(32, description="CSRF token length in bytes")
    csrf_token_expire_seconds: int = Field(
        3600, description="CSRF token expiration time"
    )

    # File upload validation
    max_file_size_mb: int = Field(50, description="Max file upload size in MB")
    allowed_file_extensions: Set[str] = Field(
        default_factory=lambda: {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".pdf",
            ".txt",
            ".wav",
            ".mp3",
            ".ogg",
        },
        description="Allowed file extensions",
    )
    check_magic_numbers: bool = Field(
        True, description="Validate file type using magic numbers"
    )

    # JSON validation
    max_json_depth: int = Field(10, description="Maximum JSON nesting depth")
    max_json_array_size: int = Field(1000, description="Maximum array size")
    max_json_string_length: int = Field(10000, description="Maximum string length")

    # URL validation
    allowed_url_schemes: Set[str] = Field(
        default_factory=lambda: {"http", "https"}, description="Allowed URL schemes"
    )
    block_private_ips: bool = Field(True, description="Block private IP addresses in URLs")

    # Path traversal protection
    enable_path_traversal_protection: bool = Field(
        True, description="Enable path traversal protection"
    )

    # Timeout protection (ReDoS)
    regex_timeout_seconds: float = Field(
        0.1, description="Maximum time for regex execution"
    )


class InputSanitizer:
    """
    Input sanitizer with XSS, SQL injection, and CSRF protection.

    Example:
        >>> config = SanitizationConfig()
        >>> sanitizer = InputSanitizer(config)
        >>> safe_html = sanitizer.sanitize_html("<script>alert('xss')</script>")
        >>> csrf_token = sanitizer.generate_csrf_token("user123")
        >>> sanitizer.validate_csrf_token(token, "user123")
    """

    # XSS patterns (compiled for performance)
    _XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),  # Event handlers
        re.compile(r"<iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<object[^>]*>", re.IGNORECASE),
        re.compile(r"<embed[^>]*>", re.IGNORECASE),
        re.compile(r"<link[^>]*>", re.IGNORECASE),
        re.compile(r"<meta[^>]*>", re.IGNORECASE),
        re.compile(r"expression\s*\(", re.IGNORECASE),  # CSS expressions
        re.compile(r"vbscript:", re.IGNORECASE),
        re.compile(r"data:text/html", re.IGNORECASE),
    ]

    # SQL injection patterns
    _SQL_PATTERNS = [
        re.compile(r"\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b", re.IGNORECASE),
        re.compile(r"--"),  # SQL comments (removed \s requirement)
        re.compile(r"/\*.*?\*/", re.DOTALL),  # Multi-line SQL comments
        re.compile(r";\s*(DROP|DELETE|TRUNCATE|UPDATE|INSERT)", re.IGNORECASE),
        re.compile(r"'\s*(OR|AND)\s*'?\d*'?\s*=\s*'?\d*'?", re.IGNORECASE),  # OR 1=1
        re.compile(r"'\s*OR\s*'1'\s*=\s*'1", re.IGNORECASE),
        re.compile(r"0x[0-9a-f]+", re.IGNORECASE),  # Hex encoding
    ]

    # Path traversal patterns
    _PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"\.\.[\\/]"),  # ../ or ..\
        re.compile(r"[\\/]\.\."),  # /.. or \..
        re.compile(r"\.\."),  # ..
        re.compile(r"%2e%2e"),  # URL encoded ..
        re.compile(r"%252e%252e"),  # Double URL encoded ..
        re.compile(r"\x00"),  # Null byte
    ]

    # Magic numbers for file type validation
    _MAGIC_NUMBERS = {
        b"\xFF\xD8\xFF": "image/jpeg",
        b"\x89PNG": "image/png",
        b"GIF87a": "image/gif",
        b"GIF89a": "image/gif",
        b"%PDF": "application/pdf",
        b"RIFF": "audio/wav",  # Check for WAVE after this
        b"ID3": "audio/mpeg",  # MP3
        b"OggS": "audio/ogg",
    }

    def __init__(self, config: SanitizationConfig):
        """
        Initialize input sanitizer.

        Args:
            config: Sanitization configuration
        """
        self.config = config

        # CSRF token storage (in production, use Redis or database)
        self._csrf_tokens: Dict[str, Tuple[str, float]] = {}  # user_id -> (token, expiry)

    def sanitize_html(self, html_input: str) -> str:
        """
        Sanitize HTML input to prevent XSS attacks.

        Args:
            html_input: Raw HTML input

        Returns:
            Sanitized HTML

        Example:
            >>> sanitizer.sanitize_html("<script>alert('xss')</script>Hello")
            'Hello'
        """
        if not self.config.enable_xss_protection:
            return html_input

        sanitized = html_input

        if self.config.xss_strip_tags:
            # Remove dangerous tags
            for pattern in self._XSS_PATTERNS:
                sanitized = pattern.sub("", sanitized)
        else:
            # HTML encode
            sanitized = html.escape(sanitized)

        return sanitized

    def detect_xss(self, input_str: str) -> bool:
        """
        Detect potential XSS attacks.

        Args:
            input_str: Input string to check

        Returns:
            True if XSS detected
        """
        if not self.config.enable_xss_protection:
            return False

        for pattern in self._XSS_PATTERNS:
            if pattern.search(input_str):
                return True

        return False

    def detect_sql_injection(self, input_str: str) -> bool:
        """
        Detect potential SQL injection attacks.

        Args:
            input_str: Input string to check

        Returns:
            True if SQL injection detected
        """
        if not self.config.enable_sql_injection_detection:
            return False

        for pattern in self._SQL_PATTERNS:
            if pattern.search(input_str):
                return True

        return False

    def sanitize_sql_input(self, input_str: str) -> str:
        """
        Sanitize input for SQL queries (use parameterized queries instead!).

        Args:
            input_str: Raw input

        Returns:
            Sanitized input

        Note:
            This is a backup. Always use parameterized queries!
        """
        # Escape single quotes
        sanitized = input_str.replace("'", "''")

        # Remove SQL comments
        sanitized = re.sub(r"--.*$", "", sanitized)
        sanitized = re.sub(r"/\*.*?\*/", "", sanitized, flags=re.DOTALL)

        return sanitized

    def generate_csrf_token(self, user_id: str) -> str:
        """
        Generate CSRF token for user.

        Args:
            user_id: User identifier

        Returns:
            CSRF token

        Example:
            >>> token = sanitizer.generate_csrf_token("user123")
        """
        if not self.config.enable_csrf_protection:
            return ""

        # Generate random token
        token = secrets.token_urlsafe(self.config.csrf_token_length)

        # Store with expiry
        expiry = time.time() + self.config.csrf_token_expire_seconds
        self._csrf_tokens[user_id] = (token, expiry)

        return token

    def validate_csrf_token(self, token: str, user_id: str) -> bool:
        """
        Validate CSRF token.

        Args:
            token: CSRF token to validate
            user_id: User identifier

        Returns:
            True if token is valid

        Example:
            >>> if not sanitizer.validate_csrf_token(token, "user123"):
            ...     raise PermissionError("Invalid CSRF token")
        """
        if not self.config.enable_csrf_protection:
            return True

        if user_id not in self._csrf_tokens:
            return False

        stored_token, expiry = self._csrf_tokens[user_id]

        # Check expiry
        if time.time() > expiry:
            del self._csrf_tokens[user_id]
            return False

        # Constant-time comparison to prevent timing attacks
        return secrets.compare_digest(token, stored_token)

    def validate_file_upload(
        self, filename: str, content: bytes, mime_type: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate file upload.

        Args:
            filename: File name
            content: File content bytes
            mime_type: Optional declared MIME type

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            >>> with open("image.jpg", "rb") as f:
            ...     valid, error = sanitizer.validate_file_upload("image.jpg", f.read())
            ...     if not valid:
            ...         print(f"Invalid file: {error}")
        """
        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            return False, f"File too large ({size_mb:.1f}MB > {self.config.max_file_size_mb}MB)"

        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.config.allowed_file_extensions:
            return False, f"File extension not allowed: {file_ext}"

        # Check magic numbers
        if self.config.check_magic_numbers:
            detected_type = self._detect_file_type(content)
            if detected_type is None:
                return False, "Could not detect file type from content"

            # Check if detected type matches extension
            expected_types = mimetypes.guess_type(filename)[0]
            if expected_types and not detected_type.startswith(expected_types.split("/")[0]):
                return (
                    False,
                    f"File content ({detected_type}) doesn't match extension ({file_ext})",
                )

        # Check for path traversal in filename
        if self.config.enable_path_traversal_protection:
            if self.detect_path_traversal(filename):
                return False, "Invalid filename: path traversal detected"

        return True, None

    def _detect_file_type(self, content: bytes) -> Optional[str]:
        """Detect file type from magic numbers."""
        for magic, mime_type in self._MAGIC_NUMBERS.items():
            if content.startswith(magic):
                # Special case for WAV files
                if magic == b"RIFF" and content[8:12] == b"WAVE":
                    return "audio/wav"
                return mime_type
        return None

    def detect_path_traversal(self, path: str) -> bool:
        """
        Detect path traversal attempts.

        Args:
            path: Path to check

        Returns:
            True if path traversal detected
        """
        if not self.config.enable_path_traversal_protection:
            return False

        for pattern in self._PATH_TRAVERSAL_PATTERNS:
            if pattern.search(path):
                return True

        # Check for absolute paths (Unix/Windows)
        if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
            return True

        return False

    def sanitize_path(self, path: str) -> str:
        """
        Sanitize file path to prevent traversal attacks.

        Args:
            path: Raw path

        Returns:
            Sanitized path (only filename)
        """
        # Get only the filename, remove any directory components
        return Path(path).name

    def validate_json_depth(self, obj: Any, current_depth: int = 0) -> bool:
        """
        Validate JSON depth to prevent DoS attacks.

        Args:
            obj: JSON object to validate
            current_depth: Current nesting depth

        Returns:
            True if depth is acceptable
        """
        if current_depth > self.config.max_json_depth:
            return False

        if isinstance(obj, dict):
            for value in obj.values():
                if not self.validate_json_depth(value, current_depth + 1):
                    return False
        elif isinstance(obj, list):
            if len(obj) > self.config.max_json_array_size:
                return False
            for item in obj:
                if not self.validate_json_depth(item, current_depth + 1):
                    return False
        elif isinstance(obj, str):
            if len(obj) > self.config.max_json_string_length:
                return False

        return True

    def sanitize_url(self, url: str) -> Optional[str]:
        """
        Sanitize and validate URL.

        Args:
            url: URL to sanitize

        Returns:
            Sanitized URL or None if invalid
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in self.config.allowed_url_schemes:
                return None

            # Check for private IPs if configured
            if self.config.block_private_ips:
                if self._is_private_ip(parsed.hostname):
                    return None

            # Reconstruct URL (removes dangerous components)
            return parsed.geturl()

        except Exception:
            return None

    def _is_private_ip(self, hostname: Optional[str]) -> bool:
        """Check if hostname is a private IP address."""
        if not hostname:
            return False

        import ipaddress

        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            # Not an IP address
            return False

    def sanitize_input(
        self, input_data: Union[str, Dict[str, Any]], input_type: str = "text"
    ) -> Union[str, Dict[str, Any]]:
        """
        General input sanitization based on type.

        Args:
            input_data: Input data to sanitize
            input_type: Type of input (text, html, sql, json, url)

        Returns:
            Sanitized input
        """
        if input_type == "html":
            return self.sanitize_html(input_data) if isinstance(input_data, str) else input_data

        elif input_type == "sql":
            return (
                self.sanitize_sql_input(input_data)
                if isinstance(input_data, str)
                else input_data
            )

        elif input_type == "url":
            return self.sanitize_url(input_data) if isinstance(input_data, str) else input_data

        elif input_type == "json":
            if isinstance(input_data, dict):
                if not self.validate_json_depth(input_data):
                    raise ValueError("JSON depth exceeds maximum allowed")
            return input_data

        else:  # Default text sanitization
            if isinstance(input_data, str):
                # Basic HTML encoding
                return html.escape(input_data)
            return input_data

    def cleanup_expired_tokens(self) -> None:
        """Clean up expired CSRF tokens."""
        now = time.time()
        expired_users = [
            user_id
            for user_id, (_, expiry) in self._csrf_tokens.items()
            if expiry < now
        ]
        for user_id in expired_users:
            del self._csrf_tokens[user_id]
