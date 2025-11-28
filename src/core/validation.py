"""
Data Validation Utilities
Provides robust validation for audio, text, and other inputs
"""
import base64
import re
from typing import Optional, Tuple, List
from pydantic import BaseModel, ValidationError, field_validator
import logging

logger = logging.getLogger(__name__)


# Configuration constants
MAX_AUDIO_SIZE_MB = 10  # 10MB max audio file
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024
MAX_TEXT_LENGTH = 10000  # Max characters for text input
MAX_TOKENS_LLM = 2000  # Max tokens for LLM
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac"]
SUPPORTED_SAMPLE_RATES = [8000, 16000, 24000, 44100, 48000]


class AudioValidationError(Exception):
    """Exception for audio validation errors"""
    pass


class TextValidationError(Exception):
    """Exception for text validation errors"""
    pass


def validate_audio_size(audio_data: bytes, max_size: int = MAX_AUDIO_SIZE_BYTES) -> Tuple[bool, Optional[str]]:
    """
    Validate audio file size
    
    Args:
        audio_data: Audio bytes
        max_size: Maximum size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(audio_data) == 0:
        return False, "Audio data is empty"
    
    if len(audio_data) > max_size:
        size_mb = len(audio_data) / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        return False, f"Audio file too large: {size_mb:.2f}MB (max: {max_mb}MB)"
    
    return True, None


def validate_audio_base64(audio_base64: str, max_size: int = MAX_AUDIO_SIZE_BYTES) -> Tuple[bool, Optional[str], Optional[bytes]]:
    """
    Validate and decode base64 audio
    
    Args:
        audio_base64: Base64 encoded audio string
        max_size: Maximum size in bytes
        
    Returns:
        Tuple of (is_valid, error_message, decoded_bytes)
    """
    if not audio_base64:
        return False, "Audio data is empty", None
    
    try:
        # Decode base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # Validate size
        is_valid, error = validate_audio_size(audio_bytes, max_size)
        if not is_valid:
            return False, error, None
        
        return True, None, audio_bytes
    
    except Exception as e:
        return False, f"Invalid base64 audio data: {e}", None


def validate_sample_rate(sample_rate: int) -> Tuple[bool, Optional[str]]:
    """
    Validate audio sample rate
    
    Args:
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        return False, f"Unsupported sample rate: {sample_rate}Hz (supported: {SUPPORTED_SAMPLE_RATES})"
    
    return True, None


def validate_text_length(text: str, max_length: int = MAX_TEXT_LENGTH) -> Tuple[bool, Optional[str]]:
    """
    Validate text length
    
    Args:
        text: Text to validate
        max_length: Maximum length in characters
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text:
        return False, "Text is empty"
    
    if len(text) > max_length:
        return False, f"Text too long: {len(text)} characters (max: {max_length})"
    
    return True, None


def sanitize_text(text: str) -> str:
    """
    Sanitize user input text
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text


def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text (rough approximation)
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def validate_text_for_llm(text: str, max_tokens: int = MAX_TOKENS_LLM) -> Tuple[bool, Optional[str], int]:
    """
    Validate text for LLM processing
    
    Args:
        text: Text to validate
        max_tokens: Maximum tokens allowed
        
    Returns:
        Tuple of (is_valid, error_message, estimated_tokens)
    """
    # Sanitize first
    sanitized = sanitize_text(text)
    
    if not sanitized:
        return False, "Text is empty after sanitization", 0
    
    # Estimate tokens
    estimated_tokens = estimate_tokens(sanitized)
    
    if estimated_tokens > max_tokens:
        return False, f"Text too long: ~{estimated_tokens} tokens (max: {max_tokens})", estimated_tokens
    
    return True, None, estimated_tokens


def validate_voice_id(voice_id: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate and normalize voice ID
    
    Args:
        voice_id: Voice ID to validate
        
    Returns:
        Tuple of (is_valid, error_message, normalized_voice_id)
    """
    if voice_id is None:
        return True, None, None
    
    if not isinstance(voice_id, str):
        return False, "Voice ID must be a string", None
    
    # Normalize: trim whitespace, convert to None if empty
    normalized = voice_id.strip() if voice_id else None
    if not normalized or normalized.lower() in ["none", "null", ""]:
        return True, None, None
    
    # Basic validation: alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', normalized):
        return False, f"Invalid voice ID format: {voice_id}", None
    
    return True, None, normalized


class AudioInputValidator:
    """Validator for audio input data"""
    
    def __init__(
        self,
        max_size_bytes: int = MAX_AUDIO_SIZE_BYTES,
        supported_formats: List[str] = None,
        supported_sample_rates: List[int] = None
    ):
        self.max_size_bytes = max_size_bytes
        self.supported_formats = supported_formats or SUPPORTED_AUDIO_FORMATS
        self.supported_sample_rates = supported_sample_rates or SUPPORTED_SAMPLE_RATES
    
    def validate(
        self,
        audio_data: Optional[bytes] = None,
        audio_base64: Optional[str] = None,
        sample_rate: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[bytes]]:
        """
        Validate audio input
        
        Args:
            audio_data: Audio bytes (if provided directly)
            audio_base64: Base64 encoded audio (if provided)
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (is_valid, error_message, decoded_audio_bytes)
        """
        # Get audio bytes
        if audio_data:
            audio_bytes = audio_data
        elif audio_base64:
            is_valid, error, decoded = validate_audio_base64(audio_base64, self.max_size_bytes)
            if not is_valid:
                return False, error, None
            audio_bytes = decoded
        else:
            return False, "Either audio_data or audio_base64 must be provided", None
        
        # Validate size
        is_valid, error = validate_audio_size(audio_bytes, self.max_size_bytes)
        if not is_valid:
            return False, error, None
        
        # Validate sample rate if provided
        if sample_rate:
            is_valid, error = validate_sample_rate(sample_rate)
            if not is_valid:
                return False, error, None
        
        return True, None, audio_bytes


class TextInputValidator:
    """Validator for text input data"""
    
    def __init__(
        self,
        max_length: int = MAX_TEXT_LENGTH,
        max_tokens: int = MAX_TOKENS_LLM
    ):
        self.max_length = max_length
        self.max_tokens = max_tokens
    
    def validate(
        self,
        text: str,
        for_llm: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate text input
        
        Args:
            text: Text to validate
            for_llm: Whether validation is for LLM (includes token estimation)
            
        Returns:
            Tuple of (is_valid, error_message, sanitized_text)
        """
        # Sanitize
        sanitized = sanitize_text(text)
        
        # Validate length
        is_valid, error = validate_text_length(sanitized, self.max_length)
        if not is_valid:
            return False, error, None
        
        # Validate for LLM if needed
        if for_llm:
            is_valid, error, estimated_tokens = validate_text_for_llm(sanitized, self.max_tokens)
            if not is_valid:
                return False, error, None
        
        return True, None, sanitized

