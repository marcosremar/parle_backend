#!/usr/bin/env python3
"""
Base Protocol Adapter
Abstract interface for protocol implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseProtocolAdapter(ABC):
    """Abstract base class for protocol adapters"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol name identifier"""
        pass

    @property
    @abstractmethod
    def content_type(self) -> str:
        """HTTP Content-Type header value"""
        pass

    @abstractmethod
    def encode_audio(self, audio_data: bytes, sample_rate: int,
                    metadata: Dict[str, Any] = None) -> Tuple[bytes, Dict[str, str]]:
        """
        Encode audio data for transmission

        Args:
            audio_data: Raw audio bytes (PCM)
            sample_rate: Sample rate in Hz
            metadata: Optional metadata dict

        Returns:
            Tuple of (encoded_data, headers_dict)
        """
        pass

    @abstractmethod
    def decode_audio_response(self, response_data: bytes,
                             content_type: str) -> Dict[str, Any]:
        """
        Decode audio processing response

        Args:
            response_data: Raw response bytes
            content_type: Response Content-Type header

        Returns:
            Dict with 'text', 'metadata', etc.
        """
        pass

    @abstractmethod
    def encode_text(self, text: str, metadata: Dict[str, Any] = None) -> Tuple[bytes, Dict[str, str]]:
        """
        Encode text for transmission

        Args:
            text: Text to send
            metadata: Optional metadata

        Returns:
            Tuple of (encoded_data, headers_dict)
        """
        pass

    @abstractmethod
    def decode_text_response(self, response_data: bytes,
                            content_type: str) -> Dict[str, Any]:
        """
        Decode text processing response

        Args:
            response_data: Raw response bytes
            content_type: Response Content-Type header

        Returns:
            Dict with 'text', 'metadata', etc.
        """
        pass

    @abstractmethod
    def estimate_size(self, audio_data: bytes = None, text: str = None) -> int:
        """
        Estimate encoded payload size

        Args:
            audio_data: Audio bytes (if audio request)
            text: Text string (if text request)

        Returns:
            Estimated size in bytes
        """
        pass
