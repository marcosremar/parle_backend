#!/usr/bin/env python3
"""
Binary Protocol Adapter
Uses BinaryProtocol for optimized audio transmission
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.protocols.base_adapter import BaseProtocolAdapter
from src.core.audio.binary_protocol import BinaryProtocol


class BinaryProtocolAdapter(BaseProtocolAdapter):
    """Binary protocol adapter for low-latency communication"""

    @property
    def name(self) -> str:
        return "http_binary"

    @property
    def content_type(self) -> str:
        return "application/octet-stream"

    def encode_audio(self, audio_data: bytes, sample_rate: int,
                    metadata: Dict[str, Any] = None) -> Tuple[bytes, Dict[str, str]]:
        """Encode audio using binary protocol"""
        meta = metadata or {}
        meta['sample_rate'] = sample_rate

        # Pack using BinaryProtocol
        binary_data = BinaryProtocol.pack_audio_request(audio_data, meta)

        headers = {
            'Content-Type': self.content_type
        }

        return binary_data, headers

    def decode_audio_response(self, response_data: bytes,
                             content_type: str) -> Dict[str, Any]:
        """Decode binary protocol response"""
        # Unpack binary response
        msg_type, metadata, payload = BinaryProtocol.unpack_message(response_data)

        # Payload is text response
        text_response = payload.decode('utf-8')

        return {
            'text': text_response,
            'metadata': metadata,
            'msg_type': msg_type
        }

    def encode_text(self, text: str, metadata: Dict[str, Any] = None) -> Tuple[bytes, Dict[str, str]]:
        """Encode text using binary protocol"""
        meta = metadata or {}

        # Pack using BinaryProtocol
        binary_data = BinaryProtocol.pack_text_request(text, meta)

        headers = {
            'Content-Type': self.content_type
        }

        return binary_data, headers

    def decode_text_response(self, response_data: bytes,
                            content_type: str) -> Dict[str, Any]:
        """Decode binary protocol text response"""
        msg_type, metadata, payload = BinaryProtocol.unpack_message(response_data)
        text_response = payload.decode('utf-8')

        return {
            'text': text_response,
            'metadata': metadata,
            'msg_type': msg_type
        }

    def estimate_size(self, audio_data: bytes = None, text: str = None) -> int:
        """Estimate binary protocol payload size"""
        if audio_data:
            # Header (9 bytes) + minimal metadata (~50 bytes) + audio
            return 9 + 50 + len(audio_data)
        elif text:
            # Header (9 bytes) + minimal metadata (~50 bytes) + text
            return 9 + 50 + len(text.encode('utf-8'))
        return 0
