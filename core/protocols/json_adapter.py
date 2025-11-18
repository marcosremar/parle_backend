#!/usr/bin/env python3
"""
JSON Protocol Adapter
Standard JSON-based communication for easy debugging and testing
NO base64 encoding - uses hex strings for better performance
"""

import json
from typing import Dict, Any, Tuple

from src.core.protocols.base_adapter import BaseProtocolAdapter


class JsonProtocolAdapter(BaseProtocolAdapter):
    """JSON protocol adapter for human-readable communication"""

    @property
    def name(self) -> str:
        return "json"

    @property
    def content_type(self) -> str:
        return "application/json"

    def encode_audio(self, audio_data: bytes, sample_rate: int,
                    metadata: Dict[str, Any] = None) -> Tuple[bytes, Dict[str, str]]:
        """Encode audio as JSON with hex string (NO BASE64 - faster!)"""
        # NO conversion to float32, NO base64 encoding
        # Send raw PCM int16 bytes as hex string
        audio_hex = audio_data.hex()

        payload = {
            "audio_data": audio_hex,  # Hex string instead of base64
            "sample_rate": sample_rate,
            "encoding": "pcm_int16_hex"  # Document the encoding format
        }

        # Add metadata fields
        if metadata:
            for key, value in metadata.items():
                if key not in ['sample_rate']:  # Avoid duplicates
                    payload[key] = value

        json_data = json.dumps(payload).encode('utf-8')

        headers = {
            'Content-Type': self.content_type
        }

        return json_data, headers

    def decode_audio_response(self, response_data: bytes,
                             content_type: str) -> Dict[str, Any]:
        """Decode JSON response"""
        response_json = json.loads(response_data.decode('utf-8'))

        return {
            'text': response_json.get('text', ''),
            'metadata': {
                'input_length': response_json.get('input_length'),
                'output_length': response_json.get('output_length'),
                'model': response_json.get('model'),
                'processing_time_ms': response_json.get('processing_time_ms'),
                'transcript': response_json.get('transcript', '')  # Audio transcription if available
            },
            'success': response_json.get('success', True)
        }

    def encode_text(self, text: str, metadata: Dict[str, Any] = None) -> Tuple[bytes, Dict[str, str]]:
        """Encode text as JSON"""
        payload = {
            "text": text
        }

        # Add metadata fields
        if metadata:
            for key, value in metadata.items():
                if key not in ['text']:
                    payload[key] = value

        json_data = json.dumps(payload).encode('utf-8')

        headers = {
            'Content-Type': self.content_type
        }

        return json_data, headers

    def decode_text_response(self, response_data: bytes,
                            content_type: str) -> Dict[str, Any]:
        """Decode JSON text response"""
        response_json = json.loads(response_data.decode('utf-8'))

        return {
            'text': response_json.get('text', ''),
            'metadata': {
                'input_length': response_json.get('input_length'),
                'output_length': response_json.get('output_length'),
                'model': response_json.get('model'),
                'processing_time_ms': response_json.get('processing_time_ms')
            },
            'success': response_json.get('success', True)
        }

    def estimate_size(self, audio_data: bytes = None, text: str = None) -> int:
        """Estimate JSON payload size (hex encoding is 2x size, NO base64)"""
        if audio_data:
            # Hex encoding: 2 chars per byte = 2x size
            hex_size = len(audio_data) * 2
            # JSON overhead (~100 bytes for structure)
            return hex_size + 100
        elif text:
            # JSON overhead + text
            return len(text.encode('utf-8')) + 100
        return 0
