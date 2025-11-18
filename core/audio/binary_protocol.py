#!/usr/bin/env python3
"""
Binary Protocol for Optimized Audio Transmission
Enhanced protocol for low-latency audio communication between services
"""

import json
import struct
from typing import Dict, Any, Tuple


class BinaryProtocol:
    """
    Binary protocol for optimized audio transmission
    Based on frontend improvements for better performance
    """
    MAGIC = 0xAD10  # "AUDIO" in hex

    # Message types
    TYPE_AUDIO_REQUEST = 0x01
    TYPE_AUDIO_RESPONSE = 0x02
    TYPE_TEXT_REQUEST = 0x03
    TYPE_TEXT_RESPONSE = 0x04
    TYPE_ERROR = 0xFF

    @staticmethod
    def pack_audio_message(audio_data: bytes, metadata: Dict[str, Any], msg_type: int = 0x01) -> bytes:
        """Pack audio and metadata into binary format"""
        # Serialize metadata
        metadata_json = json.dumps(metadata).encode('utf-8')

        # Create header (big-endian: magic 2B, type 1B, metadata_size 2B, audio_size 4B)
        header = struct.pack(
            '>HBHI',
            BinaryProtocol.MAGIC,
            msg_type,
            len(metadata_json),
            len(audio_data)
        )

        return header + metadata_json + audio_data

    @staticmethod
    def unpack_message(data: bytes) -> Tuple[int, Dict[str, Any], bytes]:
        """Unpack binary message"""
        if len(data) < 9:  # Minimum header size
            raise ValueError("Message too small")

        # Unpack header
        magic, msg_type, meta_size, audio_size = struct.unpack('>HBHI', data[:9])

        if magic != BinaryProtocol.MAGIC:
            raise ValueError(f"Invalid magic number: {magic:04X}")

        # Extract metadata and audio
        offset = 9
        metadata_json = data[offset:offset + meta_size]
        metadata = json.loads(metadata_json.decode('utf-8'))

        offset += meta_size
        audio_data = data[offset:offset + audio_size]

        return msg_type, metadata, audio_data

    @staticmethod
    def pack_text_message(text: str, metadata: Dict[str, Any], msg_type: int = 0x03) -> bytes:
        """Pack text message into binary format"""
        text_data = text.encode('utf-8')
        return BinaryProtocol.pack_audio_message(text_data, metadata, msg_type)

    @staticmethod
    def pack_error_message(error_msg: str, error_code: int = 500) -> bytes:
        """Pack error message"""
        metadata = {
            'error_code': error_code,
            'timestamp': str(__import__('time').time())
        }
        error_data = error_msg.encode('utf-8')
        return BinaryProtocol.pack_audio_message(error_data, metadata, BinaryProtocol.TYPE_ERROR)

    @staticmethod
    def pack_audio_request(audio_data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Pack audio request for HTTP transmission"""
        return BinaryProtocol.pack_audio_message(audio_data, metadata, BinaryProtocol.TYPE_AUDIO_REQUEST)