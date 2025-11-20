"""
Audio Buffer Handler for WebRTC Server
Handles accumulation of audio chunks until audio_end signal
"""

import base64
import numpy as np
import logging

logger = logging.getLogger(__name__)

async def handle_audio_chunk(audio_b64, audio_buffer, session_id):
    """
    Add audio chunk to buffer

    Args:
        audio_b64: Base64 encoded audio data
        audio_buffer: List to accumulate audio chunks
        session_id: Session identifier for logging
    """
    try:
        # Decode base64
        audio_bytes = base64.b64decode(audio_b64)

        # Add to buffer
        audio_buffer.append(audio_bytes)

        logger.info(f"🎧 Audio chunk added to buffer: {len(audio_bytes)} bytes")
        logger.info(f"   📦 Buffer size: {len(audio_buffer)} chunks")
        logger.info(f"   👤 Session: {session_id}")

        return True

    except Exception as e:
        logger.error(f"❌ Error handling audio chunk: {e}")
        return False


def combine_audio_buffer(audio_buffer):
    """
    Combine all audio chunks in buffer into single array

    Args:
        audio_buffer: List of audio byte chunks

    Returns:
        numpy array of combined audio or None if error
    """
    try:
        if not audio_buffer:
            logger.warning("⚠️ Empty audio buffer")
            return None

        # Combine all chunks
        combined_bytes = b''.join(audio_buffer)

        # Convert to numpy array
        audio_int16 = np.frombuffer(combined_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        duration_sec = len(audio_float32) / 16000

        logger.info(f"🎵 Combined audio buffer:")
        logger.info(f"   📊 Total size: {len(combined_bytes)} bytes")
        logger.info(f"   🎤 Samples: {len(audio_float32)}")
        logger.info(f"   ⏱️ Duration: {duration_sec:.2f}s")

        return audio_float32

    except Exception as e:
        logger.error(f"❌ Error combining audio buffer: {e}")
        return None


def clear_audio_buffer(audio_buffer):
    """Clear the audio buffer"""
    audio_buffer.clear()
    logger.info("🧹 Audio buffer cleared")