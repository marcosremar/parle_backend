#!/usr/bin/env python3
"""
Audio Format Converter
Converts WAV audio to various compressed formats (MP3, Opus, OGG)
Using pydub (which wraps ffmpeg) for high-quality format conversion
"""

import io
import logging
from typing import Optional
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioFormatConverter:
    """
    Converts audio between different formats

    Supported formats:
    - WAV: Uncompressed, largest size, no conversion overhead
    - MP3: Widely compatible, ~90% smaller than WAV
    - Opus: Best quality/size ratio, ~90% smaller, ideal for streaming
    - OGG: Better than MP3, ~90% smaller
    """

    # Default bitrates for each format (in kbps)
    DEFAULT_BITRATES = {
        "mp3": "128k",      # Standard quality, universal compatibility
        "opus": "64k",      # High quality at low bitrate (Opus excels here)
        "ogg": "96k",       # Good balance
    }

    # Recommended bitrates for different quality levels
    BITRATE_PRESETS = {
        "low": {"mp3": "96k", "opus": "32k", "ogg": "64k"},
        "medium": {"mp3": "128k", "opus": "64k", "ogg": "96k"},
        "high": {"mp3": "192k", "opus": "96k", "ogg": "128k"},
        "very_high": {"mp3": "256k", "opus": "128k", "ogg": "160k"},
    }

    @staticmethod
    def convert_wav_to_format(
        wav_bytes: bytes,
        output_format: str,
        bitrate: Optional[str] = None,
        quality_preset: Optional[str] = None
    ) -> bytes:
        """
        Convert WAV audio to specified format

        Args:
            wav_bytes: Input WAV audio as bytes
            output_format: Target format ("mp3", "opus", "ogg", "wav")
            bitrate: Bitrate (e.g., "128k", "64k"). If None, uses default for format.
            quality_preset: Quality preset ("low", "medium", "high", "very_high").
                          Overrides bitrate if specified.

        Returns:
            Audio bytes in target format

        Raises:
            ValueError: If format is unsupported
            RuntimeError: If conversion fails
        """
        # Validate format
        output_format = output_format.lower()
        if output_format not in ["wav", "mp3", "opus", "ogg"]:
            raise ValueError(f"Unsupported format: {output_format}. Must be wav, mp3, opus, or ogg")

        # If WAV, return as-is
        if output_format == "wav":
            logger.debug("Output format is WAV - no conversion needed")
            return wav_bytes

        # Determine bitrate
        if quality_preset:
            if quality_preset not in AudioFormatConverter.BITRATE_PRESETS:
                raise ValueError(f"Invalid quality preset: {quality_preset}")
            bitrate = AudioFormatConverter.BITRATE_PRESETS[quality_preset][output_format]
            logger.debug(f"Using quality preset '{quality_preset}': {bitrate}")
        elif not bitrate:
            bitrate = AudioFormatConverter.DEFAULT_BITRATES[output_format]
            logger.debug(f"Using default bitrate for {output_format}: {bitrate}")

        try:
            # Load WAV from bytes
            audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))

            # Export to target format
            output_buffer = io.BytesIO()

            # Format-specific export parameters
            export_params = {
                "format": output_format,
                "bitrate": bitrate
            }

            # Opus and OGG use codec parameter
            if output_format == "opus":
                export_params["codec"] = "libopus"
            elif output_format == "ogg":
                export_params["codec"] = "libvorbis"
                # Use quality parameter instead of bitrate for vorbis (more reliable)
                export_params.pop("bitrate", None)
                export_params["parameters"] = ["-q:a", "4"]  # Quality 4 ≈ 128kbps

            audio.export(output_buffer, **export_params)

            # Get bytes
            converted_bytes = output_buffer.getvalue()

            # Log conversion stats
            original_size = len(wav_bytes)
            converted_size = len(converted_bytes)
            compression_ratio = (1 - converted_size / original_size) * 100

            logger.info(
                f"Converted WAV → {output_format.upper()}: "
                f"{original_size:,} → {converted_size:,} bytes "
                f"({compression_ratio:.1f}% reduction, {bitrate})"
            )

            return converted_bytes

        except Exception as e:
            logger.error(f"Failed to convert audio to {output_format}: {e}")
            raise RuntimeError(f"Audio conversion failed: {e}")

    @staticmethod
    def get_media_type(format_name: str) -> str:
        """
        Get MIME type for audio format

        Args:
            format_name: Audio format ("wav", "mp3", "opus", "ogg")

        Returns:
            MIME type string
        """
        media_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "ogg": "audio/ogg"
        }
        return media_types.get(format_name.lower(), "application/octet-stream")

    @staticmethod
    def estimate_size_reduction(original_wav_size: int, target_format: str, bitrate: Optional[str] = None) -> dict:
        """
        Estimate the size reduction for converting to a format

        Args:
            original_wav_size: Size of WAV file in bytes
            target_format: Target format
            bitrate: Target bitrate (optional)

        Returns:
            Dict with estimated_size and reduction_percent
        """
        # Rough compression ratios (typical)
        compression_ratios = {
            "wav": 1.0,
            "mp3": 0.1,     # ~90% reduction
            "opus": 0.08,   # ~92% reduction (better than MP3)
            "ogg": 0.1,     # ~90% reduction
        }

        ratio = compression_ratios.get(target_format, 0.1)
        estimated_size = int(original_wav_size * ratio)
        reduction_percent = (1 - ratio) * 100

        return {
            "estimated_size": estimated_size,
            "reduction_percent": reduction_percent,
            "original_size": original_wav_size
        }


# Convenience function for quick conversion
def convert_audio(wav_bytes: bytes, format: str = "mp3", bitrate: Optional[str] = None) -> bytes:
    """
    Quick conversion function

    Args:
        wav_bytes: WAV audio bytes
        format: Target format
        bitrate: Optional bitrate override

    Returns:
        Converted audio bytes
    """
    return AudioFormatConverter.convert_wav_to_format(wav_bytes, format, bitrate)


if __name__ == "__main__":
    # Example usage and testing
    import wave
    import numpy as np

    print("Audio Format Converter - Test")
    print("=" * 60)

    # Create a test WAV file (1 second of sine wave at 24kHz)
    sample_rate = 24000
    duration = 1.0
    frequency = 440  # A4 note

    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)

    # Convert to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Create WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_bytes = wav_buffer.getvalue()
    original_size = len(wav_bytes)

    print(f"Original WAV: {original_size:,} bytes")
    print()

    # Test each format
    formats = ["mp3", "opus", "ogg"]

    for fmt in formats:
        try:
            converted = AudioFormatConverter.convert_wav_to_format(wav_bytes, fmt)
            reduction = (1 - len(converted) / original_size) * 100

            print(f"{fmt.upper():6} → {len(converted):,} bytes ({reduction:.1f}% smaller)")
        except Exception as e:
            print(f"{fmt.upper():6} → Failed: {e}")

    print()
    print("Test complete!")
