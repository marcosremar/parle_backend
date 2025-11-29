"""
Neural Codec Module - Neural audio compression using EnCodec
"""

import time
import base64
import pickle
from typing import Dict, Any
import numpy as np
import torch
from loguru import logger

from src.modules.base_module import BaseModule


class NeuralCodecModule(BaseModule):
    """Neural Codec Module for audio compression/decompression using EnCodec"""
    
    def __init__(self):
        super().__init__("neural_codec")
        self._codec = None
        self._codec_loaded = False
        self._device = "cpu"
    
    async def _initialize(self) -> bool:
        """Initialize neural codec model"""
        try:
            logger.info("Loading EnCodec neural audio codec...")
            
            # Detect device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device: {self._device}")
            
            # Import EnCodec
            try:
                from encodec import EncodecModel
            except ImportError as e:
                logger.error(f"EnCodec not installed. Install with: pip install encodec. Error: {e}")
                return False
            
            # Load EnCodec model for 24kHz @ 12kbps
            self._codec = EncodecModel.encodec_model_24khz()
            self._codec.set_target_bandwidth(12.0)  # 12 kbps
            self._codec.to(self._device)
            
            self._codec_loaded = True
            logger.info(f"✅ EnCodec loaded successfully on {self._device} (Sample Rate: 24kHz, Bitrate: 12kbps)")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to load codec: {e}")
            return False
    
    async def encode(
        self,
        audio_data: str,
        sample_rate: int = 24000,
        codec: str = "encodec"
    ) -> Dict[str, Any]:
        """
        Encode (compress) audio using neural codec
        
        Args:
            audio_data: Base64 encoded PCM audio data
            sample_rate: Audio sample rate in Hz (default: 24000)
            codec: Codec to use (default: "encodec")
            
        Returns:
            Dict with encoded_data, original_size, compressed_size, compression_ratio, latency_ms, codec
        """
        if not self.initialized:
            await self.initialize()
        
        if not self._codec_loaded:
            raise RuntimeError("Codec not available")
        
        start_time = time.time()
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            original_size = len(audio_bytes)
            
            # Convert bytes to numpy array (assuming int16 PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 normalized [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Convert to torch tensor [batch, channels, samples]
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).unsqueeze(0).to(self._device)
            
            # Encode using EnCodec
            with torch.no_grad():
                encoded_frames = self._codec.encode(audio_tensor)
            
            # Serialize encoded frames
            encoded_bytes = pickle.dumps(encoded_frames)
            compressed_size = len(encoded_bytes)
            
            # Encode to base64
            encoded_b64 = base64.b64encode(encoded_bytes).decode("utf-8")
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            logger.debug(f"Encoded audio: {original_size}B → {compressed_size}B ({compression_ratio:.2f}x compression, {latency_ms:.2f}ms)")
            
            return {
                "encoded_data": encoded_b64,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "latency_ms": latency_ms,
                "codec": codec
            }
        
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    async def decode(
        self,
        encoded_data: str,
        codec: str = "encodec"
    ) -> Dict[str, Any]:
        """
        Decode (decompress) audio using neural codec
        
        Args:
            encoded_data: Base64 encoded compressed audio
            codec: Codec to use (default: "encodec")
            
        Returns:
            Dict with audio_data, sample_rate, audio_size, latency_ms, codec
        """
        if not self.initialized:
            await self.initialize()
        
        if not self._codec_loaded:
            raise RuntimeError("Codec not available")
        
        start_time = time.time()
        
        try:
            # Decode base64
            encoded_bytes = base64.b64decode(encoded_data)
            
            # Deserialize encoded frames
            encoded_frames = pickle.loads(encoded_bytes)
            
            # Decode using EnCodec
            with torch.no_grad():
                decoded = self._codec.decode(encoded_frames)
            
            # Convert to numpy and denormalize
            audio_float = decoded.squeeze(0).squeeze(0).cpu().numpy()
            audio_int16 = (audio_float * 32768.0).astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_int16.tobytes()
            audio_size = len(audio_bytes)
            
            # Encode to base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            
            logger.debug(f"Decoded audio: {audio_size}B ({latency_ms:.2f}ms)")
            
            return {
                "audio_data": audio_b64,
                "sample_rate": 24000,
                "audio_size": audio_size,
                "latency_ms": latency_ms,
                "codec": codec
            }
        
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise
    
    async def health(self) -> Dict[str, Any]:
        """Health check"""
        return {
            "status": "healthy" if self._codec_loaded else "unhealthy",
            "codec_available": self._codec_loaded,
            "device": self._device
        }
    
    async def info(self) -> Dict[str, Any]:
        """Get codec information"""
        latency_ms = 6.0 if self._device == "cuda" else 10.0
        return {
            "codec": "encodec",
            "sample_rate": 24000,
            "bitrate": 12.0,
            "latency_ms": latency_ms,
            "compression_ratio": 8.0,
            "device": self._device,
            "streaming_enabled": True
        }
    
    async def _cleanup(self):
        """Cleanup resources"""
        if self._codec is not None:
            del self._codec
            self._codec = None
        self._codec_loaded = False
        logger.info("Neural codec resources cleaned up")
