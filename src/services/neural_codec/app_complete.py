"""FastAPI app for Neural Codec Service - Complete version for Nomad"""
from fastapi import FastAPI, HTTPException
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
import sys
from pathlib import Path
import time
import base64
import numpy as np
import torch
import pickle
from datetime import datetime

# Import EnCodec at module level (opcional - será importado dentro da função se necessário)
ENCODEC_AVAILABLE = None  # Será determinado na primeira chamada
EncodecModel = None  # Será importado quando necessário

# Ajustar paths
current_dir = Path(__file__).parent
v2_root = current_dir.parent
project_root = v2_root.parent

# Adicionar paths ao sys.path
possible_paths = [
    str(project_root / "src"),
    str(project_root),
    "/Users/marcos/Downloads/temp/ultravox-pipeline/src",
    "/Users/marcos/Downloads/temp/ultravox-pipeline",
]

for path in possible_paths:
    if Path(path).exists():
        sys.path.insert(0, path)

# Importar modelos (versão simplificada se necessário)
try:
    from src.services.neural_codec.models import (
        EncodeRequest, EncodeResponse, DecodeRequest, DecodeResponse,
        HealthResponse, CodecInfoResponse
    )
except ImportError:
    # Criar modelos simplificados se não conseguir importar
    class EncodeRequest(BaseModel):
        audio_data: str = Field(..., description="Base64 encoded PCM audio data")
        sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
        codec: Literal["encodec"] = Field(default="encodec", description="Codec to use")
        
        @field_validator("audio_data")
        @classmethod
        def validate_audio_data(cls, v: str) -> str:
            try:
                base64.b64decode(v)
            except Exception as e:
                raise ValueError(f"Invalid base64 audio data: {e}")
            return v
    
    class EncodeResponse(BaseModel):
        encoded_data: str
        original_size: int
        compressed_size: int
        compression_ratio: float
        latency_ms: float
        codec: str
    
    class DecodeRequest(BaseModel):
        encoded_data: str = Field(..., description="Base64 encoded compressed audio")
        codec: Literal["encodec"] = Field(default="encodec", description="Codec to use")
    
    class DecodeResponse(BaseModel):
        audio_data: str
        sample_rate: int
        audio_size: int
        latency_ms: float
        codec: str
    
    class HealthResponse(BaseModel):
        status: str
        codec_available: bool
        device: str
    
    class CodecInfoResponse(BaseModel):
        codec: str
        sample_rate: int
        bitrate: float
        latency_ms: float
        compression_ratio: float
        device: str
        streaming_enabled: bool

# Create FastAPI app
app = FastAPI(
    title="Neural Codec Service",
    version="1.0.0",
    description="Neural audio compression service using EnCodec"
)

# Global state
_codec = None
_codec_loaded = False
_device = "cpu"

def _load_codec():
    """Load neural codec model (lazy loading)."""
    global _codec, _codec_loaded, _device
    
    if _codec_loaded:
        return
    
    try:
        print("Loading EnCodec neural audio codec...")
        
        # Detect device
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {_device}")
        
        # Import EnCodec (fazer import dentro da função)
        try:
            from encodec import EncodecModel  # Note: EncodecModel (c minúsculo), não EnCodecModel
        except ImportError as e:
            raise ImportError(f"EnCodec not installed. Install with: pip install encodec. Error: {e}")
        
        # Load EnCodec model for 24kHz @ 12kbps
        _codec = EncodecModel.encodec_model_24khz()
        _codec.set_target_bandwidth(12.0)  # 12 kbps
        _codec.to(_device)
        
        _codec_loaded = True
        print(f"✅ EnCodec loaded successfully on {_device} (Sample Rate: 24kHz, Bitrate: 12kbps)")
        
    
    except Exception as e:
        print(f"❌ Failed to load codec: {e}")
        raise

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "codec_available": _codec_loaded,
        "device": _device
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "neural-codec-service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/v1/info", response_model=CodecInfoResponse)
async def codec_info():
    """Get codec information"""
    return CodecInfoResponse(
        codec="encodec",
        sample_rate=24000,
        bitrate=12.0,
        latency_ms=6.0 if _device == "cuda" else 10.0,
        compression_ratio=8.0,
        device=_device,
        streaming_enabled=True
    )

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        codec_available=_codec_loaded,
        device=_device
    )

@app.post("/api/v1/encode", response_model=EncodeResponse)
async def encode_audio(request: EncodeRequest):
    """Encode (compress) audio using neural codec"""
    start_time = time.time()
    
    try:
        # Ensure codec is loaded
        _load_codec()
        
        if not _codec_loaded:
            raise HTTPException(status_code=503, detail="Codec not available")
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        original_size = len(audio_bytes)
        
        # Convert bytes to numpy array (assuming int16 PCM)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 normalized [-1, 1]
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Convert to torch tensor [batch, channels, samples]
        audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).unsqueeze(0).to(_device)
        
        # Encode using EnCodec
        with torch.no_grad():
            encoded_frames = _codec.encode(audio_tensor)
        
        # Serialize encoded frames
        encoded_bytes = pickle.dumps(encoded_frames)
        compressed_size = len(encoded_bytes)
        
        # Encode to base64
        encoded_b64 = base64.b64encode(encoded_bytes).decode("utf-8")
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        print(f"Encoded audio: {original_size}B → {compressed_size}B ({compression_ratio:.2f}x compression, {latency_ms:.2f}ms)")
        
        return EncodeResponse(
            encoded_data=encoded_b64,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            latency_ms=latency_ms,
            codec=request.codec
        )
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

@app.post("/api/v1/decode", response_model=DecodeResponse)
async def decode_audio(request: DecodeRequest):
    """Decode (decompress) audio using neural codec"""
    start_time = time.time()
    
    try:
        # Ensure codec is loaded
        _load_codec()
        
        if not _codec_loaded:
            raise HTTPException(status_code=503, detail="Codec not available")
        
        # Decode base64
        encoded_bytes = base64.b64decode(request.encoded_data)
        
        # Deserialize encoded frames
        encoded_frames = pickle.loads(encoded_bytes)
        
        # Decode using EnCodec
        with torch.no_grad():
            decoded = _codec.decode(encoded_frames)
        
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
        
        print(f"Decoded audio: {audio_size}B ({latency_ms:.2f}ms)")
        
        return DecodeResponse(
            audio_data=audio_b64,
            sample_rate=24000,
            audio_size=audio_size,
            latency_ms=latency_ms,
            codec=request.codec
        )
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"Decoding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Decoding failed: {e}")

@app.on_event("startup")
async def startup():
    """Initialize service and load neural codec at startup"""
    print("🚀 Initializing Neural Codec Service...")

    # Load neural codec at startup (not lazy)
    try:
        _load_codec()
        print("✅ Neural Codec Service initialized successfully!")
        print(f"   Codec loaded on device: {_device}")
        print(f"   Model ready for encoding/decoding")
    except Exception as e:
        print(f"❌ Failed to initialize Neural Codec Service: {e}")
        # Don't crash the service, just log the error
        # The codec will be loaded on first use if startup fails

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8106"))
    uvicorn.run(app, host="0.0.0.0", port=port)

