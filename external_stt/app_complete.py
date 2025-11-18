"""
External STT Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, APIRouter
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import logging
import base64
import tempfile
import asyncio
import aiohttp
import aiofiles
from loguru import logger

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import src modules (fallback to local if not available)
try:
    from src.core.route_helpers import add_standard_endpoints
    from src.core.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_standard_endpoints(router):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "external_stt",
        "port": 8099,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "external_stt": {
        "provider": "groq",
        "model": "whisper-large-v3",
        "default_language": "pt",
        "timeout_seconds": 30,
        "max_retries": 3
    }
}

def get_config():
    """Get external stt service configuration"""
    config = DEFAULT_CONFIG.copy()
    return config

# ============================================================================
# Pydantic Models (Standalone)
# ============================================================================

class TranscribeRequest(BaseModel):
    """Base transcription request"""
    audio_data: Optional[str] = Field(default=None, description="Base64 encoded audio data (preferred)")
    audio_url: Optional[str] = Field(default=None, description="URL to audio file")
    language: Optional[str] = Field(default="pt", description="Language code (pt, en, etc.)")
    model: Optional[str] = Field(default="whisper-large-v3", description="Whisper model to use")
    response_format: Optional[str] = Field(default="text", description="Response format (text/json/verbose_json)")

class TranscriptionResponse(BaseModel):
    """Transcription response"""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    model: str
    provider: str

class AudioInfo(BaseModel):
    """Audio file information"""
    duration: float
    sample_rate: int
    channels: int
    format: str

# ============================================================================
# Groq Transcription Provider (Standalone)
# ============================================================================

class GroqTranscriptionProvider:
    """Groq Whisper API transcription provider"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not provided")

        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self.timeout = 30
        self.max_retries = 3

    async def transcribe_audio(self, audio_data: bytes, language: str = "pt", model: str = "whisper-large-v3") -> Dict[str, Any]:
        """Transcribe audio using Groq Whisper API"""
        temp_file_path = None
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Prepare multipart form data
            form_data = aiohttp.FormData()
            with open(temp_file_path, 'rb') as audio_file:
                audio_content = audio_file.read()
                form_data.add_field('file', audio_content, filename='audio.wav')
            form_data.add_field('model', model)
            form_data.add_field('language', language)
            form_data.add_field('response_format', 'json')

            # Make API request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                for attempt in range(self.max_retries):
                    try:
                        async with session.post(
                            f"{self.base_url}/audio/transcriptions",
                            data=form_data,
                            headers=self.headers
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                return {
                                    "text": result.get("text", ""),
                                    "language": language,
                                    "model": model,
                                    "provider": "groq"
                                }
                            else:
                                error_text = await response.text()
                                if attempt < self.max_retries - 1:
                                    print(f"Attempt {attempt + 1} failed: {response.status} - {error_text}")
                                    await asyncio.sleep(1)
                                    continue
                                else:
                                    raise HTTPException(
                                        status_code=response.status,
                                        detail=f"Groq API error: {error_text}"
                                    )
                    except aiohttp.ClientError as e:
                        if attempt < self.max_retries - 1:
                            print(f"Attempt {attempt + 1} failed: {str(e)}")
                            await asyncio.sleep(1)
                            continue
                        else:
                            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

# ============================================================================
# Global Provider Instance
# ============================================================================

try:
    transcription_provider = GroqTranscriptionProvider()
    provider_available = True
    print("✅ Groq transcription provider initialized")
except Exception as e:
    print(f"⚠️  Groq transcription provider failed: {e}")
    transcription_provider = None
    provider_available = False

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="External STT Service", version="1.0.0")

# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if provider_available else "degraded",
        "service": "external_stt",
        "timestamp": datetime.now().isoformat(),
        "transcription_provider": {
            "available": provider_available,
            "provider": "groq" if provider_available else None
        }
    }

@app.post("/transcribe")
async def transcribe_audio(request: TranscribeRequest):
    """Transcribe audio data"""
    if not provider_available:
        raise HTTPException(status_code=503, detail="Transcription provider not available")

    if not request.audio_data:
        raise HTTPException(status_code=400, detail="audio_data is required")

    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(request.audio_data)

        # Transcribe
        result = await transcription_provider.transcribe_audio(
            audio_data=audio_bytes,
            language=request.language or "pt",
            model=request.model or "whisper-large-v3"
        )

        return TranscriptionResponse(**result)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-file")
async def transcribe_audio_file(
    file: UploadFile = File(...),
    language: str = Form("pt"),
    model: str = Form("whisper-large-v3")
):
    """Transcribe uploaded audio file"""
    if not provider_available:
        raise HTTPException(status_code=503, detail="Transcription provider not available")

    try:
        # Read file content
        audio_bytes = await file.read()

        # Transcribe
        result = await transcription_provider.transcribe_audio(
            audio_data=audio_bytes,
            language=language,
            model=model
        )

        return TranscriptionResponse(**result)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/models")
async def get_models():
    """Get available transcription models"""
    if not provider_available:
        raise HTTPException(status_code=503, detail="Transcription provider not available")

    return {
        "models": [
            {
                "id": "whisper-large-v3",
                "provider": "groq",
                "description": "Latest Whisper large model v3"
            }
        ]
    }

@app.post("/audio-info")
async def get_audio_info(request: TranscribeRequest):
    """Get information about audio data without transcribing"""
    if not request.audio_data:
        raise HTTPException(status_code=400, detail="audio_data is required")

    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(request.audio_data)

        # Basic WAV header parsing (simplified)
        if len(audio_bytes) < 44:
            raise HTTPException(status_code=400, detail="Invalid audio data")

        # Parse WAV header (little endian)
        channels = int.from_bytes(audio_bytes[22:24], byteorder='little')
        sample_rate = int.from_bytes(audio_bytes[24:28], byteorder='little')
        bits_per_sample = int.from_bytes(audio_bytes[34:36], byteorder='little')

        # Calculate duration (approximate)
        data_size = len(audio_bytes) - 44  # Remove header
        duration = data_size / (sample_rate * channels * bits_per_sample / 8)

        return AudioInfo(
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            format="wav"
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

# Add standard endpoints
router = APIRouter()
add_standard_endpoints(router)
app.include_router(router)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service"""
    print("🚀 Initializing External STT Service...")
    print(f"   Provider Available: {provider_available}")
    if provider_available:
        print("   Models: whisper-large-v3 (Groq)")
        print("   Supported languages: pt, en, es, fr, etc.")
    print("✅ External STT Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8099"))
    print(f"Starting External STT Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
