"""
Speech Router - STT, TTS, Neural Codec
Usa módulos internos para chamadas diretas Python
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

router = APIRouter()

# Module instances (lazy initialization)
_stt_module = None
_tts_module = None


def get_stt_module():
    """Get STT module instance"""
    global _stt_module
    if _stt_module is None:
        from src.modules import create
        _stt_module = create("stt")
    return _stt_module


def get_tts_module():
    """Get TTS module instance"""
    global _tts_module
    if _tts_module is None:
        from src.modules import create
        _tts_module = create("tts")
    return _tts_module


# Pydantic models
class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    language: str = Field("pt", description="Language code")
    model: str = Field("whisper-large-v3", description="Whisper model")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Voice ID")
    provider: Optional[str] = Field(None, description="TTS provider (gtts, elevenlabs)")
    language: str = Field("pt", description="Language code")
    speed: float = Field(1.0, description="Speech speed")


@router.get("/health")
async def health():
    """Health check for speech services"""
    return {"status": "ok", "services": ["stt", "tts"]}


@router.post("/stt/transcribe")
async def transcribe(request: TranscribeRequest):
    """STT transcription endpoint using direct module"""
    try:
        stt = get_stt_module()
        result = await stt.transcribe(
            audio_base64=request.audio_base64,
            audio_url=request.audio_url,
            language=request.language,
            model=request.model
        )
        return result
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stt/transcribe-file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("pt"),
    model: str = Form("whisper-large-v3")
):
    """Transcribe uploaded audio file"""
    try:
        import base64
        audio_data = await file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        stt = get_stt_module()
        result = await stt.transcribe(
            audio_base64=audio_base64,
            language=language,
            model=model
        )
        return result
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stt/models")
async def get_stt_models():
    """Get available STT models"""
    try:
        stt = get_stt_module()
        return await stt.get_models()
    except Exception as e:
        logger.error(f"Failed to get STT models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts/synthesize")
async def synthesize(request: SynthesizeRequest):
    """TTS synthesis endpoint using direct module"""
    try:
        tts = get_tts_module()
        result = await tts.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            provider=request.provider,
            language=request.language,
            speed=request.speed
        )
        return result
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tts/voices")
async def get_voices(provider: Optional[str] = None):
    """Get available TTS voices"""
    try:
        tts = get_tts_module()
        return await tts.get_voices(provider=provider)
    except Exception as e:
        logger.error(f"Failed to get TTS voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))
