"""
TTS (Text-to-Speech) endpoints
"""

from fastapi import APIRouter, HTTPException
import base64
import numpy as np
import io
from gtts import gTTS
from pydub import AudioSegment
import sys
import os
from typing import Optional

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schemas.audio import TTSRequest, TTSResponse, TTSEngine

router = APIRouter(prefix="/api/tts", tags=["tts"])

# Global Communication Manager (initialized from API Gateway service)
comm_manager: Optional['ServiceCommunicationManager'] = None

def set_comm_manager(cm):
    """Set Communication Manager instance from parent service"""
    global comm_manager
    comm_manager = cm


async def call_tts_service(text: str, voice: str = "af_bella"):
    """Call the TTS service via Communication Manager"""
    if not comm_manager:
        raise HTTPException(status_code=503, detail="Communication Manager not initialized")

    result = await comm_manager.call_text_service(
        service_name="tts",
        text=text,
        endpoint="/synthesize",
        extra_params={"text": text, "voice": voice}
    )
    return result


@router.post("/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest) -> TTSResponse:
    """
    Generate audio from text using specified TTS engine

    Engines:
    - gtts: Google Text-to-Speech (supports multiple languages)
    """
    try:
        audio_bytes = None
        sample_rate = 16000

        if request.engine == TTSEngine.GTTS:
            # Generate with gTTS
            audio_bytes = generate_with_gtts(
                text=request.text,
                language=request.voice,  # For gTTS, voice is language code
                speed=request.speed
            )

        else:
            # Unsupported engine
            raise ValueError(f"Unsupported TTS engine: {request.engine}")

        if audio_bytes is None:
            raise ValueError("Failed to generate audio")

        # Calculate duration
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        duration_seconds = len(audio_array) / sample_rate

        # Encode to base64 if requested
        audio_output = None
        if request.format == "base64":
            audio_output = base64.b64encode(audio_bytes).decode('utf-8')
        else:
            audio_output = audio_bytes.hex()  # Convert bytes to hex string for JSON

        return TTSResponse(
            success=True,
            audio=audio_output,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate
        )

    except Exception as e:
        return TTSResponse(
            success=False,
            error=str(e)
        )


def generate_with_gtts(text: str, language: str = "pt-br", speed: float = 1.0) -> bytes:
    """
    Generate audio using gTTS

    Args:
        text: Text to synthesize
        language: Language code (pt-br, en, es, etc)
        speed: Speed multiplier (not directly supported by gTTS, but we adjust with slow parameter)

    Returns:
        Audio bytes in int16 format at 16kHz
    """
    try:
        # gTTS only supports slow=True/False, so we approximate
        slow = speed < 0.9

        # Generate with gTTS
        tts = gTTS(text=text, lang=language, slow=slow)

        # Save to memory buffer
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)

        # Convert MP3 to WAV using pydub
        audio_segment = AudioSegment.from_mp3(mp3_buffer)

        # Convert to 16kHz mono
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_segment = audio_segment.set_channels(1)

        # Adjust speed if needed (pydub speedup)
        if speed != 1.0 and speed > 0:
            # pydub speed change (inverse of speed factor)
            playback_speed = 1.0 / speed
            audio_segment = audio_segment.speedup(playback_speed=playback_speed)

        # Convert to int16
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

        return samples.tobytes()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gTTS error: {str(e)}")


@router.get("/voices/list")
async def list_voices():
    """
    List all available voices for each TTS engine
    """
    return {
        "gtts": {
            "description": "Google Text-to-Speech - Language codes",
            "voices": [
                {"id": "pt-br", "name": "Portuguese (Brazil)", "lang": "Portuguese", "gender": "Neutral"},
                {"id": "pt", "name": "Portuguese", "lang": "Portuguese", "gender": "Neutral"},
                {"id": "en", "name": "English", "lang": "English", "gender": "Neutral"},
                {"id": "es", "name": "Spanish", "lang": "Spanish", "gender": "Neutral"},
                {"id": "fr", "name": "French", "lang": "French", "gender": "Neutral"},
                {"id": "de", "name": "German", "lang": "German", "gender": "Neutral"},
                {"id": "it", "name": "Italian", "lang": "Italian", "gender": "Neutral"},
                {"id": "ja", "name": "Japanese", "lang": "Japanese", "gender": "Neutral"},
                {"id": "ko", "name": "Korean", "lang": "Korean", "gender": "Neutral"},
                {"id": "zh", "name": "Chinese", "lang": "Chinese", "gender": "Neutral"}
            ]
        },
        "elevenlabs": {
            "description": "Eleven Labs TTS - High quality multilingual voices",
            "model": "eleven_turbo_v2_5",
            "total_voices": 54,
            "languages": 8,
            "voices": [
                # American English (20 voices)
                {"id": "af_heart", "name": "Heart", "lang": "American English", "gender": "Female"},
                {"id": "af_alloy", "name": "Alloy", "lang": "American English", "gender": "Female"},
                {"id": "af_aoede", "name": "Aoede", "lang": "American English", "gender": "Female"},
                {"id": "af_bella", "name": "Bella", "lang": "American English", "gender": "Female"},
                {"id": "af_jessica", "name": "Jessica", "lang": "American English", "gender": "Female"},
                {"id": "af_kore", "name": "Kore", "lang": "American English", "gender": "Female"},
                {"id": "af_nicole", "name": "Nicole", "lang": "American English", "gender": "Female"},
                {"id": "af_nova", "name": "Nova", "lang": "American English", "gender": "Female"},
                {"id": "af_river", "name": "River", "lang": "American English", "gender": "Female"},
                {"id": "af_sarah", "name": "Sarah", "lang": "American English", "gender": "Female"},
                {"id": "af_sky", "name": "Sky", "lang": "American English", "gender": "Female"},
                {"id": "am_adam", "name": "Adam", "lang": "American English", "gender": "Male"},
                {"id": "am_echo", "name": "Echo", "lang": "American English", "gender": "Male"},
                {"id": "am_eric", "name": "Eric", "lang": "American English", "gender": "Male"},
                {"id": "am_fenrir", "name": "Fenrir", "lang": "American English", "gender": "Male"},
                {"id": "am_liam", "name": "Liam", "lang": "American English", "gender": "Male"},
                {"id": "am_michael", "name": "Michael", "lang": "American English", "gender": "Male"},
                {"id": "am_onyx", "name": "Onyx", "lang": "American English", "gender": "Male"},
                {"id": "am_puck", "name": "Puck", "lang": "American English", "gender": "Male"},
                {"id": "am_santa", "name": "Santa", "lang": "American English", "gender": "Male"},

                # British English (8 voices)
                {"id": "bf_alice", "name": "Alice", "lang": "British English", "gender": "Female"},
                {"id": "bf_emma", "name": "Emma", "lang": "British English", "gender": "Female"},
                {"id": "bf_isabella", "name": "Isabella", "lang": "British English", "gender": "Female"},
                {"id": "bf_lily", "name": "Lily", "lang": "British English", "gender": "Female"},
                {"id": "bm_daniel", "name": "Daniel", "lang": "British English", "gender": "Male"},
                {"id": "bm_fable", "name": "Fable", "lang": "British English", "gender": "Male"},
                {"id": "bm_george", "name": "George", "lang": "British English", "gender": "Male"},
                {"id": "bm_lewis", "name": "Lewis", "lang": "British English", "gender": "Male"},

                # Spanish (3 voices)
                {"id": "ef_dora", "name": "Dora", "lang": "Spanish", "gender": "Female"},
                {"id": "em_alex", "name": "Alex", "lang": "Spanish", "gender": "Male"},
                {"id": "em_santa", "name": "Santa", "lang": "Spanish", "gender": "Male"},

                # French (1 voice)
                {"id": "ff_siwis", "name": "Siwis", "lang": "French", "gender": "Female"},

                # Brazilian Portuguese (2 voices)
                {"id": "pm_alex", "name": "Alex", "lang": "Brazilian Portuguese", "gender": "Male"},
                {"id": "pm_santa", "name": "Santa", "lang": "Brazilian Portuguese", "gender": "Male"},

                # Japanese (5 voices)
                {"id": "jf_alpha", "name": "Alpha", "lang": "Japanese", "gender": "Female"},
                {"id": "jf_gongitsune", "name": "Gongitsune", "lang": "Japanese", "gender": "Female"},
                {"id": "jf_nezumi", "name": "Nezumi", "lang": "Japanese", "gender": "Female"},
                {"id": "jf_tebukuro", "name": "Tebukuro", "lang": "Japanese", "gender": "Female"},
                {"id": "jm_kumo", "name": "Kumo", "lang": "Japanese", "gender": "Male"},

                # Mandarin Chinese (8 voices)
                {"id": "zf_xiaobei", "name": "Xiaobei", "lang": "Mandarin Chinese", "gender": "Female"},
                {"id": "zf_xiaoni", "name": "Xiaoni", "lang": "Mandarin Chinese", "gender": "Female"},
                {"id": "zf_xiaoxiao", "name": "Xiaoxiao", "lang": "Mandarin Chinese", "gender": "Female"},
                {"id": "zf_xiaoyi", "name": "Xiaoyi", "lang": "Mandarin Chinese", "gender": "Female"},
                {"id": "zm_yunjian", "name": "Yunjian", "lang": "Mandarin Chinese", "gender": "Male"},
                {"id": "zm_yunxi", "name": "Yunxi", "lang": "Mandarin Chinese", "gender": "Male"},
                {"id": "zm_yunxia", "name": "Yunxia", "lang": "Mandarin Chinese", "gender": "Male"},
                {"id": "zm_yunyang", "name": "Yunyang", "lang": "Mandarin Chinese", "gender": "Male"},

                # Hindi (4 voices)
                {"id": "hf_alpha", "name": "Alpha", "lang": "Hindi", "gender": "Female"},
                {"id": "hf_beta", "name": "Beta", "lang": "Hindi", "gender": "Female"},
                {"id": "hm_omega", "name": "Omega", "lang": "Hindi", "gender": "Male"},
                {"id": "hm_psi", "name": "Psi", "lang": "Hindi", "gender": "Male"},

                # Italian (2 voices)
                {"id": "if_sara", "name": "Sara", "lang": "Italian", "gender": "Female"},
                {"id": "im_nicola", "name": "Nicola", "lang": "Italian", "gender": "Male"}
            ]
        }
    }



@router.post("/validate")
async def validate_text(text: str) -> dict:
    """
    Validate text before TTS generation
    """
    issues = []

    # Check length
    if len(text) == 0:
        issues.append("Text is empty")
    elif len(text) > 5000:
        issues.append(f"Text too long: {len(text)} chars (max: 5000)")

    # Check for unsupported characters (basic check)
    if text and not text.isprintable():
        issues.append("Text contains non-printable characters")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "text_length": len(text),
        "estimated_duration_seconds": len(text) * 0.06  # Rough estimate
    }