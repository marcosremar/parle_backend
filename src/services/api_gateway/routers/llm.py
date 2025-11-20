"""
LLM (Large Language Model) and TTS endpoints for SDK testing
Proxy to services via Communication Manager
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import logging
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["llm"])

# Global Communication Manager (initialized from API Gateway service)
comm_manager: Optional['ServiceCommunicationManager'] = None

def set_comm_manager(cm):
    """Set Communication Manager instance from parent service"""
    global comm_manager
    comm_manager = cm


@router.post("/chat")
async def llm_chat(request: Request):
    """
    Generate text response using LLM via Communication Manager

    Request body:
    {
        "messages": [{"role": "user", "content": "..."}],
        "session_id": "test-123",
        "use_external": false  // optional, defaults to local LLM
    }

    Response:
    {
        "response": "Generated text",
        "session_id": "test-123"
    }
    """
    if not comm_manager:
        raise HTTPException(status_code=503, detail="Communication Manager not initialized")

    try:
        body = await request.json()

        # Extract messages and convert to prompt
        messages = body.get("messages", [])
        prompt = messages[-1].get("content", "") if messages else ""

        # Call llm service via Communication Manager
        result = await comm_manager.call_text_service(
            service_name="llm",
            text=prompt,
            endpoint="/chat",
            extra_params=body
        )

        # Extract text from response
        response_text = result.get("text", result.get("choices", [{}])[0].get("message", {}).get("content", ""))

        return {
            "response": response_text,
            "session_id": body.get("session_id", "")
        }

    except Exception as e:
        logger.error(f"LLM proxy error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM proxy error: {str(e)}")


@router.get("/health")
async def llm_health():
    """
    Check if LLM service is available via Communication Manager
    """
    if not comm_manager:
        return {"status": "unavailable", "detail": "Communication Manager not initialized"}

    try:
        # Test actual LLM request via Communication Manager
        result = await comm_manager.call_text_service(
            service_name="llm",
            text="test",
            endpoint="/chat",
            extra_params={"messages": [{"role": "user", "content": "test"}]}
        )

        return {"status": "healthy", "service": "llm"}

    except Exception as e:
        logger.error(f"LLM health check error: {e}")
        return {"status": "error", "detail": str(e)}


@router.post("/tts/synthesize")
async def tts_synthesize(request: Request):
    """
    Generate speech audio from text via TTS service through Communication Manager

    Request body:
    {
        "text": "Hello world",
        "voice_id": None,
        "speed": 1.0,
        "sample_rate": 24000,
        "format": "wav"
    }

    Returns: Binary audio data (WAV format)
    """
    if not comm_manager:
        raise HTTPException(status_code=503, detail="Communication Manager not initialized")

    try:
        body = await request.json()
        text = body.get("text", "")

        # Call TTS service via Communication Manager
        result = await comm_manager.call_text_service(
            service_name="tts",
            text=text,
            endpoint="/synthesize",
            extra_params=body
        )

        # Communication Manager returns the audio data directly for TTS
        # Return binary audio
        return Response(
            content=result if isinstance(result, bytes) else result.encode(),
            media_type="audio/wav"
        )

    except Exception as e:
        logger.error(f"TTS proxy error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS proxy error: {str(e)}")
