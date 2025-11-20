"""
Conversation endpoint - Routes conversation requests to Orchestrator via OrchestratorClient
Includes automatic STT validation when audio is generated
Includes SSE streaming endpoint with adaptive parameters
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError, BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
import asyncio
import json

from src.core.orchestrator_client import OrchestratorClient, OrchestratorClientError
from src.core.exceptions import UltravoxError, wrap_exception

router = APIRouter(prefix="/api", tags=["conversation"])
logger = logging.getLogger(__name__)

# Global orchestrator client (initialized on startup via process.py)
orchestrator_client: Optional[OrchestratorClient] = None

# Global Communication Manager (initialized from API Gateway service)
comm_manager: Optional['ServiceCommunicationManager'] = None

def set_comm_manager(cm):
    """Set Communication Manager instance from parent service"""
    global comm_manager
    comm_manager = cm


class ConversationMessage(BaseModel):
    """Conversation message model"""
    role: str
    content: str
    timestamp: Optional[str] = None


class ConversationRequest(BaseModel):
    """Request model for conversation endpoint"""
    session_id: str = Field(..., description="Session ID for conversation context")
    message: str = Field(..., description="User message text")
    voice_id: Optional[str] = Field(default=None, description="Voice ID for TTS response (also triggers STT validation)")
    stt_mode: str = Field(default="local", description="STT service to use: 'local' (port 8200) or 'external' (port 8201)")


class ConversationResponse(BaseModel):
    """Response model for conversation endpoint"""
    success: bool = True
    response: str = Field(..., description="Assistant response text (original from LLM)")
    session_id: str = Field(..., description="Session ID")
    audio: Optional[str] = Field(default=None, description="Base64 encoded audio response (if voice_id provided)")
    transcription: Optional[str] = Field(default=None, description="STT transcription of audio response")
    transcription_match: Optional[bool] = Field(default=None, description="True if transcription matches original text")
    context_size: Optional[int] = Field(default=None, description="Number of messages in conversation context")
    messages_count: Optional[int] = Field(default=None, description="Total messages in conversation")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Processing metrics")
    error: Optional[str] = Field(default=None, description="Error message if success=false")


async def transcribe_audio(audio_base64: str, stt_service: str, language: str = "pt") -> Dict[str, Any]:
    """
    Transcribe audio using STT service via Communication Manager

    Args:
        audio_base64: Base64 encoded audio data
        stt_service: STT service name ("stt")
        language: Language code for transcription (default: "pt")

    Returns:
        Dict with:
            - text: Transcribed text
            - processing_time_ms: Time taken for transcription
            - error: Error message if failed
    """
    if not comm_manager:
        logger.error("Communication Manager not initialized")
        return {
            "text": "",
            "error": "Communication Manager not initialized",
            "processing_time_ms": 0
        }

    try:
        start_time = time.time()

        # Call STT service via Communication Manager
        result = await comm_manager.call_text_service(
            service_name=stt_service,
            text="",
            endpoint="/transcribe",
            extra_params={
                "audio_base64": audio_base64,
                "language": language
            }
        )

        processing_time = (time.time() - start_time) * 1000

        return {
            "text": result.get("text", ""),
            "processing_time_ms": processing_time,
            "error": None
        }

    except asyncio.TimeoutError:
        logger.error("STT transcription timed out")
        return {
            "text": "",
            "error": "STT transcription timed out",
            "processing_time_ms": 30000
        }
    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        return {
            "text": "",
            "error": f"STT error: {str(e)}",
            "processing_time_ms": 0
        }


async def initialize_orchestrator_client():
    """Initialize orchestrator client (called on app startup)"""
    global orchestrator_client
    if orchestrator_client is None:
        # Use same client from process router
        from . import process
        orchestrator_client = process.orchestrator_client
        if orchestrator_client:
            logger.info("✅ Conversation router using shared orchestrator client")


@router.post("/conversation", response_model=ConversationResponse)
async def handle_conversation(request: ConversationRequest) -> ConversationResponse:
    """
    Handle text-based conversation through the orchestrator

    This endpoint processes text messages and maintains conversation context.
    Optionally returns audio response if voice_id is provided.
    When audio is generated, automatically transcribes it using STT for validation.
    """
    global orchestrator_client

    # Ensure orchestrator client is initialized
    if not orchestrator_client:
        # Try to get from process router
        from . import process
        orchestrator_client = process.orchestrator_client

        if not orchestrator_client:
            return ConversationResponse(
                success=False,
                response="",
                session_id=request.session_id,
                error="Orchestrator client not initialized"
            )

    try:
        # Call orchestrator service for real AI conversation processing
        result = await orchestrator_client.process_text_conversation(
            message=request.message,
            session_id=request.session_id,
            voice_id=request.voice_id
        )

        transcription = None
        transcription_match = None

        # If audio was generated, transcribe it for validation
        if result.get("audio") and request.voice_id:
            # Determine STT service name based on mode
            stt_service = "stt"  # Always use stt service (renamed from external_stt)

            logger.info(f"Transcribing audio with {request.stt_mode} STT (service: {stt_service})")

            # Transcribe the generated audio
            stt_result = await transcribe_audio(
                audio_base64=result["audio"],
                stt_service=stt_service,
                language="pt"
            )

            if not stt_result.get("error"):
                transcription = stt_result["text"]

                # Compare transcription with original text (normalized)
                original = result.get("response", "").strip().lower()
                transcribed = transcription.strip().lower()
                transcription_match = (original == transcribed)

                logger.info(
                    f"STT validation: Match={transcription_match}, "
                    f"Original='{result.get('response', '')[:50]}...', "
                    f"Transcribed='{transcription[:50]}...'"
                )
            else:
                logger.error(f"STT transcription failed: {stt_result['error']}")

        # Map orchestrator response to API response format
        return ConversationResponse(
            success=result.get("success", True),
            response=result.get("response", ""),
            session_id=result.get("session_id", request.session_id),
            context_size=result.get("context_size"),
            messages_count=result.get("messages_count"),
            audio=result.get("audio"),
            transcription=transcription,
            transcription_match=transcription_match,
            metrics=result.get("metrics"),
            error=result.get("error")
        )

    except OrchestratorClientError as e:
        logger.error(f"Orchestrator error in conversation: {e}")
        return ConversationResponse(
            success=False,
            response="",
            session_id=request.session_id,
            error=f"Orchestrator error: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Unexpected error in conversation: {e}")
        return ConversationResponse(
            success=False,
            response="",
            session_id=request.session_id,
            error=f"Internal error: {str(e)}"
        )


# ============================================================================
# STREAMING ENDPOINT WITH ADAPTIVE PARAMETERS (Phase 3)
# ============================================================================


class StreamingConversationRequest(BaseModel):
    """Request model for streaming conversation endpoint"""
    session_id: str = Field(..., description="Session ID for conversation context")
    audio_data: Optional[bytes] = Field(None, description="Raw audio bytes (for streaming)")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio data (alternative to audio_data)")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    voice_id: Optional[str] = Field(None, description="Voice ID for TTS response")


@router.post("/conversation/stream", tags=["conversation"])
async def send_message_streaming(request: StreamingConversationRequest):
    """
    Streaming conversation endpoint with adaptive parameters.

    Uses Communication Manager for transparent protocol selection:
    - ZeroMQ (primary, 0.01ms latency, 410k msg/s)
    - gRPC (fallback 1, ~7ms latency)
    - HTTP Binary (fallback 2)
    - JSON (fallback 3)

    Returns Server-Sent Events (SSE) with:
    - text_chunk: LLM response text chunks
    - analysis: Conversation analysis (response_type, theme, tone, confidence)
    - adaptive_instructions: Instructions for next LLM response
    - error_correction: Error pattern detection and suggestions
    - complete: Final completion event with metrics

    Example client-side SSE handling:
    ```javascript
    const eventSource = new EventSource('/api/conversation/stream');
    eventSource.addEventListener('text_chunk', (e) => {
      const data = JSON.parse(e.data);
      console.log('Text:', data.data);
    });
    eventSource.addEventListener('analysis', (e) => {
      const analysis = JSON.parse(e.data);
      console.log('Theme:', analysis.data.theme);
      console.log('Tone:', analysis.data.tone);
    });
    ```
    """
    if not comm_manager:
        logger.error("Communication Manager not initialized for streaming")
        yield f"event: error\ndata: {json.dumps({'error': 'Communication Manager not initialized'})}\n\n"
        return

    try:
        # Convert base64 to bytes if needed
        audio_data = request.audio_data
        if not audio_data and request.audio_base64:
            import base64
            audio_data = base64.b64decode(request.audio_base64)

        if not audio_data:
            logger.error("No audio data provided for streaming")
            yield f"event: error\ndata: {json.dumps({'error': 'No audio data provided'})}\n\n"
            return

        logger.info(f"🎬 Starting streaming conversation via Communication Manager: session={request.session_id}")

        # Prepare payload for orchestrator (base64 encoded audio)
        import base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        payload = {
            "audio": audio_b64,
            "session_id": request.session_id,
            "sample_rate": request.sample_rate,
            "voice_id": request.voice_id or None
        }

        # Call orchestrator's streaming endpoint via Communication Manager
        # The Communication Manager transparently selects the optimal protocol
        # (ZeroMQ → gRPC → HTTP Binary → JSON)
        logger.info(f"📡 Calling orchestrator streaming endpoint via Communication Manager")

        async for event_dict in comm_manager.stream_json_events(
            service_name="orchestrator",
            action="conversation-stream",
            payload=payload
        ):
            # event_dict is already a parsed JSON object from Communication Manager
            event = event_dict

            # Format as SSE for browser
            event_type = event.get('event', 'unknown')
            event_json = json.dumps(event)

            yield f"event: {event_type}\n"
            yield f"data: {event_json}\n\n"

            logger.debug(f"📤 Streamed event: {event_type} (seq={event.get('sequence', '?')})")

        logger.info(f"✅ Streaming conversation complete: session={request.session_id}")

    except Exception as e:
        logger.error(f"❌ Streaming conversation error: {e}", exc_info=True)
        error_event = {
            "event": "error",
            "sequence": 0,
            "data": {"error": str(e)},
            "timestamp": time.time(),
            "is_final": True
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
