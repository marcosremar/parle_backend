"""
Process endpoint - Routes requests to Orchestrator via ConversationController
"""

from fastapi import APIRouter, HTTPException, File, Form, UploadFile
import base64
import os
import time
from typing import Dict, Any, Optional
import logging

from schemas.audio import ProcessRequest, ProcessResponse
from src.core.orchestrator_client import OrchestratorClient, OrchestratorClientError
from src.core.controllers.conversation_controller import ConversationController

router = APIRouter(tags=["process"])
logger = logging.getLogger(__name__)

# Global orchestrator client and controller (initialized on startup)
orchestrator_client: Optional[OrchestratorClient] = None
conversation_controller: Optional[ConversationController] = None


async def initialize_orchestrator_client():
    """Initialize orchestrator client and controller (called on app startup)"""
    global orchestrator_client, conversation_controller

    if orchestrator_client is None:
        # Connect to orchestrator service via Service Manager (port 8900 by default)
        # Disable binary protocol until orchestrator is registered in Communication Manager
        orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8900")
        orchestrator_client = OrchestratorClient(
            orchestrator_url=orchestrator_url,
            use_binary_protocol=False
        )
        await orchestrator_client.initialize()
        logger.info("✅ Orchestrator client initialized for API Gateway")

    if conversation_controller is None:
        # Create controller with orchestrator client
        conversation_controller = ConversationController(orchestrator_client=orchestrator_client)
        await conversation_controller.initialize()
        logger.info("✅ ConversationController initialized for API Gateway")


async def cleanup_orchestrator_client():
    """Cleanup orchestrator client (called on app shutdown)"""
    global orchestrator_client
    if orchestrator_client:
        await orchestrator_client.cleanup()
        logger.info("✅ Orchestrator client cleaned up")


@router.post("/process", response_model=ProcessResponse)
async def process_request(request: ProcessRequest) -> ProcessResponse:
    """
    Process audio or text input through the ConversationController

    This endpoint uses ConversationController with OrchestratorClient backend.
    """
    # ==========================================
    # TELEMETRY START
    # ==========================================
    start_time = time.time()
    request_type = request.type
    session_id = request.session_id or f"api_{int(time.time()*1000)}"

    logger.info(f"🌐 [API GATEWAY] Received {request_type} request (session: {session_id})")

    global conversation_controller

    # Ensure controller is initialized
    if not conversation_controller:
        await initialize_orchestrator_client()

    try:
        # Prepare request for controller
        prep_start = time.time()
        request_dict = {
            'type': request.type,
            'session_id': session_id,
            'request_id': f"api_{int(time.time()*1000)}"
        }

        # Add type-specific fields
        if request.type == "audio":
            # Check for audio in different fields (backward compatibility)
            audio_b64 = request.audio or request.audio_data
            if not audio_b64:
                return ProcessResponse(
                    success=False,
                    error="Audio data required for audio type"
                )

            logger.info(f"   📊 Audio size: {len(audio_b64)} chars (base64)")

            request_dict.update({
                'audio': audio_b64,
                'sample_rate': request.sample_rate or 16000,
                'voice_id': request.voice_id,
                'force_external_llm': request.force_external_llm or False
            })

        elif request.type == "text":
            if not request.text:
                return ProcessResponse(
                    success=False,
                    error="Text data required for text type"
                )

            logger.info(f"   📝 Text: '{request.text[:50]}...'")

            request_dict.update({
                'text': request.text,
                'voice_id': request.voice_id
            })

        prep_time = (time.time() - prep_start) * 1000

        # ==========================================
        # TELEMETRY: Controller Processing
        # ==========================================
        controller_start = time.time()
        logger.info(f"   ➡️  Calling ConversationController...")

        # Process through controller (handles validation, processing, formatting)
        result = await conversation_controller.handle_request(request_dict)

        controller_time = (time.time() - controller_start) * 1000
        logger.info(f"   ⬅️  Controller responded: {controller_time:.0f}ms")

        # ==========================================
        # TELEMETRY END
        # ==========================================
        total_time = (time.time() - start_time) * 1000

        logger.info(f"✅ [API GATEWAY] Request processing complete: {total_time:.0f}ms total")
        logger.info(f"   📊 Breakdown: Prep={prep_time:.0f}ms + Controller={controller_time:.0f}ms")

        # Add API Gateway metrics to result
        if 'metrics' not in result:
            result['metrics'] = {}

        result['metrics'].update({
            'api_gateway_total_ms': int(total_time),
            'api_gateway_prep_ms': int(prep_time),
            'api_gateway_controller_ms': int(controller_time)
        })

        # Convert controller response to ProcessResponse
        return ProcessResponse(
            success=result.get('success', False),
            transcript=result.get('transcript'),
            response=result.get('response', ''),
            audio=result.get('audio'),
            metrics=result.get('metrics', {}),
            session_id=result.get('session_id'),
            error=result.get('error')
        )

    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        logger.error(f"❌ [API GATEWAY] Process request error ({total_time:.0f}ms): {e}")
        return ProcessResponse(
            success=False,
            error=str(e)
        )


@router.post("/process_audio")
async def process_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form("pt-BR"),
    session_id: Optional[str] = Form(None)
):
    """
    Process audio-only request through the orchestrator

    Accepts multipart/form-data with:
    - audio: Audio file (WAV format)
    - language: Language code (default: pt-BR)
    - session_id: Optional session ID

    Returns JSON with:
    - audio: Base64 encoded response audio
    - success: Boolean status
    """
    global orchestrator_client

    # Ensure orchestrator client is initialized
    if not orchestrator_client:
        await initialize_orchestrator_client()

    try:
        # Read audio file
        audio_data = await audio.read()

        session_id = session_id or f"audio_api_{int(time.time()*1000)}"

        # Process through orchestrator
        result = await orchestrator_client.process_turn(
            audio_data=audio_data,
            session_id=session_id,
            sample_rate=16000,  # Default for WAV
            voice_id=None  # Use session default
        )

        # Return audio response (audio is already base64)
        return {
            "success": result.get("success", False),
            "audio": result.get("audio", ""),
            "session_id": result.get("session_id"),
            "transcript": result.get("transcript"),
            "text": result.get("text")
        }

    except OrchestratorClientError as e:
        logger.error(f"Orchestrator client error: {e}")
        raise HTTPException(status_code=503, detail=f"Orchestrator error: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.post("/control")
async def control_request(command: str, session_id: str = "default_session", **kwargs) -> Dict[str, Any]:
    """
    Send control commands (not yet implemented with orchestrator)

    Commands:
    - clear_context: Clear conversation context
    - get_context: Get current context
    - set_voice: Change default voice
    """
    return {
        "success": True,
        "command": command,
        "message": f"Control command '{command}' received (not yet implemented with orchestrator)"
    }


@router.get("/process/status")
async def get_process_status() -> Dict[str, Any]:
    """
    Get the status of the processing pipeline
    """
    global orchestrator_client

    # Ensure orchestrator client is initialized
    if not orchestrator_client:
        await initialize_orchestrator_client()

    try:
        # Check orchestrator health via client
        health = await orchestrator_client.health_check()

        orchestrator_healthy = health.get("status") in ["healthy", "degraded"]

        return {
            "healthy": orchestrator_healthy,
            "orchestrator": health.get("status", "unknown"),
            "orchestrator_stats": health.get("stats", {}),
            "services": health.get("services", {}),
            "fallback": health.get("fallback", {}),
            "endpoint": "/process"
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "healthy": False,
            "error": str(e)
        }