"""
HTTP Routes for Orchestrator Service
All FastAPI endpoints organized by domain
"""

from fastapi import APIRouter, File, UploadFile, Form
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import logging
import time
import asyncio
import aiohttp
import os

from .models import ConversationRequest, ConversationResponse
from .utils.route_helpers import add_standard_endpoints

# Import validation utilities
try:
    from src.core.validation import (
        AudioInputValidator, TextInputValidator,
        validate_voice_id, AudioValidationError, TextValidationError
    )
except ImportError:
    # Fallback if validation not available
    AudioInputValidator = None
    TextInputValidator = None
    validate_voice_id = None
    AudioValidationError = Exception
    TextValidationError = Exception

logger = logging.getLogger(__name__)

async def transcribe_audio(
    audio_base64: str,
    stt_url: str,
    language: str = "pt"
) -> Dict[str, Any]:
    """
    Transcribe audio using STT service via HTTP

    Args:
        audio_base64: Base64 encoded audio data
        stt_url: STT service URL (e.g., http://localhost:8099)
        language: Language code for transcription (default: "pt")

    Returns:
        Dict with:
            - text: Transcribed text
            - processing_time_ms: Time taken for transcription
            - error: Error message if failed
    """
    try:
        start_time = time.time()

        # Direct HTTP call
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{stt_url}/transcribe",
                json={
                    "audio_base64": audio_base64,
                    "language": language
                },
                timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"STT service returned {resp.status}: {error_text}")
                        return {
                            "text": "",
                            "error": f"STT failed: {error_text}",
                            "processing_time_ms": (time.time() - start_time) * 1000
                        }

                    result = await resp.json()
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

def create_router(orchestrator_service: Any) -> APIRouter:
    """
    Create and configure the Orchestrator Service router

    Args:
        orchestrator_service: OrchestratorService instance for accessing state

    Returns:
        Configured APIRouter with all endpoints
    """
    router = APIRouter()

    # ==================== Health & Info ====================

    @router.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "orchestrator",
            "version": "1.0.0",
            "status": "running",
            "description": "Speech-to-Speech Orchestrator Service",
            "endpoints": {
                "health": "/api/health",
                "process": "/api/process",
                "conversation": "/api/conversation"
            }
        }

    @router.get("/health")
    async def health_check():
        """Basic health check endpoint (liveness)"""
        return {
            "status": "healthy",
            "service": "orchestrator",
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/health/live")
    async def liveness_probe():
        """Liveness probe - checks if service is alive"""
        return {
            "status": "healthy",
            "service": "orchestrator",
            "probe": "liveness",
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/health/ready")
    async def readiness_probe():
        """Readiness probe - checks if service is ready to accept traffic"""
        if not orchestrator_service or not orchestrator_service.orchestrator:
            return {
                "status": "unhealthy",
                "service": "orchestrator",
                "probe": "readiness",
                "message": "Service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Check dependencies
            health_status = await orchestrator_service.orchestrator._health_check_services()
            
            # Determine if ready
            critical_services = ["llm", "stt", "tts"]
            critical_healthy = all(health_status.get(s, False) for s in critical_services)
            
            if critical_healthy:
                status = "healthy"
                message = "Service is ready"
            else:
                status = "degraded"
                message = "Service is ready but some dependencies are unhealthy"
            
            return {
                "status": status,
                "service": "orchestrator",
                "probe": "readiness",
                "message": message,
                "dependencies": health_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "orchestrator",
                "probe": "readiness",
                "message": f"Readiness check failed: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    @router.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with comprehensive information"""
        if not orchestrator_service or not orchestrator_service.orchestrator:
            return {
                "status": "unhealthy",
                "service": "orchestrator",
                "message": "Service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Check dependencies
            health_status = await orchestrator_service.orchestrator._health_check_services()
            
            # Collect metrics
            metrics = {}
            if orchestrator_service.orchestrator:
                metrics["clients_initialized"] = len(orchestrator_service.orchestrator.clients)
                metrics["in_process_mode"] = orchestrator_service.orchestrator.in_process_mode
            
            # Determine overall status
            critical_services = ["llm", "stt", "tts"]
            critical_healthy = all(health_status.get(s, False) for s in critical_services)
            all_healthy = all(health_status.values())
            
            if all_healthy:
                status = "healthy"
                message = "All systems operational"
            elif critical_healthy:
                status = "degraded"
                message = "Critical services healthy, some non-critical services unavailable"
            else:
                status = "unhealthy"
                message = "Critical services unavailable"
            
            return {
                "status": status,
                "service": "orchestrator",
                "message": message,
                "dependencies": health_status,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "orchestrator",
                "message": f"Health check failed: {e}",
                "timestamp": datetime.now().isoformat()
            }

    # ==================== Speech-to-Speech Processing ====================

    @router.post("/process")
    async def process_speech_to_speech(
        file: Optional[UploadFile] = File(None),
        audio_base64: Optional[str] = Form(None),
        language: str = Form("en"),
        voice_id: Optional[str] = Form(default=None),  # None uses TTS service default
        max_tokens: int = Form(100),
        temperature: float = Form(0.7),
        voice_speed: float = Form(1.0),
        stt_model: str = Form("base")
    ):
        """
        Process speech-to-speech: Audio Input → STT → LLM → TTS → Audio Output

        Args:
            file: Audio file (WAV, MP3, etc.) - uploaded directly
            audio_base64: Base64 encoded audio data
            language: Language for STT (default: "en")
            voice_id: Voice ID for TTS (default: None, uses TTS service default)
            max_tokens: Max tokens for LLM response (default: 100)
            temperature: Temperature for LLM generation (default: 0.7)
            voice_speed: Voice speed multiplier for TTS (default: 1.0)
            stt_model: Whisper model for STT (default: "base")

        Returns:
            Dict with:
                - audio_base64: Base64 encoded response audio
                - transcription: Original transcribed text
                - response_text: LLM generated response
                - processing_times: Time breakdown for each step
        """
        import base64
        import io

        start_time = time.time()
        processing_times = {}

        try:
            # Validate and normalize voice_id
            if validate_voice_id:
                is_valid, error, normalized_voice_id = validate_voice_id(voice_id)
                if not is_valid:
                    raise ValueError(f"Invalid voice_id: {error}")
                voice_id = normalized_voice_id
            else:
                # Fallback normalization
                original_voice_id = voice_id
                if voice_id in (None, "None", ""):
                    voice_id = None
                logger.info(f"🎤 Using voice: {voice_id if voice_id else 'default'}")
            
            # Step 1: Get and validate audio data
            audio_data = None
            if file and file.size > 0:
                audio_data = await file.read()
                logger.info(f"📁 Received uploaded file: {len(audio_data)} bytes")
            elif audio_base64:
                if AudioInputValidator:
                    validator = AudioInputValidator()
                    is_valid, error, decoded = validator.validate(audio_base64=audio_base64)
                    if not is_valid:
                        raise AudioValidationError(error)
                    audio_data = decoded
                else:
                    audio_data = base64.b64decode(audio_base64)
                logger.info(f"📄 Received base64 audio: {len(audio_data)} bytes")
            else:
                raise ValueError("Either 'file' or 'audio_base64' must be provided")
            
            # Validate audio data
            if AudioInputValidator:
                validator = AudioInputValidator()
                is_valid, error, validated_audio = validator.validate(audio_data=audio_data)
                if not is_valid:
                    raise AudioValidationError(error)
                audio_data = validated_audio
            
            # Validate max_tokens
            if max_tokens > 2000:
                logger.warning(f"⚠️  max_tokens ({max_tokens}) exceeds recommended limit (2000)")
                max_tokens = min(max_tokens, 2000)

            # Step 2: STT - Transcribe audio to text
            logger.info("🎙️ Step 2: Starting STT transcription...")
            stt_start = time.time()

            # Try to get STT service URL
            stt_url = os.getenv("STT_SERVICE_URL", "http://localhost:8099")  # STT service default port

            # Call STT service with multipart form data (using /transcribe-file endpoint)
            async with aiohttp.ClientSession() as session:
                # Create multipart form data
                from aiohttp import FormData
                form = FormData()
                form.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')
                form.add_field('language', language)
                form.add_field('model', 'whisper-large-v3')  # Use correct model name

                async with session.post(
                    f"{stt_url}/transcribe-file",  # Use correct endpoint for file upload
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"STT service returned {resp.status}: {error_text}")
                        raise Exception(f"STT service returned {resp.status}: {error_text}")

                    stt_result = await resp.json()
                    transcription = stt_result.get("text", "").strip()
            
            # Validate transcription text
            if TextInputValidator and transcription:
                text_validator = TextInputValidator()
                is_valid, error, sanitized = text_validator.validate(transcription, for_llm=False)
                if not is_valid:
                    logger.warning(f"⚠️  Transcription validation warning: {error}")
                else:
                    transcription = sanitized

            processing_times["stt"] = (time.time() - stt_start) * 1000
            if not transcription:
                raise Exception("STT returned empty transcription")

            logger.info(f"✅ STT completed: '{transcription}' ({processing_times['stt']:.1f}ms)")

            # Step 3: LLM - Generate response
            logger.info("🤖 Step 3: Starting LLM processing...")
            llm_start = time.time()

            # Get LLM service URL
            llm_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8110")  # LLM service default port

            # Create a conversational prompt
            prompt = f"User said: '{transcription}'\n\nRespond naturally and helpfully:"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{llm_url}/generate",
                    json={
                        "prompt": prompt,  # LLM service expects "prompt" not "text"
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"LLM failed: {error_text}")

                    llm_result = await resp.json()
                    response_text = llm_result.get("text", "").strip()

            processing_times["llm"] = (time.time() - llm_start) * 1000

            if not response_text:
                raise Exception("LLM returned empty response")

            logger.info(f"✅ LLM completed: '{response_text}' ({processing_times['llm']:.1f}ms)")

            # Step 4: TTS - Convert response to speech
            logger.info("🔊 Step 4: Starting TTS synthesis...")
            tts_start = time.time()

            # Get TTS service URL
            tts_url = os.getenv("TTS_SERVICE_URL", "http://localhost:8103")  # TTS service default port

            # Prepare TTS request - always specify provider
            logger.info(f"🎤 Preparing TTS payload - voice_id: '{voice_id}' (type: {type(voice_id)})")
            tts_payload = {
                "text": response_text,
                "speed": voice_speed,
                "provider": "elevenlabs"  # Always specify provider (Eleven Labs is default)
            }
            # Only include voice if explicitly specified and valid
            # The TTS service will use default voice for the provider if not specified
            # Validate voice before adding to payload
            # CRITICAL: Normalize known invalid voices FIRST
            known_invalid_voices = ["pf_dora"]
            if voice_id in known_invalid_voices:
                logger.warning(f"⚠️  Voice '{voice_id}' is known to be invalid, normalizing to None")
                voice_id = None
            
            if voice_id and voice_id != "None" and voice_id != "":
                # Check if voice is valid for Eleven Labs (basic check)
                valid_elevenlabs_voices = ["Rachel", "Drew", "Clyde", "Paul", "Domi", "Dave", "Fin", "Bella", "Antoni", "Thomas", "Charlie", "Emily", "Elli", "Josh", "Arnold", "Adam", "Sam"]
                if voice_id not in valid_elevenlabs_voices:
                    logger.warning(f"⚠️  Voice '{voice_id}' is not valid for Eleven Labs, using default")
                    voice_id = None
            
            if voice_id and voice_id != "None" and voice_id != "":
                tts_payload["voice"] = voice_id
                logger.info(f"🎤 Added voice '{voice_id}' to TTS payload with provider 'elevenlabs'")
            else:
                logger.info(f"🎤 Not adding voice to TTS payload (voice_id='{voice_id}'), will use default voice for provider")
            # If voice is not specified, TTS service will use default voice for the selected provider
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{tts_url}/synthesize",
                    json=tts_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"TTS failed: {error_text}")

                    tts_result = await resp.json()
                    # TTS service returns "audio_data" not "audio_base64"
                    response_audio_base64 = tts_result.get("audio_data") or tts_result.get("audio_base64")

            processing_times["tts"] = (time.time() - tts_start) * 1000

            if not response_audio_base64:
                raise Exception("TTS returned no audio data")

            logger.info(f"✅ TTS completed: {len(response_audio_base64)} chars ({processing_times['tts']:.1f}ms)")

            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000
            processing_times["total"] = total_time

            logger.info(f"🎉 Pipeline completed successfully in {total_time:.1f}ms")

            # Record success and latency
            total_time = time.time() - start_time
            try:
                from src.core.prometheus_metrics import increment_counter, record_latency
                increment_counter("orchestrator", "requests_total", labels={"endpoint": "process", "status": "success"})
                record_latency("orchestrator", "process", total_time)
            except ImportError:
                pass
            
            return {
                "success": True,
                "audio_base64": response_audio_base64,
                "transcription": transcription,
                "response_text": response_text,
                "processing_times": processing_times,
                "metadata": {
                    "language": language,
                    "voice_id": voice_id,
                    "model": "speech-to-speech-pipeline"
                }
            }

        except Exception as e:
            # Record error
            total_time = time.time() - start_time
            try:
                from src.core.prometheus_metrics import increment_counter, record_latency
                increment_counter("orchestrator", "requests_total", labels={"endpoint": "process", "status": "error"})
                increment_counter("orchestrator", "errors_total", labels={"endpoint": "process", "error_type": type(e).__name__})
            except ImportError:
                pass
            error_msg = f"Speech-to-speech processing failed: {str(e)}"
            logger.error(error_msg)

            # Calculate total time even on error
            total_time = (time.time() - start_time) * 1000

            return {
                "success": False,
                "error": error_msg,
                "processing_times": {
                    "total": total_time,
                    **processing_times
                }
            }

    # ==================== Legacy Conversation Endpoint ====================

    @router.post("/conversation")
    async def conversation_endpoint(request: ConversationRequest) -> ConversationResponse:
        """
        Text conversation endpoint - processes text messages and returns text responses
        Optionally generates audio if voice_id is provided
        """
        if not orchestrator_service or not orchestrator_service.orchestrator:
            return ConversationResponse(
                success=False,
                response="",
                session_id=request.session_id,
                error="Orchestrator service not initialized"
            )
        
        try:
            # Use process_text_conversation method
            result = await orchestrator_service.orchestrator.process_text_conversation(
                message=request.message,
                session_id=request.session_id,
                voice_id=request.voice_id,
                scenario_id_override=request.scenario_id
            )
            
            return ConversationResponse(
                success=result.get("success", False),
                response=result.get("response", ""),
                session_id=result.get("session_id", request.session_id),
                audio=result.get("audio"),  # Base64 encoded if voice_id provided
                transcription=None,
                context_size=result.get("context_size"),
                messages_count=result.get("messages_count"),
                metrics=result.get("metrics"),
                error=result.get("error")
            )
        except Exception as e:
            logger.error(f"Error in conversation endpoint: {e}", exc_info=True)
            return ConversationResponse(
                success=False,
                response="",
                session_id=request.session_id,
                error=f"Conversation processing failed: {str(e)}"
            )

    # ==================== Add Standard Endpoints ====================

    try:
        add_standard_endpoints(router, orchestrator_service, "orchestrator")
    except Exception as e:
        logger.warning(f"Failed to add standard endpoints: {e}")

    return router