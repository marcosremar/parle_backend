"""
HTTP Routes for Webrtc Signaling Service
All FastAPI endpoints organized by domain
"""

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from typing import Dict, Optional, List, Any
from datetime import datetime
import asyncio
import logging
import os
from .utils.route_helpers import add_standard_endpoints

logger = logging.getLogger(__name__)

# ✅ Phase 3c: Dynamic port support
def _get_webrtc_signaling_port():
    """Get webrtc_signaling port from environment or registry"""
    try:
        from src.config.service_config import ServiceType, get_service_port
        return int(os.getenv("WEBRTC_SIGNALING_PORT") or get_service_port(ServiceType.WEBRTC_SIGNALING))
    except (ImportError, ValueError, TypeError, KeyError):
        logger.debug("Could not retrieve webrtc_signaling port from config, using fallback")
        return 8090  # Fallback to PORT_MATRIX default

def create_router(webrtc_signaling_service: Any) -> APIRouter:
    """
    Create and configure the Webrtc Signaling Service router

    Args:
        webrtc_signaling_service: WebrtcSignalingService instance

    Returns:
        Configured APIRouter with all endpoints
    """
    router = APIRouter()

    # Add standard endpoints (health, info, etc.)
    add_standard_endpoints(router, webrtc_signaling_service)

    # Define WebSocket endpoint directly to avoid router nesting issues
    @router.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        logger.info(f"🔌 WebSocket connection REQUEST from {client_id}")
        logger.info(f"   Client: {websocket.client}")
        logger.info(f"   Headers: {websocket.headers}")
        
        if hasattr(webrtc_signaling_service, "signaling_manager"):
            manager = webrtc_signaling_service.signaling_manager
            try:
                await manager.connect(client_id, websocket)
                logger.info(f"✅ WebSocket CONNECTED for {client_id}")
            except Exception as e:
                logger.error(f"❌ WebSocket connection FAILED for {client_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
                
            try:
                while True:
                    data = await websocket.receive_json()
                    message_type = data.get('type')
                    
                    logger.info(f"📨 Received {message_type} from {client_id}")
                    
                    # Handle ping/pong for latency measurement
                    if message_type == 'ping':
                        await websocket.send_json({
                            'type': 'pong',
                            'timestamp': data.get('timestamp')
                        })
                        continue
                    
                    # Handle audio data for speech-to-speech
                    if message_type == 'audio_data':
                        try:
                            import base64
                            
                            # Get audio bytes from client
                            audio_array = data.get('audio', [])
                            audio_bytes = bytes(audio_array)
                            
                            logger.info(f"🎤 Received {len(audio_bytes)} bytes of audio from {client_id}")
                            
                            # Step 1: Transcribe audio (STT)
                            from src.api.main import get_module
                            logger.info(f"📝 Getting STT module...")
                            stt_module = get_module("stt")
                            
                            # Ensure module is initialized
                            if not stt_module.initialized:
                                logger.info(f"🔄 Initializing STT module...")
                                await stt_module.initialize()
                            
                            # Check if provider is available
                            if not stt_module.provider:
                                error_msg = "STT provider not initialized. Please check GROQ_API_KEY is set in .env"
                                logger.error(f"❌ {error_msg}")
                                await websocket.send_json({
                                    'type': 'error',
                                    'error': error_msg
                                })
                                continue
                            
                            logger.info(f"🎯 Transcribing audio...")
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                            transcription_result = await stt_module.transcribe(
                                audio_base64=audio_base64,
                                language="pt"
                            )
                            
                            user_text = transcription_result.get('text', '')
                            logger.info(f"📝 Transcribed: {user_text}")
                            
                            # Send transcription to client
                            await websocket.send_json({
                                'type': 'transcription',
                                'text': user_text
                            })
                            
                            # Step 2: Generate response using orchestrator
                            logger.info(f"🤖 Getting Orchestrator module...")
                            orchestrator_module = get_module("orchestrator")
                            
                            # Ensure module is initialized
                            if not orchestrator_module.initialized:
                                logger.info(f"🔄 Initializing Orchestrator module...")
                                await orchestrator_module.initialize()
                            
                            logger.info(f"🧠 Processing conversation...")
                            # Process conversation with real AI
                            conversation_result = await orchestrator_module.process_text_conversation(
                                message=user_text,
                                session_id=client_id,  # Use client_id as session
                                voice_id="pt"  # Portuguese voice
                            )
                            
                            # Get bot response text
                            bot_response = conversation_result.get('response_text', 
                                                                  conversation_result.get('text', 
                                                                                        'Desculpe, não entendi.'))
                            
                            logger.info(f"🤖 Bot response: {bot_response}")
                            
                            # Send text response to client
                            await websocket.send_json({
                                'type': 'response_text',
                                'text': bot_response
                            })
                            
                            # Step 3: Synthesize speech (TTS)
                            logger.info(f"🔊 Getting TTS module...")
                            tts_module = get_module("tts")
                            
                            # Ensure module is initialized
                            if not tts_module.initialized:
                                logger.info(f"🔄 Initializing TTS module...")
                                await tts_module.initialize()
                            
                            logger.info(f"🎵 Synthesizing speech...")
                            tts_result = await tts_module.synthesize(
                                text=bot_response,
                                language="pt",
                                provider="gtts"
                            )
                            
                            # Get audio data
                            audio_response_base64 = tts_result.get('audio_base64', '')
                            audio_response_bytes = base64.b64decode(audio_response_base64)
                            
                            logger.info(f"🔊 Generated {len(audio_response_bytes)} bytes of audio response")
                            
                            # Send audio response to client
                            await websocket.send_json({
                                'type': 'audio_response',
                                'audio': list(audio_response_bytes)
                            })
                            
                            logger.info(f"✅ Speech-to-speech completed for {client_id}")
                            
                        except Exception as e:
                            logger.error(f"❌ Speech processing failed for {client_id}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            await websocket.send_json({
                                'type': 'error',
                                'error': str(e)
                            })
                        continue
                    
                    # Handle WebRTC offer - generate answer
                    if message_type == 'offer':
                        try:
                            # Generate SDP answer using the service
                            answer = await webrtc_signaling_service.generate_answer(
                                client_sdp=data.get('sdp'),
                                client_id=client_id
                            )
                            
                            # Send answer back to client
                            await websocket.send_json({
                                'type': 'answer',
                                'sdp': answer
                            })
                            logger.info(f"✅ Sent SDP answer to {client_id}")
                        except Exception as e:
                            logger.error(f"❌ Failed to generate answer for {client_id}: {e}")
                            await websocket.send_json({
                                'type': 'error',
                                'error': str(e)
                            })
                        continue
                    
                    # Handle ICE candidates
                    if message_type == 'ice-candidate':
                        candidate = data.get('candidate')
                        if candidate:
                            try:
                                await webrtc_signaling_service.add_ice_candidate(
                                    client_id=client_id,
                                    candidate_data=candidate
                                )
                                logger.info(f"🧊 Added ICE candidate for {client_id}")
                            except Exception as e:
                                logger.error(f"❌ Failed to add ICE candidate: {e}")
                        continue
                    
                    # For other message types, broadcast to others (mesh)
                    await manager.broadcast(data, exclude=client_id)
                    
            except WebSocketDisconnect as e:
                logger.info(f"🔌 Client {client_id} disconnected (code: {e.code})")
                await manager.disconnect(client_id)
            except Exception as e:
                logger.error(f"❌ Error in WebSocket handler for {client_id}: {e}")
                logger.error(f"   Error type: {type(e).__name__}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    await manager.disconnect(client_id)
                except Exception:
                    pass
        else:
            logger.error(f"❌ No signaling_manager available for {client_id}")
            await websocket.close(code=1011, reason="Server configuration error")

    return router