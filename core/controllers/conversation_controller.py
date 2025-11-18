"""
Conversation Controller
Handles conversation-related requests from WebRTC and API Gateway
Supports both Pipeline and OrchestratorClient backends
"""

import base64
import numpy as np
from typing import Dict, Any, Optional
from functools import lru_cache
from .base_controller import BaseController, RateLimiterConfig
from src.core.model_manager import get_shared_pipeline


class ConversationController(BaseController):
    """
    Controller for handling conversation requests

    Supports two backends:
    1. Pipeline mode: Uses ConversationPipeline directly (legacy)
    2. Orchestrator mode: Uses OrchestratorClient (recommended)
    """

    def __init__(self,
                 pipeline: Optional[object] = None,
                 orchestrator_client: Optional[object] = None,
                 rate_limiter_config: Optional[RateLimiterConfig] = None):
        """
        Initialize conversation controller

        Args:
            pipeline: Optional pre-initialized ConversationPipeline or Orchestrator
            orchestrator_client: Optional OrchestratorClient for API Gateway integration
            rate_limiter_config: Optional rate limiter configuration (default: disabled)
        """
        super().__init__(rate_limiter_config=rate_limiter_config)
        self.pipeline = pipeline  # Direct pipeline/orchestrator instance
        self.orchestrator_client = orchestrator_client  # OrchestratorClient for HTTP calls
        self.use_orchestrator = orchestrator_client is not None
        self.sessions = {}  # Track sessions for context management

    async def initialize(self):
        """Initialize the backend (pipeline or orchestrator client)"""
        if self.orchestrator_client:
            # Using OrchestratorClient mode
            self.logger.info("🔗 ConversationController using OrchestratorClient mode")
            self.use_orchestrator = True
        elif self.pipeline:
            # Using pre-initialized pipeline
            self.logger.info("🔗 ConversationController using pre-initialized pipeline")
        else:
            # Fallback: get shared pipeline from ModelManager
            self.logger.info("🔗 Getting shared ConversationPipeline from ModelManager...")
            self.pipeline = await get_shared_pipeline()
            self.logger.info("✅ ConversationController connected to shared pipeline")

    @lru_cache(maxsize=1000)
    def _validate_base64_format(self, audio_b64: str) -> bool:
        """
        Validate Base64 format with LRU cache for performance

        Args:
            audio_b64: Base64 encoded audio string

        Returns:
            True if valid Base64, False otherwise
        """
        try:
            base64.b64decode(audio_b64)
            return True
        except (ValueError, TypeError):
            # Invalid base64 string
            return False

    def set_orchestrator_client(self, orchestrator_client):
        """
        Set orchestrator client for API Gateway integration

        Args:
            orchestrator_client: OrchestratorClient instance
        """
        self.orchestrator_client = orchestrator_client
        self.use_orchestrator = True
        self.logger.info("🔗 ConversationController switched to OrchestratorClient mode")

    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate conversation request (Ultravox LLM format)

        Expected format for audio requests (what Ultravox LLM needs):
        - audio: Base64 string → decoded to bytes → converted to numpy float32
        - sample_rate: 16000 Hz (REQUIRED by Ultravox)
        - type: "audio" or "text"

        The flow is:
        1. Client sends: {audio: "base64_string", sample_rate: 16000}
        2. Controller validates: Base64 is valid, can be decoded
        3. Controller decodes: Base64 → bytes
        4. Pipeline converts: bytes → numpy float32 array
        5. Ultravox processes: numpy array with <|audio|> placeholder

        Args:
            request_data: Request to validate

        Returns:
            Validation result dict with 'valid' and optional 'error'
        """
        # Check request type
        request_type = request_data.get('type')
        if not request_type:
            return {'valid': False, 'error': 'Missing request type'}

        # Validate based on type
        if request_type == 'text':
            if 'text' not in request_data:
                return {'valid': False, 'error': 'Missing text field'}
            if not request_data['text'].strip():
                return {'valid': False, 'error': 'Empty text'}

        elif request_type == 'audio':
            # Validate audio field (must be Base64 string)
            if 'audio' not in request_data:
                return {'valid': False, 'error': 'Missing audio field - Ultravox requires audio data'}

            audio_b64 = request_data['audio']
            if not audio_b64 or not isinstance(audio_b64, str):
                return {'valid': False, 'error': 'Audio must be a Base64 encoded string'}

            # Validate Base64 format using cached method (LRU cache for performance)
            if not self._validate_base64_format(audio_b64):
                return {'valid': False, 'error': 'Invalid Base64 audio format'}

            # Decode to bytes (now we know it's valid)
            try:
                audio_bytes = base64.b64decode(audio_b64)
                if len(audio_bytes) == 0:
                    return {'valid': False, 'error': 'Audio data is empty after Base64 decode'}

                # Log decoded size for debugging
                self.logger.debug(f"✅ Audio decoded: {len(audio_bytes)} bytes ({len(audio_bytes)/16000:.2f}s @ 16kHz)")

            except Exception as e:
                return {'valid': False, 'error': f'Invalid Base64 audio data: {str(e)}'}

            # Validate sample_rate (Ultravox REQUIRES 16000 Hz)
            sample_rate = request_data.get('sample_rate', 16000)
            if sample_rate != 16000:
                self.logger.warning(f"⚠️ Ultravox requires 16000 Hz, received {sample_rate} Hz - may cause processing errors")
                # Don't fail - pipeline may handle resampling, but warn user

            # Validate minimum audio duration (Ultravox needs at least 2 hops)
            min_samples = 640  # ~40ms at 16kHz (2 hops for Whisper encoder)
            estimated_samples = len(audio_bytes) // 2  # Assuming int16 format
            if estimated_samples < min_samples:
                return {'valid': False, 'error': f'Audio too short: {estimated_samples} samples ({estimated_samples/16000*1000:.0f}ms), minimum {min_samples} samples'}

            # Validate audio size (should be reasonable)
            audio_size_mb = len(audio_bytes) / (1024 * 1024)
            MAX_AUDIO_SIZE_MB = 50  # 50 MB max
            if audio_size_mb > MAX_AUDIO_SIZE_MB:
                return {'valid': False, 'error': f'Audio too large: {audio_size_mb:.1f} MB (max {MAX_AUDIO_SIZE_MB} MB)'}

        elif request_type == 'control':
            command = request_data.get('command')
            if command not in ['clear_context', 'get_context', 'set_voice']:
                return {'valid': False, 'error': f'Unknown command: {command}'}

        else:
            return {'valid': False, 'error': f'Unknown request type: {request_type}'}

        return {'valid': True}

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process validated conversation request

        Args:
            request_data: Validated request

        Returns:
            Response dictionary
        """
        request_type = request_data['type']
        session_id = request_data.get('session_id', 'default')

        # Handle different request types
        if request_type == 'text':
            return await self._process_text_request(request_data, session_id)

        elif request_type == 'audio':
            return await self._process_audio_request(request_data, session_id)

        elif request_type == 'control':
            return await self._process_control_request(request_data, session_id)

        else:
            return self._error_response(
                error=f"Unhandled request type: {request_type}",
                request_id=request_data.get('request_id', 'unknown')
            )

    async def _process_text_request(self,
                                   request_data: Dict[str, Any],
                                   session_id: str) -> Dict[str, Any]:
        """Process text conversation request"""
        text = request_data['text']
        voice_id = request_data.get('voice_id', 'af_bella')
        language = request_data.get('language', 'English')

        # Process through pipeline
        result = await self.pipeline.process(
            input_data=text,
            voice_id=voice_id,
            language=language,
            return_audio=True
        )

        if result['success']:
            # Encode audio to base64 for WebRTC transport
            audio_base64 = None
            if result.get('audio_output'):
                audio_base64 = base64.b64encode(result['audio_output']).decode()

            return {
                'success': True,
                'response': result['response'],
                'audio': audio_base64,
                'metrics': result['metrics'],
                'session_id': session_id
            }
        else:
            return self._error_response(
                error=result.get('error', 'Processing failed'),
                request_id=request_data.get('request_id', 'unknown')
            )

    async def _process_audio_request(self,
                                    request_data: Dict[str, Any],
                                    session_id: str) -> Dict[str, Any]:
        """Process audio conversation request"""
        # Decode audio from base64
        audio_base64 = request_data['audio']
        sample_rate = request_data.get('sample_rate', 16000)
        voice_id = request_data.get('voice_id')
        force_external_llm = request_data.get('force_external_llm', False)

        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)

            # Choose backend: OrchestratorClient or Pipeline
            if self.use_orchestrator and self.orchestrator_client:
                # Use OrchestratorClient (API Gateway mode)
                result = await self.orchestrator_client.process_turn(
                    audio_data=audio_bytes,
                    session_id=session_id,
                    sample_rate=sample_rate,
                    voice_id=voice_id,
                    force_external_llm=force_external_llm
                )

                # Handle audio encoding (orchestrator may return bytes or base64)
                audio_response = result.get("audio")
                if audio_response:
                    if isinstance(audio_response, bytes):
                        audio_response = base64.b64encode(audio_response).decode('utf-8')
                    elif isinstance(audio_response, dict):
                        audio_response = audio_response.get("audio", "")

                return {
                    'success': result.get('success', False),
                    'transcript': result.get('transcript', ''),
                    'response': result.get('text', ''),  # Orchestrator returns 'text'
                    'audio': audio_response,
                    'metrics': result.get('metrics', {}),
                    'session_id': result.get('session_id', session_id),
                    'error': result.get('error')
                }

            else:
                # Use Pipeline directly (WebRTC mode or legacy)
                language = request_data.get('language', 'English')

                # Convert to numpy array if needed
                if request_data.get('format') == 'int16':
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                else:
                    audio_array = audio_bytes

                # Process through pipeline
                result = await self.pipeline.process(
                    input_data=audio_array,
                    sample_rate=sample_rate,
                    voice_id=voice_id,
                    language=language,
                    return_audio=True
                )

                if result['success']:
                    # Encode output audio to base64
                    audio_output_base64 = None
                    if result.get('audio_output'):
                        audio_output_base64 = base64.b64encode(result['audio_output']).decode()

                    # Add timing information to response
                    response_data = {
                        'success': True,
                        'transcript': result.get('transcript', ''),
                        'response': result['response'],
                        'audio': audio_output_base64,
                        'metrics': result['metrics'],
                        'session_id': session_id
                    }

                    # Add timing details if available
                    if 'timings' in result:
                        response_data['timings'] = result['timings']

                    return response_data
                else:
                    return self._error_response(
                        error=result.get('error', 'Audio processing failed'),
                        request_id=request_data.get('request_id', 'unknown')
                    )

        except Exception as e:
            return self._error_response(
                error=f"Audio processing error: {str(e)}",
                request_id=request_data.get('request_id', 'unknown')
            )

    async def _process_control_request(self,
                                      request_data: Dict[str, Any],
                                      session_id: str) -> Dict[str, Any]:
        """Process control commands"""
        command = request_data['command']

        if command == 'clear_context':
            self.pipeline.clear_context()
            return {
                'success': True,
                'message': 'Context cleared',
                'session_id': session_id
            }

        elif command == 'get_context':
            context = self.pipeline.get_context()
            return {
                'success': True,
                'context': context,
                'session_id': session_id
            }

        elif command == 'set_voice':
            voice_id = request_data.get('voice_id')
            if voice_id:
                self.pipeline.config['default_voice'] = voice_id
                return {
                    'success': True,
                    'message': f'Voice set to {voice_id}',
                    'session_id': session_id
                }
            else:
                return self._error_response(
                    error='Missing voice_id',
                    request_id=request_data.get('request_id', 'unknown')
                )

        else:
            return self._error_response(
                error=f'Unknown command: {command}',
                request_id=request_data.get('request_id', 'unknown')
            )