"""
Orchestrator Integration with Perceived Latency Manager

Shows how to integrate the PerceivedLatencyManager with the Orchestrator service.
This module provides helper functions and mixin classes for seamless integration.

Example Usage in Orchestrator:
    async def process_request(self, request: ConversationRequest):
        # Create/get latency manager for session
        manager = self.get_latency_manager(request.session_id)
        manager.start_turn(request.turn_number)

        # Track STT phase
        manager.start_stt()
        transcript = await self.transcribe_audio(request.audio)
        manager.stop_stt()

        # Track LLM phase
        manager.start_llm()
        response = await self.generate_response(transcript)
        manager.stop_llm()

        # Track TTS phase (if applicable)
        manager.start_tts()
        audio_response = await self.synthesize_speech(response)
        manager.stop_tts()

        # Generate component outputs in parallel
        internal_talk = self.generate_internal_talk(...)
        hints = self.generate_hints_for_next_turn(...)
        errors = self.detect_errors(...)

        # Finalize and get metrics
        metrics = manager.finalize_turn(
            internal_talk=internal_talk,
            hints=hints,
            errors=errors
        )

        # Return response with metrics
        return self.create_streaming_response(
            response=response,
            metrics=metrics
        )
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import asyncio

# Import models
from .perceived_latency_manager import (
    PerceivedLatencyManager,
    InternalTalk,
    HintsForNextTurn,
    ErrorDetection,
    PerceivedLatencyMetrics
)
from .streaming_response_models import (
    StreamingConversationResponse,
    LatencyMetrics,
    InternalTalkResponse,
    HintsForNextTurnResponse,
    ErrorDetectionResponse,
    ComponentOutputsResponse,
    ResponseSegmentsMetrics,
    PerceivedLatencyClassification
)


class OrchestratorLatencyMixin:
    """
    Mixin for Orchestrator to integrate with PerceivedLatencyManager

    Add this to OrchestratorService to enable perceived latency tracking.

    Usage:
        class OrchestratorService(BaseService, OrchestratorLatencyMixin):
            def __init__(self, ...):
                super().__init__(...)
                self.initialize_latency_tracking()
    """

    def initialize_latency_tracking(self) -> None:
        """Initialize latency tracking infrastructure"""
        self._latency_managers: Dict[str, PerceivedLatencyManager] = {}
        self._lock = asyncio.Lock()
        self.logger.info("✅ Latency tracking initialized")

    def get_latency_manager(self, session_id: str) -> PerceivedLatencyManager:
        """Get or create latency manager for session"""
        if session_id not in self._latency_managers:
            self._latency_managers[session_id] = PerceivedLatencyManager(session_id)
        return self._latency_managers[session_id]

    def cleanup_latency_manager(self, session_id: str) -> None:
        """Clean up latency manager for completed session"""
        if session_id in self._latency_managers:
            del self._latency_managers[session_id]

    async def get_latency_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of latency metrics for session"""
        manager = self.get_latency_manager(session_id)
        return manager.get_summary()


class OrchestrationContext:
    """Context for a single conversation turn's orchestration"""

    def __init__(
        self,
        session_id: str,
        turn_number: int,
        manager: PerceivedLatencyManager
    ):
        self.session_id = session_id
        self.turn_number = turn_number
        self.manager = manager
        self.user_text = ""
        self.llm_response = ""
        self.user_audio_bytes = 0
        self.response_audio_bytes = 0

    def start_phase(self, phase: str) -> None:
        """Start tracking a phase (stt, llm, tts)"""
        if phase == "stt":
            self.manager.start_stt()
        elif phase == "llm":
            self.manager.start_llm()
        elif phase == "tts":
            self.manager.start_tts()

    def stop_phase(self, phase: str) -> float:
        """Stop tracking a phase and return duration"""
        if phase == "stt":
            return self.manager.stop_stt()
        elif phase == "llm":
            return self.manager.stop_llm()
        elif phase == "tts":
            return self.manager.stop_tts()
        return 0.0

    def set_response_data(
        self,
        user_text: str,
        llm_response: str,
        user_audio_bytes: int = 0,
        response_audio_bytes: int = 0
    ) -> None:
        """Set response data"""
        self.user_text = user_text
        self.llm_response = llm_response
        self.user_audio_bytes = user_audio_bytes
        self.response_audio_bytes = response_audio_bytes

    async def finalize(
        self,
        internal_talk: Optional[InternalTalk] = None,
        hints: Optional[HintsForNextTurn] = None,
        errors: Optional[ErrorDetection] = None
    ) -> PerceivedLatencyMetrics:
        """Finalize turn and return metrics"""
        return self.manager.finalize_turn(
            internal_talk=internal_talk,
            hints=hints,
            errors=errors
        )


class StreamingResponseBuilder:
    """Helper to build streaming responses with latency metrics"""

    @staticmethod
    def from_metrics(
        turn_number: int,
        user_text: str,
        llm_response: str,
        metrics: PerceivedLatencyMetrics
    ) -> StreamingConversationResponse:
        """
        Build a streaming response from metrics

        Args:
            turn_number: Conversation turn
            user_text: User input
            llm_response: AI response
            metrics: PerceivedLatencyMetrics from manager

        Returns:
            StreamingConversationResponse ready for streaming
        """
        # Convert latency metrics
        latencies = LatencyMetrics.from_components(
            stt_ms=metrics.latencies.stt_time_ms,
            llm_ms=metrics.latencies.llm_time_ms,
            tts_ms=metrics.latencies.tts_time_ms
        )

        # Convert components
        components = None
        if metrics.components:
            internal_talk = None
            if metrics.components.internal_talk:
                internal_talk = InternalTalkResponse(
                    content=metrics.components.internal_talk.content,
                    generation_time_ms=metrics.components.internal_talk.generation_time_ms
                )

            hints = None
            if metrics.components.hints:
                hints = HintsForNextTurnResponse.from_hints(
                    hints=metrics.components.hints.hints,
                    generation_time_ms=metrics.components.hints.generation_time_ms
                )

            errors = None
            if metrics.components.errors:
                error_dicts = [e.to_dict() for e in metrics.components.errors.errors]
                errors = ErrorDetectionResponse.from_errors(
                    errors=error_dicts,
                    detection_time_ms=metrics.components.errors.detection_time_ms
                )

            response_metrics = ResponseSegmentsMetrics(
                max_parallel_generation_time_ms=metrics.components.max_generation_time_ms
            )

            components = ComponentOutputsResponse(
                internal_talk=internal_talk,
                hints_for_next_turn=hints,
                error_detection=errors,
                metrics=response_metrics
            )

        # Build classification
        classification = PerceivedLatencyClassification(metrics.classification)

        return StreamingConversationResponse(
            turn_number=turn_number,
            timestamp=metrics.timestamp,
            user_text=user_text,
            llm_response=llm_response,
            latencies=latencies,
            components=components,
            classification=classification
        )

    @staticmethod
    def to_ndjson(response: StreamingConversationResponse) -> str:
        """Convert response to NDJSON format (single line JSON)"""
        import json
        return json.dumps(response.dict())


async def orchestrator_process_with_latency_tracking(
    orchestrator: Any,  # OrchestratorService instance
    session_id: str,
    turn_number: int,
    user_text: str,
    audio_data: Optional[bytes] = None
) -> StreamingConversationResponse:
    """
    Example: Process a request with full latency tracking

    This shows the complete flow of how to use the latency manager
    in the Orchestrator.

    Args:
        orchestrator: OrchestratorService instance
        session_id: Conversation session ID
        turn_number: Turn number
        user_text: User input text
        audio_data: Optional audio data

    Returns:
        StreamingConversationResponse with all metrics
    """
    # Get or create manager for this session
    manager = orchestrator.get_latency_manager(session_id)
    manager.start_turn(turn_number)

    # Create orchestration context
    context = OrchestrationContext(session_id, turn_number, manager)

    try:
        # ================= STT PHASE =================
        context.start_phase("stt")
        if audio_data:
            # Transcribe audio
            transcript = await orchestrator.transcribe_audio(audio_data)
            context.set_response_data(user_text=transcript, llm_response="")
        else:
            context.set_response_data(user_text=user_text, llm_response="")
        stt_time = context.stop_phase("stt")

        # ================= LLM PHASE =================
        context.start_phase("llm")
        llm_response = await orchestrator.generate_response(
            context.user_text,
            session_id=session_id,
            turn_number=turn_number
        )
        context.set_response_data(
            user_text=context.user_text,
            llm_response=llm_response
        )
        llm_time = context.stop_phase("llm")

        # ================= TTS PHASE =================
        context.start_phase("tts")
        response_audio = await orchestrator.synthesize_speech(llm_response)
        context.set_response_data(
            user_text=context.user_text,
            llm_response=llm_response,
            response_audio_bytes=len(response_audio) if response_audio else 0
        )
        tts_time = context.stop_phase("tts")

        # ================= COMPONENT GENERATION (PARALLEL) =================
        # These run in parallel and should complete within TTS window
        internal_talk = await orchestrator.generate_internal_talk(
            user_text=context.user_text,
            response=llm_response,
            session_id=session_id
        )

        hints = await orchestrator.generate_hints_for_next_turn(
            user_text=context.user_text,
            response=llm_response,
            session_id=session_id
        )

        errors = await orchestrator.detect_user_errors(
            user_text=context.user_text,
            session_id=session_id
        )

        # ================= FINALIZE =================
        metrics = await context.finalize(
            internal_talk=internal_talk,
            hints=hints,
            errors=errors
        )

        # Build streaming response
        response = StreamingResponseBuilder.from_metrics(
            turn_number=turn_number,
            user_text=context.user_text,
            llm_response=llm_response,
            metrics=metrics
        )

        return response

    except Exception as e:
        orchestrator.logger.error(f"Error processing turn {turn_number}: {e}")
        # Cleanup on error
        orchestrator.cleanup_latency_manager(session_id)
        raise


# Example streaming endpoint integration
async def example_streaming_endpoint_handler(
    orchestrator: Any,
    request: Any,
    response_generator
) -> None:
    """
    Example: Streaming endpoint that yields NDJSON events

    Usage in API Gateway:
        @router.post("/stream/process")
        async def stream_process(request: ConversationRequest):
            async def event_generator():
                # Get manager and process
                response = await orchestrator_process_with_latency_tracking(
                    orchestrator,
                    request.session_id,
                    request.turn_number,
                    request.text
                )
                # Yield as NDJSON
                yield StreamingResponseBuilder.to_ndjson(response) + "\n"

            return StreamingResponse(event_generator(), media_type="application/x-ndjson")
    """
    # Implementation would go in API Gateway router
    pass
