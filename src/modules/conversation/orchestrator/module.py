"""
Orchestrator Module - Direct Python calls for conversation orchestration
"""

from typing import Any

from src.modules.base_module import BaseModule


class OrchestratorModule(BaseModule):
    """Orchestrator Module for direct Python calls"""

    def __init__(self):
        super().__init__("orchestrator")
        self.orchestrator = None

    async def _initialize(self) -> bool:
        """Initialize orchestrator engine"""
        try:
            # Import orchestrator engine (now in modules)
            from .engine import ConversationOrchestrator

            # Create orchestrator (all services are external)
            self.orchestrator = ConversationOrchestrator()
            await self.orchestrator.initialize()

            self.logger.info("✅ Orchestrator Module initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Orchestrator Module: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    async def process_turn(
        self,
        session_id: str,
        audio_base64: str | None = None,
        text: str | None = None,
        language: str = "pt",
        voice_id: str | None = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        voice_speed: float = 1.0,
        stt_model: str = "whisper-large-v3",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Process a conversation turn

        Args:
            session_id: Session ID
            audio_base64: Base64 encoded audio (optional)
            text: Text input (optional, if audio not provided)
            language: Language code
            voice_id: Voice ID for TTS
            max_tokens: Max tokens for LLM
            temperature: Temperature for LLM
            voice_speed: Voice speed for TTS
            stt_model: STT model to use
            **kwargs: Additional parameters

        Returns:
            Dict with response_audio, response_text, etc.
        """
        if not self.initialized:
            await self.initialize()

        try:
            import base64

            # Process turn using orchestrator
            if audio_base64:
                # Decode base64 to bytes
                audio_data = base64.b64decode(audio_base64)
                sample_rate = 16000  # Default sample rate

                # Call orchestrator.process_turn with audio_data
                result = await self.orchestrator.process_turn(
                    audio_data=audio_data,
                    session_id=session_id,
                    sample_rate=sample_rate,
                    voice_id=voice_id,
                    force_external_llm=False,
                )

                # Convert response audio to base64 if needed
                if "audio" in result and isinstance(result["audio"], bytes):
                    result["audio_base64"] = base64.b64encode(result["audio"]).decode("utf-8")
                    result.pop("audio", None)

                return result
            elif text:
                # For text input, use process_text_conversation
                result = await self.orchestrator.process_text_conversation(
                    message=text, session_id=session_id, voice_id=voice_id
                )
                return result
            else:
                raise ValueError("Either audio_base64 or text must be provided")
        except Exception as e:
            self.logger.error(f"❌ Turn processing failed: {e}")
            raise

    async def process_text_conversation(
        self,
        message: str,
        session_id: str,
        voice_id: str | None = None,
        scenario_id_override: str | None = None,
    ) -> dict[str, Any]:
        """Process text conversation"""
        if not self.initialized:
            await self.initialize()

        try:
            result = await self.orchestrator.process_text_conversation(
                message=message,
                session_id=session_id,
                voice_id=voice_id,
                scenario_id_override=scenario_id_override,
            )
            return result
        except Exception as e:
            self.logger.error(f"❌ Text conversation failed: {e}")
            raise

    async def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics"""
        if self.orchestrator:
            return self.orchestrator.get_stats()
        return {}
