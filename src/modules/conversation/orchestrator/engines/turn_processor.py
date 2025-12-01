"""
Turn Processor Engine

Processes complete conversation turns, coordinating STT, LLM, and TTS.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..constants import (
    DEFAULT_MASTERY_PROBABILITY,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SYSTEM_PROMPT,
    HIGH_CONFIDENCE_THRESHOLD,
)
from ..strategies import (
    HTTPLLMStrategy,
    HTTPTTSStrategy,
    LLMStrategyFactory,
    TTSStrategyFactory,
)

logger = logging.getLogger(__name__)


class TurnProcessor:
    """Processes conversation turns end-to-end."""

    def __init__(
        self,
        clients: dict[str, Any],
        fallback_manager: Any,
        context_loader: Any,
        knowledge_analyzer: Any,
        stats_tracker: Any,
        get_relevant_skills_func: Any | None = None,
    ) -> None:
        """
        Initialize turn processor.

        Args:
            clients: Dictionary of service clients
            fallback_manager: LLM fallback manager
            context_loader: Context loader engine
            knowledge_analyzer: Knowledge analyzer engine
            stats_tracker: Stats tracker engine
            get_relevant_skills_func: Function to get relevant skills
        """
        self.clients = clients
        self.fallback_manager = fallback_manager
        self.context_loader = context_loader
        self.knowledge_analyzer = knowledge_analyzer
        self.stats_tracker = stats_tracker
        self._get_relevant_skills_func = get_relevant_skills_func

        # Initialize strategies (all services are external)
        self.llm_strategy = LLMStrategyFactory.create_strategy(
            fallback_manager=fallback_manager, stats_tracker=stats_tracker
        )
        self.tts_strategy = TTSStrategyFactory.create_strategy(tts_client=clients.get("tts"))

    async def process_turn(
        self,
        audio_data: bytes,
        session_id: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        voice_id: str | None = None,
        force_external_llm: bool = False,
    ) -> dict[str, Any]:
        """
        Process complete conversation turn.

        Args:
            audio_data: Input audio bytes
            session_id: Session identifier
            sample_rate: Audio sample rate
            voice_id: TTS voice ID
            force_external_llm: Force use of external LLM

        Returns:
            Dictionary with success, text, audio, transcript, llm_used, metrics
        """
        start_time = time.time()
        self.stats_tracker.increment_total_turns()

        try:
            logger.info(
                f"🎤 Processing conversation turn: session={session_id}, audio={len(audio_data)} bytes"
            )

            # Step 1: Load context
            session_data = await self.clients["session"].get_session(session_id)
            (
                scenario_data,
                conversation_history,
                student_cefr_progress,
                target_skill,
                conversation_id,
                scenario_id,
                user_id,
            ) = await self.context_loader.load_context(session_id, session_data)

            # Update voice_id from session if not provided
            if session_data and not voice_id:
                voice_id = session_data.get("voice_id")

            # Get system prompt
            system_prompt = DEFAULT_SYSTEM_PROMPT
            if scenario_data:
                system_prompt = scenario_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

            # Step 2: Transcribe audio
            user_transcript = await self._transcribe_audio(
                audio_data, sample_rate, force_external_llm
            )

            # Step 3: Analyze turn (before generating response)
            turn_analysis, skills_extraction = await self._analyze_turn(
                user_transcript, student_cefr_progress
            )

            # Step 4: Update knowledge (before prompt composition)
            if turn_analysis and target_skill:
                await self._update_knowledge_before_response(
                    user_id, target_skill, user_transcript, turn_analysis, skills_extraction
                )

            # Step 5: Compose pedagogical prompt
            if "pedagogical_policy" in self.clients and (
                scenario_data or student_cefr_progress or target_skill
            ):
                system_prompt = await self._compose_pedagogical_prompt(
                    scenario_data,
                    student_cefr_progress,
                    target_skill,
                    conversation_history,
                    turn_analysis,
                )

            # Step 6: Generate LLM response
            text_response, llm_used, llm_result = await self._generate_llm_response(
                audio_data,
                sample_rate,
                system_prompt,
                conversation_history,
                conversation_id,
                force_external_llm,
            )

            if not llm_result.get("success"):
                self.stats_tracker.increment_failed_turns()
                return {
                    "success": False,
                    "error": llm_result.get("error", "LLM processing failed"),
                    "session_id": session_id,
                }

            # Update transcript if not already set
            if not user_transcript:
                user_transcript = llm_result.get("transcript", "")

            # Step 7: Synthesize audio
            audio_response = await self._synthesize_audio(text_response, voice_id)

            # Step 8: Save turn and update session (parallel)
            await self._save_turn_and_session(
                conversation_id,
                session_id,
                audio_data,
                user_transcript,
                text_response,
                audio_response,
                llm_used,
            )

            # Step 9: Return response
            processing_time = time.time() - start_time
            self.stats_tracker.increment_successful_turns()
            self.stats_tracker.add_processing_time(processing_time)

            response = {
                "success": True,
                "text": text_response,
                "audio": audio_response,
                "transcript": user_transcript,
                "session_id": session_id,
                "llm_used": llm_used,
                "voice_id": voice_id,
                "circuit_state": llm_result.get("circuit_state", {}),
                "metrics": {
                    "input_audio_size": len(audio_data),
                    "output_audio_size": len(audio_response) if audio_response else 0,
                    "processing_time_ms": int(processing_time * 1000),
                    "llm_used": llm_used,
                    "has_tts": audio_response is not None,
                },
            }

            logger.info(f"✅ Turn completed in {processing_time:.2f}s using {llm_used} LLM")
            return response

        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}", exc_info=True)
            self.stats_tracker.increment_failed_turns()
            return {
                "success": False,
                "error": f"Orchestration failed: {e!s}",
                "session_id": session_id,
            }

    async def _transcribe_audio(
        self, audio_data: bytes, sample_rate: int, force_external_llm: bool
    ) -> str:
        """Transcribe audio to text."""
        user_transcript = ""

        # Use HTTP STT (all services are external)
        if not user_transcript and "stt" in self.clients:
            try:
                stt_result = await self.clients["stt"].transcribe(audio_data, sample_rate)
                user_transcript = stt_result.get("text", "")
                logger.info(f"📝 Got transcript from STT service: {user_transcript[:50]}...")
            except Exception as e:
                logger.warning(f"⚠️ Failed to get transcript: {e}")

        return user_transcript

    async def _analyze_turn(
        self, user_transcript: str, student_cefr_progress: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Analyze turn for errors and skills."""
        if not user_transcript or "speech_grader" not in self.clients:
            return None, None

        try:
            current_cefr_level = "A1"
            if student_cefr_progress:
                current_cefr_level = (
                    student_cefr_progress.get("current_estimated_level")
                    or student_cefr_progress.get("current_level")
                    or "A1"
                )

            # Get valid skills
            valid_skills: list[str] = []
            if self._get_relevant_skills_func:
                valid_skills = self._get_relevant_skills_func(
                    user_transcript, current_cefr_level, "production"
                )
            else:
                try:
                    from src.modules.tutoring.student_model.skill_registry import SKILL_CEFR_MAP

                    for level_skills in SKILL_CEFR_MAP.values():
                        valid_skills.extend(level_skills)
                except Exception:
                    pass

            # Call both endpoints in parallel
            if valid_skills:
                try:
                    turn_analysis_task = self.clients["speech_grader"].analyze_turn(
                        user_text=user_transcript, ai_text=None, valid_skills=valid_skills
                    )
                    skills_extraction_task = self.clients["speech_grader"].extract_skills(
                        user_text=user_transcript, valid_skills=valid_skills, ai_text=None
                    )

                    turn_analysis, skills_extraction = await asyncio.gather(
                        turn_analysis_task, skills_extraction_task, return_exceptions=True
                    )

                    if isinstance(turn_analysis, Exception):
                        logger.error(f"⚠️ analyze_turn failed: {turn_analysis}")
                        turn_analysis = None
                    if isinstance(skills_extraction, Exception):
                        logger.error(f"⚠️ extract_skills failed: {skills_extraction}")
                        skills_extraction = None

                    # Combine results
                    if turn_analysis or skills_extraction:
                        combined_linguistic_features = {}
                        if skills_extraction and skills_extraction.get(
                            "overall_linguistic_features"
                        ):
                            combined_linguistic_features = skills_extraction.get(
                                "overall_linguistic_features", {}
                            )
                        elif turn_analysis:
                            combined_linguistic_features = turn_analysis.get(
                                "linguistic_features", {}
                            )

                        combined_skills_with_confidence: dict[str, float] = {}
                        if turn_analysis:
                            for skill_id in turn_analysis.get("correct_skills", []):
                                combined_skills_with_confidence[skill_id] = 1.0
                        if skills_extraction:
                            for skill_data in skills_extraction.get("skills", []):
                                skill_id = skill_data.get("skill_id")
                                confidence = skill_data.get("confidence", 0.0)
                                if skill_id and confidence >= HIGH_CONFIDENCE_THRESHOLD:
                                    combined_skills_with_confidence[skill_id] = max(
                                        combined_skills_with_confidence.get(skill_id, 0.0),
                                        confidence,
                                    )

                        if turn_analysis:
                            turn_analysis["linguistic_features"] = combined_linguistic_features
                            turn_analysis["correct_skills"] = list(
                                combined_skills_with_confidence.keys()
                            )

                    logger.info(
                        f"🔍 Analyzed turn: {len(turn_analysis.get('errors', [])) if turn_analysis else 0} errors"
                    )
                    return turn_analysis, skills_extraction

                except Exception as e:
                    logger.warning(f"⚠️ Failed to run parallel analysis: {e}")
                    try:
                        turn_analysis = await self.clients["speech_grader"].analyze_turn(
                            user_text=user_transcript, ai_text=None, valid_skills=valid_skills
                        )
                        return turn_analysis, None
                    except Exception as e2:
                        logger.warning(f"⚠️ Fallback analyze_turn also failed: {e2}")
                        return None, None
            else:
                turn_analysis = await self.clients["speech_grader"].analyze_turn(
                    user_text=user_transcript, ai_text=None, valid_skills=None
                )
                return turn_analysis, None

        except Exception as e:
            logger.warning(f"⚠️ Failed to analyze turn: {e}")
            return None, None

    async def _update_knowledge_before_response(
        self,
        user_id: str,
        target_skill: dict[str, Any],
        user_transcript: str,
        turn_analysis: dict[str, Any],
        skills_extraction: dict[str, Any] | None,
    ) -> None:
        """Update student knowledge before generating response."""
        if "student_model" not in self.clients:
            return

        try:
            linguistic_features = turn_analysis.get("linguistic_features", {})
            semantic_skill_mapping = turn_analysis.get("semantic_skill_mapping", {})
            correct_skills = turn_analysis.get("correct_skills", [])

            # Process successes
            success_skills_with_confidence: dict[str, float] = {}
            for skill_id in correct_skills:
                success_skills_with_confidence[skill_id] = 1.0
            for skill_id, confidence in semantic_skill_mapping.items():
                if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                    success_skills_with_confidence[skill_id] = max(
                        success_skills_with_confidence.get(skill_id, 0.0), confidence
                    )
            if skills_extraction:
                for skill_data in skills_extraction.get("skills", []):
                    skill_id = skill_data.get("skill_id")
                    confidence = skill_data.get("confidence", 0.0)
                    if skill_id and confidence >= HIGH_CONFIDENCE_THRESHOLD:
                        success_skills_with_confidence[skill_id] = max(
                            success_skills_with_confidence.get(skill_id, 0.0), confidence
                        )

            # Update student model for successes
            for usage_skill_id in success_skills_with_confidence:
                await self.clients["student_model"].assess(
                    user_id=user_id,
                    skill_id=usage_skill_id,
                    correct=True,
                    context={"type": "explicit_success", "source": "pre_response_analysis"},
                    user_text=user_transcript,
                    ai_text=None,
                    difficulty=None,
                    linguistic_features=linguistic_features,
                )

            # Process errors
            errors = turn_analysis.get("errors", [])
            for error in errors:
                error_skill_id = error.get("skill_id")
                if error_skill_id:
                    error_linguistic_features = error.get(
                        "linguistic_features", linguistic_features
                    )
                    await self.clients["student_model"].assess(
                        user_id=user_id,
                        skill_id=error_skill_id,
                        correct=False,
                        context={
                            "type": "error",
                            "error_type": error.get("error_type"),
                            "explanation": error.get("explanation"),
                            "severity": error.get("severity"),
                        },
                        user_text=user_transcript,
                        ai_text=None,
                        difficulty=None,
                        linguistic_features=error_linguistic_features,
                    )

            logger.info(
                f"📊 Updated knowledge: {len(success_skills_with_confidence)} successes, {len(errors)} errors"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to update knowledge: {e}")

    async def _compose_pedagogical_prompt(
        self,
        scenario_data: dict[str, Any] | None,
        student_cefr_progress: dict[str, Any] | None,
        target_skill: dict[str, Any] | None,
        conversation_history: list[dict[str, Any]],
        turn_analysis: dict[str, Any] | None,
    ) -> str:
        """Compose pedagogical prompt based on context."""
        try:
            interpretable_state = None
            if "student_model" in self.clients:
                user_id = student_cefr_progress.get("user_id") if student_cefr_progress else None
                if user_id:
                    try:
                        interpretable_state = await self.clients[
                            "student_model"
                        ].get_interpretable_knowledge_state(user_id)
                    except Exception:
                        pass

            session_analysis = None
            if conversation_history and len(conversation_history) >= 2:
                if interpretable_state and interpretable_state.get("linguistic_error_patterns"):
                    session_analysis = {
                        "historical_patterns": interpretable_state.get("linguistic_error_patterns"),
                        "session_context": f"Conversa com {len(conversation_history)} mensagens anteriores",
                    }

            prompt_context_dict = {
                "scenario": scenario_data,
                "cefr_level": (
                    student_cefr_progress.get("current_estimated_level", "A1")
                    if student_cefr_progress
                    else "A1"
                ),
                "cefr_details": (
                    student_cefr_progress.get("cefr_details", {}) if student_cefr_progress else {}
                ),
                "native_language": "pt",
                "target_skill": target_skill,
                "mastery_probability": (
                    target_skill.get("mastery_probability", DEFAULT_MASTERY_PROBABILITY)
                    if target_skill
                    else DEFAULT_MASTERY_PROBABILITY
                ),
                "emotional_state": "neutral",
                "conversation_history": conversation_history,
                "interpretable_knowledge_state": interpretable_state,
                "current_turn_analysis": turn_analysis,
                "session_analysis": session_analysis,
            }

            prompt_result = await self.clients["pedagogical_policy"].compose_prompt(
                prompt_context_dict
            )
            if prompt_result and prompt_result.get("prompt"):
                logger.info(
                    f"📚 Composed pedagogical prompt (strategy: {prompt_result.get('strategy', 'unknown')})"
                )
                return prompt_result["prompt"]
            else:
                logger.warning("⚠️ Failed to compose pedagogical prompt, using default")
                return DEFAULT_SYSTEM_PROMPT
        except Exception as e:
            logger.warning(f"⚠️ Error composing pedagogical prompt: {e}, using default")
            return DEFAULT_SYSTEM_PROMPT

    async def _generate_llm_response(
        self,
        audio_data: bytes,
        sample_rate: int,
        system_prompt: str,
        conversation_history: list[dict[str, Any]],
        conversation_id: str | None,
        force_external_llm: bool,
    ) -> tuple[str, str, dict[str, Any]]:
        """Generate LLM response using strategy pattern."""
        # Use HTTP strategy (all services are external)
        http_strategy = HTTPLLMStrategy(self.fallback_manager, self.stats_tracker)
        return await http_strategy.process_audio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            conversation_id=conversation_id,
            force_external_llm=force_external_llm,
        )

    async def _synthesize_audio(self, text_response: str, voice_id: str | None) -> bytes | None:
        """Synthesize audio from text using strategy pattern."""
        # Use HTTP strategy (all services are external)
        if "tts" in self.clients:
            http_tts_strategy = HTTPTTSStrategy(self.clients["tts"])
            return await http_tts_strategy.synthesize(text_response, voice_id)

        return None

    async def _save_turn_and_session(
        self,
        conversation_id: str | None,
        session_id: str,
        audio_data: bytes,
        user_transcript: str,
        text_response: str,
        audio_response: bytes | None,
        llm_used: str,
    ) -> None:
        """Save turn and update session in parallel."""
        save_tasks = []

        if conversation_id:
            save_tasks.append(
                self.clients["conversation_store"].add_turn(
                    conversation_id=conversation_id,
                    user_audio=audio_data,
                    user_text=user_transcript,
                    ai_text=text_response,
                    ai_audio=audio_response,
                )
            )

        save_tasks.append(self.clients["session"].update_session_llm(session_id, llm_used))

        if save_tasks:
            save_results = await asyncio.gather(*save_tasks, return_exceptions=True)
            task_idx = 0
            if conversation_id:
                if isinstance(save_results[task_idx], Exception):
                    logger.warning(f"⚠️ Failed to save turn: {save_results[task_idx]}")
                else:
                    logger.info(f"💾 Turn saved to conversation {conversation_id}")
                task_idx += 1
            if isinstance(save_results[task_idx], Exception):
                logger.warning(f"⚠️ Failed to update session: {save_results[task_idx]}")
            else:
                logger.debug(f"✅ Session {session_id} updated with LLM: {llm_used}")
