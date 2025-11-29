#!/usr/bin/env python3
"""
Simplified process_turn using Talker abstraction

This is a cleaner implementation that delegates the audio→text→audio pipeline
to the appropriate Talker (Internal or External), while the Orchestrator focuses
on session management, scenario context, and conversation storage.
"""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def process_turn_with_talker(
    orchestrator,
    audio_data: bytes,
    session_id: str,
    sample_rate: int = 16000,
    voice_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process conversation turn using Talker abstraction (SIMPLIFIED PIPELINE)

    Flow:
    1. Get session + scenario context
    2. Get conversation history
    3. **Delegate to Talker** (audio → transcript + response + audio)
    4. Save conversation turn
    5. Update session state
    6. Return response

    Args:
        orchestrator: ConversationOrchestrator instance
        audio_data: Input audio bytes
        session_id: Session identifier
        sample_rate: Audio sample rate (default 16000)
        voice_id: TTS voice ID (optional)

    Returns:
        Dict with:
            - success: bool
            - text: AI response text
            - audio: AI response audio bytes
            - transcript: User input transcript
            - talker: Which talker was used ("internal" or "external")
            - metrics: Processing metrics
    """

    start_time = time.time()
    if orchestrator.stats_tracker:
        orchestrator.stats_tracker.increment_total_turns()

    try:
        logger.info(f"🎤 Processing turn with Talker: session={session_id}, audio={len(audio_data)} bytes")

        # Check if Talker is available
        if not orchestrator.talker:
            logger.error("❌ Talker not initialized - falling back to legacy process_turn()")
            return await orchestrator.process_turn(
                audio_data=audio_data,
                session_id=session_id,
                sample_rate=sample_rate,
                voice_id=voice_id
            )

        # ==========================================
        # STEP 1: Get Session and Scenario Context
        # ==========================================
        session_data = await orchestrator.clients["session"].get_session(session_id)

        # Default system prompt
        system_prompt = """You are a helpful AI assistant. Answer questions concisely and accurately."""
        conversation_id = None
        conversation_history = []
        scenario_id = None

        if session_data:
            conversation_id = session_data.get("conversation_id")
            scenario_id = session_data.get("scenario_id")
            voice_id = voice_id or session_data.get("voice_id", None)

            # Get scenario system prompt
            if scenario_id:
                scenario_data = await orchestrator.clients["scenarios"].get_scenario(scenario_id)
                if scenario_data:
                    system_prompt = scenario_data.get("system_prompt", system_prompt)
                    logger.info(f"📝 Using scenario: {scenario_data.get('name', scenario_id)}")

            # Get conversation history
            if conversation_id:
                messages = await orchestrator.clients["conversation_store"].get_context(
                    conversation_id, limit=10
                )
                conversation_history = orchestrator._format_conversation_history(messages)
                logger.info(f"📚 Loaded {len(messages)} previous messages")
        else:
            logger.warning(f"⚠️ Session {session_id} not found, using defaults")
            voice_id = voice_id or None

        # ==========================================
        # STEP 2: Delegate to Talker
        # ==========================================
        logger.info(f"🎯 Delegating to {orchestrator.talker.name}...")

        talker_result = await orchestrator.talker.process_turn(
            audio_data=audio_data,
            sample_rate=sample_rate,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            voice_id=voice_id
        )

        if not talker_result.get("success"):
            logger.error(f"❌ Talker failed: {talker_result.get('error')}")
            if orchestrator.stats_tracker:
                orchestrator.stats_tracker.increment_failed_turns()
            return {
                "success": False,
                "error": talker_result.get("error", "Talker processing failed"),
                "session_id": session_id
            }

        # Extract Talker results
        transcript = talker_result.get("transcript", "")
        text_response = talker_result.get("text", "")
        audio_response = talker_result.get("audio")
        talker_name = talker_result.get("talker", "unknown")
        talker_metrics = talker_result.get("metrics", {})

        logger.info(f"✅ {orchestrator.talker.name} completed: {transcript[:50]}... → {text_response[:50]}...")

        # ==========================================
        # STEP 3: Save Conversation Turn
        # ==========================================
        if conversation_id:
            try:
                await orchestrator.clients["conversation_store"].add_turn(
                    conversation_id=conversation_id,
                    user_audio=audio_data,
                    user_text=transcript,
                    ai_text=text_response,
                    ai_audio=audio_response
                )
                logger.info(f"💾 Turn saved to conversation {conversation_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save turn: {e}")

        # ==========================================
        # STEP 3.5: Metrics recording removed
        # (metrics_persister no longer available)
        # ==========================================

        # ==========================================
        # STEP 4: Update Session State
        # ==========================================
        if session_data:
            try:
                await orchestrator.clients["session"].update_session_llm(session_id, talker_name)
            except Exception as e:
                logger.warning(f"⚠️ Failed to update session: {e}")

        # ==========================================
        # STEP 5: Return Response
        # ==========================================
        total_time = time.time() - start_time
        if orchestrator.stats_tracker:
            orchestrator.stats_tracker.increment_successful_turns()
            orchestrator.stats_tracker.add_processing_time(total_time)
            # All services are external
            orchestrator.stats_tracker.increment_fallback_llm_count()

        response = {
            "success": True,
            "text": text_response,
            "audio": audio_response,
            "transcript": transcript,
            "session_id": session_id,
            "llm_used": talker_name,
            "voice_id": voice_id,
            "talker": talker_name,
            "metrics": {
                **talker_metrics,
                "total_orchestrator_time_ms": int(total_time * 1000),
                "input_audio_size": len(audio_data),
                "output_audio_size": len(audio_response) if audio_response else 0
            }
        }

        logger.info(f"✅ Turn completed in {total_time:.2f}s using {talker_name} Talker")
        return response

    except Exception as e:
        logger.error(f"❌ Orchestrator error: {e}", exc_info=True)
        if orchestrator.stats_tracker:
            orchestrator.stats_tracker.increment_failed_turns()
        return {
            "success": False,
            "error": f"Orchestration failed: {str(e)}",
            "session_id": session_id
        }
