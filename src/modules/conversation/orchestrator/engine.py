#!/usr/bin/env python3
"""
Orchestrator Engine - Core conversation orchestration logic
Moved from LocalConversationPipeline to be a proper service

This is the brain of the system that:
- Coordinates all services (LLM, TTS, STT, Session, Scenarios, ConversationStore)
- Manages failover between primary and fallback LLMs
- Preserves conversation context
- Handles session lifecycle
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

import aiohttp

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .clients import ServiceClientError, create_service_clients
from .constants import (
    AUDIO_INT16_MAX,
    AUDIO_INT16_MIN,
    DEFAULT_CONVERSATION_HISTORY_URL,
    DEFAULT_CONVERSATION_STORE_URL,
    DEFAULT_EXTERNAL_ULTRAVOX_URL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TEXT_SYSTEM_PROMPT,
    ENV_CONVERSATION_HISTORY_URL,
    ENV_CONVERSATION_STORE_URL,
    ENV_ORCHESTRATOR_EXTERNAL_ULTRAVOX_URL,
    ENV_ORCHESTRATOR_SKIP_HEALTH_CHECKS,
    HEURISTIC_ANALYSIS_CONFIDENCE,
    HEURISTIC_COMPLEX_QUESTION_ESTIMATED_TURNS,
    HEURISTIC_CONFUSION_DETECTION_THRESHOLD,
    HEURISTIC_DEFAULT_ESTIMATED_TURNS,
    HEURISTIC_FALLBACK_CONFIDENCE,
    HEURISTIC_LONG_RESPONSE_THRESHOLD,
    HEURISTIC_PROMPT_PREFIX_TRUNCATE_LENGTH,
    HEURISTIC_SHORT_LLM_OUTPUT_THRESHOLD,
    HEURISTIC_SHORT_RESPONSE_ESTIMATED_TURNS,
    HEURISTIC_SHORT_RESPONSE_THRESHOLD,
    MAXIMUM_AUDIO_SIZE_MB,
    MINIMUM_AUDIO_DURATION_MS,
    MINIMUM_AUDIO_SAMPLES,
    VALID_SKILLS_CACHE_TTL_SECONDS,
    ContextType,
    StatsKey,
)
from .engines import ContextLoader, HealthChecker, KnowledgeAnalyzer, StatsTracker, TurnProcessor
from .fallback_manager import FallbackManager
from .process_turn_with_talker import process_turn_with_talker
from .talkers import AbstractTalker, TalkerFactory
from .types import (
    HealthStatus,
    ServiceConfig,
    StatsDict,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.shared.models.response_models import (
        AdaptiveInstructions,
        ConversationAnalysis,
        ErrorCorrection,
    )

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """
    Main orchestration engine for conversation processing

    Responsibilities:
    1. Coordinate multi-service conversation turns
    2. Manage LLM failover (primary → fallback)
    3. Preserve conversation context across turns
    4. Track session state
    5. Save conversation history
    """

    # Cache para valid_skills (evita recalcular a cada turno)
    _valid_skills_cache: list[str] | None = None
    _valid_skills_cache_timestamp: float | None = None
    _valid_skills_cache_ttl: float = VALID_SKILLS_CACHE_TTL_SECONDS

    def __init__(self, config: ServiceConfig | None = None) -> None:
        """
        Initialize orchestrator with service clients

        Args:
            config: Optional service URL overrides
        """
        # Get service URLs from environment or use defaults
        self.config = config or {}
        self._load_config_from_env()

        # Service clients (will be initialized later)
        self.clients: dict[str, Any] = {}
        self.http_session: aiohttp.ClientSession | None = None

        # Fallback manager (will be initialized later)
        self.fallback_manager: FallbackManager | None = None

        # Talker abstraction (will be initialized later)
        self.talker: AbstractTalker | None = None

        # Engines (will be initialized later)
        self.stats_tracker: StatsTracker | None = None
        self.health_checker: HealthChecker | None = None
        self.knowledge_analyzer: KnowledgeAnalyzer | None = None
        self.context_loader: ContextLoader | None = None
        self.turn_processor: TurnProcessor | None = None

        logger.info("🏗️ ConversationOrchestrator created - Mode: HTTP (cloud APIs)")

        # Import get_skill_difficulty once (for use in loops)
        self._get_skill_difficulty: Callable[[str], float | None] | None = None
        self._get_relevant_skills_func: Callable[[str, str, str], list[str]] | None = None
        try:
            from src.modules.tutoring.student_model.skill_registry import (
                get_relevant_skills_for_context,
                get_skill_difficulty,
            )

            self._get_skill_difficulty = get_skill_difficulty
            self._get_relevant_skills_func = (
                get_relevant_skills_for_context  # Renamed to avoid conflict
            )
        except ImportError:
            # Fallback: skill registry not available
            logger.warning("⚠️  Skill registry not available, using fallback functions")
            self._get_skill_difficulty = None
            self._get_relevant_skills_func = None
        except Exception as e:
            logger.warning(f"Could not import skill registry functions: {e}")
            self._get_skill_difficulty = None
            self._get_relevant_skills_func = None

    def _load_config_from_env(self) -> None:
        """Load service URLs from environment variables (only for external services)"""
        # Note: Module services (llm, tts, stt, session, scenarios) use direct calls, no URLs needed
        # Only external services need URLs
        env_mappings: dict[str, tuple[str, str]] = {
            "external_ultravox_url": (
                ENV_ORCHESTRATOR_EXTERNAL_ULTRAVOX_URL,
                DEFAULT_EXTERNAL_ULTRAVOX_URL,
            ),
            "conversation_store_url": (ENV_CONVERSATION_STORE_URL, DEFAULT_CONVERSATION_STORE_URL),
            "conversation_history_url": (
                ENV_CONVERSATION_HISTORY_URL,
                DEFAULT_CONVERSATION_HISTORY_URL,
            ),
        }

        for key, (env_var, default) in env_mappings.items():
            if key not in self.config:
                self.config[key] = os.getenv(env_var, default)

        logger.info("🏗️  Orchestrator using direct module calls (monolithic modular architecture)")

    def _get_relevant_skills_for_context(
        self,
        user_text: str,
        cefr_level: str,
        context_type: str = ContextType.PRODUCTION,
        force_refresh: bool = False,
    ) -> list[str]:
        """
        Get relevant skills for context with caching

        Args:
            user_text: Texto do aluno
            cefr_level: Nível CEFR atual
            context_type: Tipo de contexto ("production", "comprehension", "interaction")
            force_refresh: Forçar atualização do cache

        Returns:
            Lista filtrada de skill_ids relevantes
        """
        # Se temos a função importada, usar ela
        if self._get_relevant_skills_func:
            return self._get_relevant_skills_func(user_text, cefr_level, context_type)

        # Fallback: usar cache simples de todas as skills
        current_time = time.time()

        if (
            not force_refresh
            and self._valid_skills_cache is not None
            and self._valid_skills_cache_timestamp is not None
            and current_time - self._valid_skills_cache_timestamp < self._valid_skills_cache_ttl
        ):
            return self._valid_skills_cache

        # Recalcular cache
        try:
            from src.modules.tutoring.student_model.skill_registry import SKILL_CEFR_MAP

            valid_skills = []
            for level_skills in SKILL_CEFR_MAP.values():
                valid_skills.extend(level_skills)

            self._valid_skills_cache = valid_skills
            self._valid_skills_cache_timestamp = current_time
            return valid_skills
        except Exception as e:
            logger.warning(f"Could not fetch valid skills: {e}")
            return []

    async def initialize(self) -> None:
        """
        Initialize HTTP session and all service clients.

        Must be called before processing any requests.

        Raises:
            Exception: If initialization fails critically
        """
        logger.info("🚀 Initializing ConversationOrchestrator...")

        # ==========================================
        # Initialize service clients (direct calls for modules, HTTP for external services)
        # ==========================================
        logger.info("🔌 Initializing service clients...")

        # Create shared HTTP session only for external HTTP services (not for module services)
        from src.core.http_client import HTTPClient

        self.http_session = await HTTPClient.get_session()

        # Create all service clients
        # Module services will use direct calls, external services will use HTTP
        self.clients = create_service_clients(self.config)

        # Initialize each client
        # Module services don't need HTTP session, external services do
        for name, client in self.clients.items():
            if client.is_module_service:
                # Module services use direct calls, no HTTP session needed
                await client.initialize(None)
            else:
                # External services need HTTP session
                await client.initialize(self.http_session)

        # Create fallback manager with 2-tier failover
        self.fallback_manager = FallbackManager(
            primary_llm=self.clients["llm"], secondary_llm=self.clients["external_ultravox"]
        )

        # Initialize engines
        self.stats_tracker = StatsTracker()
        self.health_checker = HealthChecker(self.clients)
        self.knowledge_analyzer = KnowledgeAnalyzer(
            self.clients, get_skill_difficulty=self._get_skill_difficulty
        )
        self.context_loader = ContextLoader(self.clients)
        self.turn_processor = TurnProcessor(
            clients=self.clients,
            fallback_manager=self.fallback_manager,
            context_loader=self.context_loader,
            knowledge_analyzer=self.knowledge_analyzer,
            stats_tracker=self.stats_tracker,
            get_relevant_skills_func=self._get_relevant_skills_func,
        )

        # Health check all services (skip if ORCHESTRATOR_SKIP_HEALTH_CHECKS is set)
        skip_health_checks = (
            os.getenv(ENV_ORCHESTRATOR_SKIP_HEALTH_CHECKS, "false").lower() == "true"
        )
        if not skip_health_checks:
            await self._health_check_services()
        else:
            logger.info("🏁 Skipping health checks (startup mode - will run in background)")

        # Run profile-aware warmup
        await self._run_profile_warmup()

        # ==========================================
        # TALKER ABSTRACTION: Create appropriate Talker
        # ==========================================
        logger.info("🎯 Creating Talker (conversation pipeline abstraction)...")
        try:
            # Create Talker (all services are external, no GPU needed)
            self.talker = await TalkerFactory.create_talker(service_clients=self.clients)

            logger.info(f"✅ Talker ready: {self.talker.name}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to create Talker: {e}")
            logger.info("   Orchestrator will use TurnProcessor engine")
            self.talker = None

        logger.info("✅ ConversationOrchestrator initialized and ready - Mode: HTTP (cloud APIs)")

    async def _health_check_services(self) -> HealthStatus:
        """
        Check health of all downstream services.

        Returns:
            Dictionary mapping service names to health status (bool)
        """
        if self.health_checker:
            return await self.health_checker.check_all_services()
        # Fallback to old implementation if health_checker not initialized
        logger.info("🔍 Checking downstream services health...")
        health_status: HealthStatus = {}
        for name, client in self.clients.items():
            try:
                is_healthy = await client.health_check()
                health_status[name] = is_healthy
                status_icon = "✅" if is_healthy else "❌"
                logger.info(f"   {status_icon} {name}: {'healthy' if is_healthy else 'unhealthy'}")
            except Exception as e:
                health_status[name] = False
                logger.warning(f"   ❌ {name}: {e}")
        healthy_count = sum(1 for v in health_status.values() if v)
        total_count = len(health_status)
        logger.info(f"📊 Services health: {healthy_count}/{total_count} healthy")
        return health_status

    async def _run_profile_warmup(self) -> None:
        """
        Run profile-aware warmup for all services.

        Note: Currently skipped as services are independent.
        Each service handles its own warmup if needed.
        """
        logger.info("✅ Warmup skipped (services are independent)")

    async def process_turn(
        self,
        audio_data: bytes,
        session_id: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        voice_id: str | None = None,
        force_external_llm: bool = False,
    ) -> dict[str, Any]:
        """
        Process complete conversation turn - THE MAIN ORCHESTRATION METHOD

        Flow:
        1. Get session and scenario context
        2. Call LLM with failover (Ultravox → Groq)
        3. Generate TTS response
        4. Save conversation turn
        5. Update session state
        6. Return response

        Args:
            audio_data: Input audio bytes
            session_id: Session identifier
            sample_rate: Audio sample rate (default 16000)
            voice_id: TTS voice ID (optional)
            force_external_llm: Force use of external LLM (skip primary) for benchmarking

        Returns:
            Dict with:
                - success: bool
                - text: AI response text
                - audio: AI response audio bytes
                - transcript: User input transcript
                - llm_used: Which LLM was used
                - metrics: Processing metrics
        """
        if self.turn_processor:
            # Use new TurnProcessor engine
            result = await self.turn_processor.process_turn(
                audio_data=audio_data,
                session_id=session_id,
                sample_rate=sample_rate,
                voice_id=voice_id,
                force_external_llm=force_external_llm,
            )
            return result

        # TurnProcessor should always be initialized in initialize()
        raise RuntimeError(
            "TurnProcessor not initialized. Call initialize() before processing turns."
        )

    def _format_conversation_history(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Format conversation history for LLM context

        Args:
            messages: Raw messages from conversation store

        Returns:
            List of dicts with role and content
        """
        formatted = []
        for msg in messages:
            role = "user" if msg.get("sender") == "user" else "assistant"
            content = msg.get("text", msg.get("content", ""))
            if content:
                formatted.append({"role": role, "content": content})
        return formatted

    async def _analyze_and_update_knowledge(
        self, user_id: str, target_skill: dict[str, Any], user_text: str, ai_text: str
    ) -> None:
        """
        Background task: Analyze turn and update student knowledge.

        This runs asynchronously after the main response is sent,
        so it doesn't block the conversation flow.

        Args:
            user_id: User ID
            target_skill: Target skill being practiced
            user_text: User's input text
            ai_text: AI's response text
        """
        if self.knowledge_analyzer:
            await self.knowledge_analyzer.analyze_and_update_knowledge(
                user_id=user_id, target_skill=target_skill, user_text=user_text, ai_text=ai_text
            )
        else:
            logger.warning("KnowledgeAnalyzer not initialized, skipping knowledge update")

    async def get_services_health(self) -> HealthStatus:
        """
        Get health status of all services.

        Returns:
            Dictionary mapping service names to health status (bool)
        """
        return await self._health_check_services()

    async def get_fallback_health(self) -> HealthStatus:
        """
        Get health status of LLM failover components.

        Returns:
            Dictionary mapping component names to health status (bool)
        """
        if self.fallback_manager is None:
            return {}
        return self.fallback_manager.health_check()

    def get_stats(self) -> StatsDict:
        """
        Get orchestrator statistics including controller integration metrics

        Returns:
            Dict with processing stats, controller integration, and data flow info
        """
        if not self.stats_tracker:
            return {}

        stats = self.stats_tracker.get_stats()
        avg_time = self.stats_tracker.get_average_processing_time_ms() / 1000
        success_rate = self.stats_tracker.get_success_rate()
        primary_llm_rate = self.stats_tracker.get_primary_llm_rate()

        return {
            **stats,
            "average_processing_time_ms": int(avg_time * 1000),
            "success_rate": success_rate,
            "primary_llm_rate": primary_llm_rate,
            # Controller Integration Metrics
            "controller_integration": {
                "entry_points": [
                    "API Gateway (POST /process) - port 8888",
                    "WebRTC (POST /process) - port 8020",
                    "WebSocket (Socket.IO audio event) - port 8022",
                    "REST Polling (POST /api/session/{id}/audio) - port 8600",
                ],
                "controller_type": "ConversationController",
                "validation_format": "Ultravox LLM Format",
                "requirements": {
                    "audio_format": "Base64 encoded int16 PCM",
                    "sample_rate": f"{DEFAULT_SAMPLE_RATE} Hz (recommended)",
                    "minimum_duration": f"{MINIMUM_AUDIO_DURATION_MS}ms ({MINIMUM_AUDIO_SAMPLES} samples @ 16kHz)",
                    "maximum_size": f"{MAXIMUM_AUDIO_SIZE_MB} MB",
                },
            },
            # Data Flow Documentation
            "data_flow": {
                "description": "Audio data transformation pipeline",
                "stages": [
                    "1. Entry Point: Base64 string received from client",
                    "2. Controller: Validates Base64 format, sample rate, size",
                    "3. Controller: Decodes Base64 → raw bytes (int16 PCM)",
                    "4. Orchestrator: Converts bytes → numpy float32 array",
                    f"5. Orchestrator: Normalizes int16 [{AUDIO_INT16_MIN}, {AUDIO_INT16_MAX}] → float32 [-1.0, 1.0]",
                    "6. Ultravox LLM: Processes numpy array with <|audio|> placeholder",
                    "7. Ultravox LLM: Returns text transcript + AI response",
                    "8. TTS: Converts AI text → audio bytes",
                    "9. Response: Returns {transcript, text, audio} to client",
                ],
                "critical_note": "Controllers ONLY validate format. Audio conversion happens in Orchestrator.",
            },
            # Backend Mode Info
            "backend_mode": {
                "mode": "HTTP (cloud APIs)",
                "in_process_enabled": False,
                "http_fallback_calls": stats.get(StatsKey.HTTP_FALLBACK_COUNT, 0),
            },
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        if self.stats_tracker:
            self.stats_tracker.reset()
        logger.info("📊 Statistics reset")

    async def process_text_conversation(
        self,
        message: str,
        session_id: str,
        voice_id: str | None = None,
        scenario_id_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Process text conversation with optional audio output

        This is a lighter version of process_turn() for text-based chat.
        Uses external_llm for text-to-text processing.
        If voice_id is provided, generates audio response in Opus 24kHz format.

        Args:
            message: User text message
            session_id: Session identifier
            voice_id: Optional voice ID (if provided, generates Opus 24kHz audio)

        Returns:
            Dict with:
                - success: bool
                - response: AI response text
                - session_id: str
                - audio: Optional[str] - base64 encoded Opus 24kHz audio
                - context_size: int
                - messages_count: int
                - metrics: processing metrics
        """
        start_time = time.time()
        if self.stats_tracker:
            self.stats_tracker.increment_total_turns()

        try:
            logger.debug(
                f"💬 Processing text conversation: session={session_id}, message={message[:50]}..."
            )
            logger.info(
                f"💬 Processing text conversation: session={session_id}, message={message[:50]}..."
            )

            # ==========================================
            # STEP 1: Get Session and Scenario Context
            # ==========================================
            session_data = await self.clients["session"].get_session(session_id)
            system_prompt = DEFAULT_TEXT_SYSTEM_PROMPT
            conversation_id = None
            conversation_history = []
            scenario_id = None
            validation_result = None
            turn_number = 1

            if session_data:
                conversation_id = session_data.get("conversation_id")
                scenario_id = scenario_id_override or session_data.get("scenario_id")

                # Get scenario for system prompt (overrides default)
                if scenario_id:
                    logger.info(f"🔍 DEBUG: Fetching scenario with ID: {scenario_id}")
                    scenario_data = await self.clients["scenarios"].get_scenario(scenario_id)
                    logger.info(f"🔍 DEBUG: scenario_data = {scenario_data}")
                    if scenario_data:
                        base_system_prompt = scenario_data.get("system_prompt")
                        logger.info(f"📝 Using scenario: {scenario_data.get('name', scenario_id)}")

                        # Build scenario context for structured LLM
                        scenario_context = {
                            "type": scenario_data.get("type", "conversation"),
                            "expected_topics": scenario_data.get("expected_topics", []),
                            "ai_role": scenario_data.get("ai_role", "assistant"),
                            "user_role": scenario_data.get("user_role", "user"),
                            "language": scenario_data.get("language", "pt-BR"),
                            "system_prompt": base_system_prompt,
                        }

                        # Use system prompt (structured validation will happen in LLM call)
                        system_prompt = base_system_prompt

                # Get conversation history for context
                if conversation_id:
                    try:
                        messages = await self.clients["conversation_store"].get_context(
                            conversation_id, limit=10
                        )
                        # Convert to format expected by LLM
                        conversation_history = self._format_conversation_history(messages)
                        logger.info(f"📚 Loaded {len(messages)} previous messages for context")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get conversation context: {e}, continuing without history")
                        conversation_history = []

            else:
                # Session not found - use scenario_id_override if provided
                logger.warning(f"⚠️  Session {session_id} not found")
                scenario_id = scenario_id_override  # Use override even without session!

                try:
                    # Try to create new conversation (optional service)
                    conv_result = await self.clients["conversation_store"].create_conversation()
                    conversation_id = conv_result.get("conversation_id")
                    logger.info(f"✅ Created new conversation: {conversation_id}")

                    # Try to create new session (optional service)
                    session_result = await self.clients["session"].create_session(
                        conversation_id=conversation_id, session_id=session_id
                    )
                    created_session_id = session_result.get("id")
                    logger.info(f"✅ Created new session: {created_session_id}")

                except Exception as e:
                    # Session/Conversation services are optional - continue without them
                    logger.warning(
                        "⚠️  Session/Conversation services unavailable - continuing without history"
                    )
                    logger.debug(f"   Error details: {e}")
                    # Continue without conversation_id - won't save history but conversation will still work

                # Get scenario data if scenario_id is provided
                if scenario_id:
                    logger.info(f"🔍 DEBUG: Fetching scenario (no session) with ID: {scenario_id}")
                    scenario_data = await self.clients["scenarios"].get_scenario(scenario_id)
                    logger.info(f"🔍 DEBUG: scenario_data (no session) = {scenario_data}")
                    if scenario_data:
                        base_system_prompt = scenario_data.get("system_prompt")
                        logger.info(
                            f"📝 Using scenario (no session): {scenario_data.get('name', scenario_id)}"
                        )

                        # Build scenario context for structured LLM
                        scenario_context = {
                            "type": scenario_data.get("type", "conversation"),
                            "expected_topics": scenario_data.get("expected_topics", []),
                            "ai_role": scenario_data.get("ai_role", "assistant"),
                            "user_role": scenario_data.get("user_role", "user"),
                            "language": scenario_data.get("language", "pt-BR"),
                            "system_prompt": base_system_prompt,
                        }

                        # Use system prompt (structured validation will happen in LLM call)
                        system_prompt = base_system_prompt

            # ==========================================
            # STEP 2: Call LLM (Structured with validation OR simple text)
            # ==========================================
            logger.debug(
                f"Step 2: scenario_id={scenario_id}, session_data={session_data is not None}"
            )
            validation_metrics = None

            # DEBUG: Log scenario_id and scenario_context existence
            logger.debug(f"Checking scenario_context in locals: {'scenario_context' in locals()}")
            logger.info(
                f"🔍 DEBUG: scenario_id={scenario_id}, scenario_context_exists={'scenario_context' in locals()}"
            )
            if "scenario_context" in locals():
                logger.info(f"🔍 DEBUG: scenario_context keys={list(scenario_context.keys())}")

            # If scenario exists, use structured LLM (validation + response in one call)
            if scenario_id and "scenario_context" in locals():
                try:
                    logger.info("🤖 Using structured LLM (validation + response in single call)...")

                    # Initialize structured LLM client (lazy init)
                    if not hasattr(self, "structured_llm"):
                        from .structured_llm_client import StructuredLLMClient

                        self.structured_llm = StructuredLLMClient()
                        await self.structured_llm.initialize(self.http_session)

                    # Single LLM call with structured output (Pydantic validated!)
                    validated_response = await self.structured_llm.generate_with_validation(
                        user_message=message,
                        scenario_context=scenario_context,
                        conversation_history=conversation_history,
                    )

                    # Extract text response and validation
                    text_response = validated_response.assistant_response
                    validation_metrics = {
                        "coherence_score": validated_response.validation.coherence_score,
                        "in_scope": validated_response.validation.in_scope,
                        "should_redirect": validated_response.validation.should_redirect,
                        "found_topics": validated_response.validation.found_topics,
                        "missing_topics": validated_response.validation.missing_topics,
                        "reason": validated_response.validation.reason,
                    }

                    logger.info(
                        f"✅ Structured LLM response: "
                        f"coherence={validation_metrics['coherence_score']:.2f}, "
                        f"in_scope={validation_metrics['in_scope']}, "
                        f"redirect={validation_metrics['should_redirect']}"
                    )

                    if self.stats_tracker:
                        self.stats_tracker.increment_primary_llm_count()

                except Exception as e:
                    logger.warning(f"⚠️ Structured LLM failed: {e}, falling back to simple LLM")
                    # Fallback to simple external_llm if structured fails
                    text_response = await self.clients["llm"].generate(
                        text=message,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history,
                    )
                    logger.info(f"🤖 LLM response (fallback): {text_response[:100]}...")
                    if self.stats_tracker:
                        self.stats_tracker.increment_primary_llm_count()

            # No scenario - use simple external_llm
            else:
                try:
                    # Call external LLM service using generate() method
                    text_response = await self.clients["llm"].generate(
                        text=message,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history,
                    )

                    logger.info(f"🤖 LLM response: {text_response[:100]}...")
                    if self.stats_tracker:
                        self.stats_tracker.increment_primary_llm_count()

                except ServiceClientError as e:
                    logger.error(f"❌ External LLM failed: {e}")
                    if self.stats_tracker:
                        self.stats_tracker.increment_failed_turns()
                    return {
                        "success": False,
                        "error": f"LLM processing failed: {e!s}",
                        "response": "",
                        "session_id": session_id,
                    }

            # ==========================================
            # STEP 2.5: Generate Audio (TTS) if voice_id provided
            # ==========================================
            audio_base64 = None
            if voice_id:
                try:
                    logger.info(f"🔊 Generating audio with voice_id={voice_id} (Opus 24kHz)")

                    # Try local TTS first
                    audio_bytes = await self.clients["tts"].synthesize(
                        text=text_response,
                        voice_id=voice_id,
                        speed=1.0,
                        sample_rate=24000,
                        format="opus",
                    )

                    # Convert to base64 for JSON transport
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    logger.info(
                        f"✅ Audio (local TTS) generated: {len(audio_bytes)} bytes → {len(audio_base64)} base64 chars"
                    )

                except ServiceClientError as e:
                    logger.warning("⚠️  Local TTS service unavailable - trying external TTS")
                    logger.debug(f"   Local TTS error: {e}")

                    # Fallback to external TTS (HuggingFace)
                    try:
                        audio_bytes = await self.clients["tts"].synthesize(
                            text=text_response, voice="af_heart", format="wav"
                        )

                        # Convert to base64 for JSON transport
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                        logger.info(
                            f"✅ Audio (external TTS) generated: {len(audio_bytes)} bytes → {len(audio_base64)} base64 chars"
                        )

                    except ServiceClientError as e2:
                        logger.warning("⚠️  External TTS also failed - continuing without audio")
                        logger.debug(f"   External TTS error: {e2}")
                        # Continue without audio - don't fail the whole request

                except Exception as e:
                    logger.warning("⚠️  TTS service unavailable - continuing without audio")
                    logger.debug(f"   TTS error: {e}")
                    # Continue without audio

            # ==========================================
            # STEP 3: Save Conversation Turn
            # ==========================================
            if conversation_id:
                try:
                    await self.clients["conversation_store"].add_turn(
                        conversation_id=conversation_id,
                        user_audio=None,  # No audio in text mode
                        user_text=message,
                        ai_text=text_response,
                        ai_audio=None,  # No audio in text mode
                    )
                    logger.info(f"💾 Turn saved to conversation {conversation_id}")
                except ServiceClientError as e:
                    logger.warning(f"⚠️  Failed to save turn: {e}")

            # ==========================================
            # STEP 4: Return Response
            # ==========================================
            processing_time = time.time() - start_time
            if self.stats_tracker:
                self.stats_tracker.increment_successful_turns()
                self.stats_tracker.add_processing_time(processing_time)

            # Get updated message count
            messages_count = len(conversation_history) + 2  # +2 for current turn

            response = {
                "success": True,
                "response": text_response,
                "session_id": session_id,
                "audio": audio_base64,  # Opus 24kHz base64 encoded (if voice_id provided)
                "context_size": len(conversation_history),
                "messages_count": messages_count,
                "metrics": {"processing_time_ms": processing_time * 1000, "timestamp": time.time()},
            }

            # Add validation metrics if available (from structured LLM)
            if validation_metrics:
                response["validation"] = validation_metrics

            return response

        except Exception as e:
            logger.error(f"❌ Unexpected error in text conversation: {e}")
            logger.exception("Full traceback:")
            if self.stats_tracker:
                self.stats_tracker.increment_failed_turns()
            return {
                "success": False,
                "error": f"Internal error: {e!s}",
                "response": "",
                "session_id": session_id,
            }

    async def process_turn_structured(
        self,
        audio_data: bytes,
        session_id: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        voice_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process conversation turn with STRUCTURED VALIDATION (Pydantic)

        Flow:
        1. STT (External Groq Whisper)
        2. Single LLM call with structured output (validation + response in one)
        3. TTS (HTTP Service)

        This uses Pydantic models to guarantee JSON format without regex/parsing.
        The LLM returns validation metrics AND response in a single call.

        Args:
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
                - validation: Validation metrics (coherence, in_scope, etc)
                - metadata: Additional metadata (sentiment, intent, etc)
                - metrics: Processing metrics
        """

        from datetime import datetime, timezone

        start_time = time.time()
        if self.stats_tracker:
            self.stats_tracker.increment_total_turns()

        try:
            logger.info(
                f"🎤 Processing structured turn: session={session_id}, audio={len(audio_data)} bytes"
            )

            # ==========================================
            # STEP 1: STT (External Groq Whisper)
            # ==========================================
            logger.info("📝 Transcribing audio with External STT (Groq Whisper)...")

            try:
                transcribed_text = await self.clients["stt"].transcribe(
                    audio_data=audio_data, language="pt"  # or auto-detect from scenario
                )
                logger.info(f"✅ Transcribed: {transcribed_text[:100]}...")
            except ServiceClientError as e:
                logger.error(f"❌ STT failed: {e}")
                if self.stats_tracker:
                    self.stats_tracker.increment_failed_turns()
                return {"success": False, "error": f"STT failed: {e}", "session_id": session_id}

            # ==========================================
            # STEP 2: Get Scenario Context
            # ==========================================
            session_data = await self.clients["session"].get_session(session_id)
            scenario_id = session_data.get("scenario_id") if session_data else None
            conversation_id = session_data.get("conversation_id") if session_data else None
            voice_id = voice_id or (session_data.get("voice_id", None) if session_data else None)

            scenario_context = {}
            conversation_history = []

            if scenario_id:
                scenario_data = await self.clients["scenarios"].get_scenario(scenario_id)
                if scenario_data:
                    scenario_context = {
                        "type": scenario_data.get("type", "conversation"),
                        "expected_topics": scenario_data.get("expected_topics", []),
                        "ai_role": scenario_data.get("ai_role", "assistant"),
                        "user_role": scenario_data.get("user_role", "user"),
                        "language": scenario_data.get("language", "pt-BR"),
                        "system_prompt": scenario_data.get("system_prompt", ""),
                    }
                    logger.info(f"📝 Using scenario: {scenario_data.get('name')}")
            else:
                # Default context when no scenario
                scenario_context = {
                    "type": "general_conversation",
                    "expected_topics": [],
                    "ai_role": "helpful assistant",
                    "user_role": "user",
                    "language": "pt-BR",
                    "system_prompt": "You are a helpful AI assistant.",
                }

            # Get conversation history
            if conversation_id:
                messages = await self.clients["conversation_store"].get_context(
                    conversation_id, limit=10
                )
                conversation_history = self._format_conversation_history(messages)
                logger.info(f"📚 Loaded {len(messages)} previous messages for context")

            # ==========================================
            # STEP 3: Structured LLM (Validation + Response)
            # ==========================================
            logger.info("🤖 Calling structured LLM (validation + generation in one call)...")

            # Initialize structured LLM client (lazy init)
            if not hasattr(self, "structured_llm"):
                from .structured_llm_client import StructuredLLMClient

                self.structured_llm = StructuredLLMClient()
                await self.structured_llm.initialize(self.http_session)

            try:
                # Single LLM call with structured output (Pydantic validated!)
                validated_response = await self.structured_llm.generate_with_validation(
                    user_message=transcribed_text,
                    scenario_context=scenario_context,
                    conversation_history=conversation_history,
                )

                # Extract fields (typed, no parsing!)
                validation = validated_response.validation
                assistant_text = validated_response.assistant_response
                metadata = validated_response.metadata

                logger.info(
                    f"✅ Structured LLM response: "
                    f"coherence={validation.coherence_score:.2f}, "
                    f"in_scope={validation.in_scope}, "
                    f"redirect={validation.should_redirect}"
                )

                if validation.should_redirect:
                    logger.info(f"🔄 Redirection detected: {validation.reason}")

            except Exception as e:
                logger.error(f"❌ Structured LLM failed: {e}")
                if self.stats_tracker:
                    self.stats_tracker.increment_failed_turns()
                return {
                    "success": False,
                    "error": f"LLM processing failed: {e}",
                    "session_id": session_id,
                }

            # ==========================================
            # STEP 4: TTS (HTTP Service)
            # ==========================================
            logger.info("🔊 Generating audio with TTS service...")

            audio_response = None
            try:
                audio_response = await self.clients["tts"].synthesize(
                    text=assistant_text, voice_id=voice_id
                )
                logger.info(f"✅ TTS generated: {len(audio_response)} bytes")
            except ServiceClientError as e:
                logger.warning(f"⚠️ Local TTS failed: {e}, trying external...")

                # Fallback to external TTS
                try:
                    audio_response = await self.clients["external_tts"].synthesize(
                        text=assistant_text, voice="af_heart", format="wav"
                    )
                    logger.info(f"✅ External TTS generated: {len(audio_response)} bytes")
                except ServiceClientError as e2:
                    logger.error(f"❌ All TTS failed: {e2}")
                    # Continue without audio

            # ==========================================
            # STEP 5: Save Conversation Turn
            # ==========================================
            if conversation_id:
                try:
                    await self.clients["conversation_store"].add_turn(
                        conversation_id=conversation_id,
                        user_audio=audio_data,
                        user_text=transcribed_text,
                        ai_text=assistant_text,
                        ai_audio=audio_response,
                    )
                    logger.info(f"💾 Turn saved to conversation {conversation_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save turn: {e}")

            # ==========================================
            # STEP 6: Update Scenario State
            # ==========================================
            if scenario_id:
                try:
                    # Update state with validation metrics
                    await self.clients["scenarios"].update_scenario_state(
                        session_id=session_id,
                        validation_result={
                            "coherence_score": validation.coherence_score,
                            "in_scope": validation.in_scope,
                            "should_redirect": validation.should_redirect,
                            "found_topics": validation.found_topics,
                            "validated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    logger.info("📊 Scenario state updated")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update scenario state: {e}")

            # ==========================================
            # STEP 7: Return Response
            # ==========================================
            processing_time = time.time() - start_time
            if self.stats_tracker:
                self.stats_tracker.increment_successful_turns()
                self.stats_tracker.add_processing_time(processing_time)

            return {
                "success": True,
                "text": assistant_text,
                "audio": audio_response,
                "transcript": transcribed_text,
                "session_id": session_id,
                "voice_id": voice_id,
                "validation": {
                    "coherence_score": validation.coherence_score,
                    "in_scope": validation.in_scope,
                    "should_redirect": validation.should_redirect,
                    "found_topics": validation.found_topics,
                    "missing_topics": validation.missing_topics,
                    "reason": validation.reason,
                },
                "metadata": {
                    "sentiment": metadata.sentiment if metadata else None,
                    "intent": metadata.intent if metadata else None,
                    "language_quality": metadata.language_quality if metadata else None,
                    "confidence": metadata.confidence if metadata else None,
                },
                "metrics": {
                    "processing_time_ms": int(processing_time * 1000),
                    "stt_provider": "external_groq",
                    "llm_provider": "structured_groq_pydantic",
                    "tts_provider": "http_service",
                    "has_audio": audio_response is not None,
                },
            }

        except Exception as e:
            logger.error(f"❌ Structured turn error: {e}", exc_info=True)
            if self.stats_tracker:
                self.stats_tracker.increment_failed_turns()
            return {
                "success": False,
                "error": f"Orchestration failed: {e!s}",
                "session_id": session_id,
            }

    async def process_turn_with_talker(
        self,
        audio_data: bytes,
        session_id: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        voice_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process conversation turn using Talker abstraction (SIMPLIFIED VERSION)

        This method delegates the audio→text→audio pipeline to the appropriate
        Talker (cloud-based APIs), making the orchestration
        much simpler and cleaner.

        Args:
            audio_data: Input audio bytes
            session_id: Session identifier
            sample_rate: Audio sample rate (default 16000)
            voice_id: TTS voice ID (optional)

        Returns:
            Dict with success, text, audio, transcript, talker, and metrics
        """
        return await process_turn_with_talker(self, audio_data, session_id, sample_rate, voice_id)

    # ============================================================================
    # STREAMING METHODS WITH ADAPTIVE PARAMETERS (Phase 2: Streaming JSON)
    # ============================================================================

    async def process_turn_streaming(
        self,
        audio_data: bytes,
        session_id: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        voice_id: str | None = None,
    ):
        """
        Process conversation turn with streaming JSON output and adaptive parameters.

        Yields StreamingJSONEvent objects with:
        - text_chunk: LLM response chunks as they arrive
        - analysis: Conversation analysis (response_type, theme, tone)
        - adaptive_instructions: Instructions for next LLM response
        - error_correction: Error detection and correction suggestions
        - complete: Final completion event

        Args:
            audio_data: Input audio bytes
            session_id: Session identifier
            sample_rate: Audio sample rate (default 16000)
            voice_id: TTS voice ID (optional)

        Yields:
            StreamingJSONEvent: Streaming events with different data payloads
        """
        from src.core.shared.models.response_models import StreamingJSONEvent

        start_time = time.time()
        sequence = 0
        llm_text_buffer = ""
        user_transcript = ""

        try:
            logger.info(f"🎬 Starting streaming turn: session={session_id}")

            # ==========================================
            # STEP 1: Get session context (non-blocking)
            # ==========================================
            session_data = await self.clients["session"].get_session(session_id)
            if not session_data:
                logger.warning(f"⚠️ Session {session_id} not found")
                session_data = {"voice_id": voice_id}

            voice_id = voice_id or session_data.get("voice_id")
            conversation_history = []

            # Get conversation context if available
            conversation_id = session_data.get("conversation_id")
            if conversation_id:
                try:
                    messages = await self.clients["conversation_store"].get_context(
                        conversation_id, limit=5
                    )
                    conversation_history = self._format_conversation_history(messages)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load conversation history: {e}")

            # ==========================================
            # STEP 2: Transcribe audio (blocking, brief)
            # ==========================================
            stt_start = time.time()
            try:
                stt_result = await self.clients["stt"].transcribe(audio_data, sample_rate)
                user_transcript = stt_result.get("text", "")
                stt_duration = time.time() - stt_start
                logger.info(f"✅ STT: {user_transcript} ({stt_duration*1000:.0f}ms)")
            except Exception as e:
                logger.error(f"❌ STT failed: {e}")
                yield StreamingJSONEvent(
                    event="error",
                    sequence=sequence,
                    data={"error": "Speech-to-text failed", "details": str(e)},
                    is_final=True,
                )
                return

            # ==========================================
            # STEP 3: Stream LLM response (with chunks)
            # ==========================================
            sequence = 0
            llm_start = time.time()

            async for text_chunk in self._stream_llm_response(
                user_transcript, conversation_history, session_data
            ):
                llm_text_buffer += text_chunk
                sequence += 1

                # Yield text chunk
                yield StreamingJSONEvent(
                    event="text_chunk", sequence=sequence, data=text_chunk, timestamp=time.time()
                )

            llm_duration = time.time() - llm_start
            logger.info(
                f"✅ LLM: Generated {len(llm_text_buffer)} chars ({llm_duration*1000:.0f}ms)"
            )

            # ==========================================
            # STEP 4: Analyze response (async, in-parallel)
            # ==========================================
            analysis = await self._generate_analysis(user_transcript, llm_text_buffer)
            sequence += 1

            yield StreamingJSONEvent(
                event="analysis",
                sequence=sequence,
                data=analysis.model_dump(),
                timestamp=time.time(),
            )

            # ==========================================
            # STEP 5: Generate adaptive instructions
            # ==========================================
            adaptive_instructions = await self._generate_adaptive_instructions(
                user_transcript, llm_text_buffer, analysis
            )
            sequence += 1

            yield StreamingJSONEvent(
                event="adaptive_instructions",
                sequence=sequence,
                data=adaptive_instructions.model_dump(),
                timestamp=time.time(),
            )

            # ==========================================
            # STEP 6: Error correction detection
            # ==========================================
            error_correction = await self._detect_error_patterns(user_transcript, llm_text_buffer)
            if error_correction.correction_needed or error_correction.pattern_detected:
                sequence += 1
                yield StreamingJSONEvent(
                    event="error_correction",
                    sequence=sequence,
                    data=error_correction.model_dump(),
                    timestamp=time.time(),
                )

            # ==========================================
            # STEP 7: Complete event
            # ==========================================
            total_time = time.time() - start_time
            sequence += 1

            yield StreamingJSONEvent(
                event="complete",
                sequence=sequence,
                data={
                    "success": True,
                    "transcript": user_transcript,
                    "response": llm_text_buffer,
                    "total_time_ms": int(total_time * 1000),
                    "session_id": session_id,
                },
                timestamp=time.time(),
                is_final=True,
            )

            logger.info(f"✅ Streaming turn complete ({total_time*1000:.0f}ms total)")

        except Exception as e:
            logger.error(f"❌ Streaming turn error: {e}", exc_info=True)
            sequence += 1

            yield StreamingJSONEvent(
                event="error",
                sequence=sequence,
                data={"error": str(e)},
                timestamp=time.time(),
                is_final=True,
            )

    async def _stream_llm_response(
        self, user_input: str, history: list[dict[str, Any]], session_data: dict[str, Any]
    ) -> Any:  # Generator type would be AsyncGenerator[str, None] but keeping Any for compatibility
        """
        Stream LLM response as chunks.

        Yields text chunks as they arrive from LLM.
        Falls back to HTTP if in-process is not available.
        """
        try:
            # TODO: Implement streaming LLM response from service
            # For now, yield the full response at once (simulating chunks)
            if self.clients.get("llm"):
                result = await self.clients["llm"].call_conversation(
                    user_input, history, session_data
                )
                response_text = result.get("text", user_input)

                # Simulate chunking by yielding in parts
                chunk_size = 50
                for i in range(0, len(response_text), chunk_size):
                    yield response_text[i : i + chunk_size]
            else:
                logger.warning("⚠️ LLM client not available, using fallback")
                yield f"[Resposta simulada] {user_input}"

        except Exception as e:
            logger.error(f"❌ LLM streaming error: {e}")
            yield f"[Erro no LLM: {e!s}]"

    async def _generate_analysis(self, user_input: str, llm_output: str) -> ConversationAnalysis:
        """
        Generate ConversationAnalysis metadata from response.

        Analyzes the response to determine:
        - response_type: question, explanation, suggestion, confirmation, etc.
        - theme: main topic detected
        - tone: helpful, formal, casual, urgent, empathetic
        - confidence: 0-1 confidence in the analysis
        """
        from src.core.shared.models.response_models import ConversationAnalysis

        try:
            # Simple heuristic-based analysis (can be replaced with LLM)
            response_type = "explanation"
            if llm_output.endswith("?"):
                response_type = "question"
            elif len(llm_output) < HEURISTIC_SHORT_RESPONSE_THRESHOLD:
                response_type = "confirmation"
            elif any(word in llm_output.lower() for word in ["sugestão", "recomendo", "tente"]):
                response_type = "suggestion"

            # Detect theme from user input
            theme = "general"
            theme_keywords = {
                "transportation": ["táxi", "uber", "transporte", "ônibus", "carro", "car"],
                "weather": ["tempo", "chuva", "sol", "temperatura"],
                "food": ["comida", "restaurante", "pizza", "café"],
                "travel": ["viagem", "hotel", "passagem", "destino"],
            }

            for theme_key, keywords in theme_keywords.items():
                if any(kw in user_input.lower() for kw in keywords):
                    theme = theme_key
                    break

            # Determine tone
            tone = "helpful"
            if any(word in llm_output.lower() for word in ["desculpe", "lamento", "error"]):
                tone = "empathetic"
            elif any(word in llm_output.lower() for word in ["urgente", "rápido", "imediato"]):
                tone = "urgent"

            return ConversationAnalysis(
                response_type=response_type,
                theme=theme,
                tone=tone,
                confidence=HEURISTIC_ANALYSIS_CONFIDENCE,
            )

        except Exception as e:
            logger.error(f"⚠️ Analysis generation failed: {e}")
            from src.core.shared.models.response_models import ConversationAnalysis

            return ConversationAnalysis(
                response_type="other",
                theme="general",
                tone="helpful",
                confidence=HEURISTIC_FALLBACK_CONFIDENCE,
            )

    async def _generate_adaptive_instructions(
        self, user_input: str, llm_output: str, analysis: ConversationAnalysis
    ) -> AdaptiveInstructions:
        """
        Generate AdaptiveInstructions for next LLM response.

        Creates instructions to help the LLM adapt to:
        - Current conversation flow
        - Expected next topics
        - Tone adjustments
        - Verbosity level
        """
        from src.core.shared.models.response_models import AdaptiveInstructions

        try:
            # Predict next topics based on current context
            expected_topics = []
            if "location" in user_input.lower() or "where" in user_input.lower():
                expected_topics = ["destination", "time", "preferences"]
            elif "when" in user_input.lower() or "tempo" in user_input.lower():
                expected_topics = ["date", "time", "availability"]
            else:
                expected_topics = ["clarification", "details", "confirmation"]

            # Estimate turns remaining
            estimated_turns = HEURISTIC_DEFAULT_ESTIMATED_TURNS
            if len(llm_output) > HEURISTIC_LONG_RESPONSE_THRESHOLD:
                estimated_turns = HEURISTIC_SHORT_RESPONSE_ESTIMATED_TURNS  # Long response, likely nearing resolution
            elif len(user_input) > HEURISTIC_LONG_RESPONSE_THRESHOLD:
                estimated_turns = (
                    HEURISTIC_COMPLEX_QUESTION_ESTIMATED_TURNS  # Complex question, might need more
                )

            # Generate prefix for next prompt
            next_prompt_prefix = f"""Usuário anterior perguntou: {user_input[:HEURISTIC_PROMPT_PREFIX_TRUNCATE_LENGTH]}...
Você respondeu sobre: {analysis.theme}
Tom a manter: {analysis.tone}
Próximos tópicos esperados: {', '.join(expected_topics)}

Próxima resposta:"""

            return AdaptiveInstructions(
                next_prompt_prefix=next_prompt_prefix,
                tone_adjustment="maintain_current",
                verbosity=(
                    "medium" if len(llm_output) > HEURISTIC_LONG_RESPONSE_THRESHOLD else "concise"
                ),
                expected_next_topics=expected_topics,
                estimated_turns_remaining=estimated_turns,
            )

        except Exception as e:
            logger.error(f"⚠️ Adaptive instructions generation failed: {e}")
            from src.core.shared.models.response_models import AdaptiveInstructions

            return AdaptiveInstructions(
                next_prompt_prefix="Continue the conversation naturally.",
                tone_adjustment="maintain_current",
                verbosity="medium",
                expected_next_topics=["clarification"],
                estimated_turns_remaining=HEURISTIC_DEFAULT_ESTIMATED_TURNS,
            )

    async def _detect_error_patterns(self, user_input: str, llm_output: str) -> ErrorCorrection:
        """
        Detect common error patterns and suggest corrections.

        Detects:
        - location_format_error: Address format issues
        - misunderstanding: User seems confused
        - incomplete_answer: Response incomplete
        """
        from src.core.shared.models.response_models import ErrorCorrection

        try:
            pattern_detected = None
            suggested_clarification = None

            # Detect location format errors
            if (
                "location" in user_input.lower()
                and len(llm_output) < HEURISTIC_SHORT_LLM_OUTPUT_THRESHOLD
            ):
                pattern_detected = "location_format_error"
                suggested_clarification = "Pode detalhar o endereço? (rua, número, bairro)"

            # Detect if user seems confused
            elif any(word in user_input.lower() for word in ["?", "hã", "o que", "como"]):
                if len(llm_output) < HEURISTIC_CONFUSION_DETECTION_THRESHOLD:
                    pattern_detected = "possible_confusion"
                    suggested_clarification = "Deixe-me explicar melhor..."

            return ErrorCorrection(
                pattern_detected=pattern_detected,
                suggested_clarification=suggested_clarification,
                correction_needed=pattern_detected is not None,
                correction_urgency="medium" if pattern_detected else "low",
            )

        except Exception as e:
            logger.error(f"⚠️ Error pattern detection failed: {e}")
            from src.core.shared.models.response_models import ErrorCorrection

            return ErrorCorrection(pattern_detected=None, correction_needed=False)

    async def cleanup(self) -> None:
        """
        Cleanup resources.

        Note: HTTP session is managed by HTTPClient singleton,
        so we don't close it here to allow reuse.
        """
        logger.info("🧹 Cleaning up ConversationOrchestrator...")
        # Don't close http_session - it's managed by HTTPClient singleton
        # if self.http_session:
        #     await self.http_session.close()

        # Cleanup structured LLM client
        if hasattr(self, "structured_llm"):
            await self.structured_llm.cleanup()

        # Cleanup Talker
        if self.talker:
            await self.talker.cleanup()

        logger.info("✅ Cleanup complete")
