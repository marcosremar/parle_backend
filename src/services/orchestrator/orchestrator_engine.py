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

import aiohttp
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path
import time
import base64

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .service_clients import (
    create_service_clients,
    LLMClient, TTSClient, STTClient, ExternalLLMClient,
    SessionClient, ScenariosClient, ConversationStoreClient,
    ServiceClientError
)
from .fallback_manager import FallbackManager
from .talkers import TalkerFactory, AbstractTalker
from .process_turn_with_talker import process_turn_with_talker

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
    _valid_skills_cache: Optional[List[str]] = None
    _valid_skills_cache_timestamp: Optional[float] = None
    _valid_skills_cache_ttl: float = 300.0  # 5 minutos

    def __init__(self, config: Optional[Dict[str, str]] = None, in_process_mode: bool = False):
        """
        Initialize orchestrator with service clients

        Args:
            config: Optional service URL overrides
            in_process_mode: If True, load LLM/TTS modules in-process (0 HTTP overhead)
                           If False, use HTTP clients (default, with fallback)
        """
        # Get service URLs from environment or use defaults
        self.config = config or {}
        self._load_config_from_env()

        # In-process mode flag (for ultra-low latency)
        self.in_process_mode = in_process_mode

        # In-process module instances (only loaded if in_process_mode=True)
        self.llm_instance = None
        self.tts_instance = None
        self.gpu_available = False

        # Service clients (will be initialized later)
        self.clients: Dict[str, Any] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Fallback manager (will be initialized later)
        self.fallback_manager: Optional[FallbackManager] = None

        # Talker abstraction (will be initialized later)
        self.talker: Optional[AbstractTalker] = None

        # Stats
        self.stats = {
            "total_turns": 0,
            "successful_turns": 0,
            "failed_turns": 0,
            "primary_llm_count": 0,
            "fallback_llm_count": 0,
            "in_process_count": 0,  # New stat for in-process calls
            "http_fallback_count": 0,  # New stat for HTTP fallback
            "total_processing_time": 0.0
        }

        mode_str = "IN-PROCESS (ultra-low latency)" if in_process_mode else "HTTP (with fallback)"
        logger.info(f"🏗️ ConversationOrchestrator created - Mode: {mode_str}")
        
        # Import get_skill_difficulty once (for use in loops)
        try:
            from src.services.student_model.skill_registry import get_skill_difficulty, get_relevant_skills_for_context
            self._get_skill_difficulty = get_skill_difficulty
            self._get_relevant_skills_func = get_relevant_skills_for_context  # Renamed to avoid conflict
        except Exception as e:
            logger.warning(f"Could not import skill registry functions: {e}")
            self._get_skill_difficulty = None
            self._get_relevant_skills_func = None

    def _load_config_from_env(self):
        """Load service URLs from environment variables (with Nomad service discovery support)"""
        # Check if running in monolith mode
        self.monolith_mode = os.getenv("MONOLITH_MODE", "false").lower() == "true"
        
        env_mappings = {
            "llm_url": ("LLM_SERVICE_URL", "http://localhost:8110"),
            "tts_url": ("TTS_SERVICE_URL", "http://localhost:8103"),
            "stt_url": ("STT_SERVICE_URL", "http://localhost:8099"),
            "external_ultravox_url": ("ORCHESTRATOR_EXTERNAL_ULTRAVOX_URL", "http://localhost:8112"),
            "session_url": ("SESSION_SERVICE_URL", "http://localhost:8600"),
            "scenarios_url": ("SCENARIOS_SERVICE_URL", "http://localhost:8700"),
            "conversation_store_url": ("CONVERSATION_STORE_URL", "http://localhost:8800"),
            "conversation_history_url": ("CONVERSATION_HISTORY_URL", "http://localhost:8501")
        }

        for key, (env_var, default) in env_mappings.items():
            if key not in self.config:
                self.config[key] = os.getenv(env_var, default)
        
        if self.monolith_mode:
            logger.info("🏗️  Orchestrator running in MONOLITH mode (direct module calls)")
    
    def _get_relevant_skills_for_context(
        self,
        user_text: str,
        cefr_level: str,
        context_type: str = "production",
        force_refresh: bool = False
    ) -> List[str]:
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
        
        if (not force_refresh and 
            self._valid_skills_cache and 
            self._valid_skills_cache_timestamp and
            current_time - self._valid_skills_cache_timestamp < self._valid_skills_cache_ttl):
            return self._valid_skills_cache
        
        # Recalcular cache
        try:
            from src.services.student_model.skill_registry import SKILL_CEFR_MAP
            valid_skills = []
            for level_skills in SKILL_CEFR_MAP.values():
                valid_skills.extend(level_skills)
            
            self._valid_skills_cache = valid_skills
            self._valid_skills_cache_timestamp = current_time
            return valid_skills
        except Exception as e:
            logger.warning(f"Could not fetch valid skills: {e}")
            return []

    async def initialize(self):
        """
        Initialize HTTP session and all service clients
        Must be called before processing any requests
        """
        logger.info("🚀 Initializing ConversationOrchestrator...")

        # ==========================================
        # IN-PROCESS MODE: Load modules directly
        # ==========================================
        if self.in_process_mode:
            logger.info("⚡ IN-PROCESS MODE: Loading modules directly (0 HTTP overhead)...")
            try:
                # Check GPU availability
                import torch
                self.gpu_available = torch.cuda.is_available()

                if self.gpu_available:
                    logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

                    # Load Ultravox Universal (auto-detects GPU profile)
                    from src.services.llm.ultravox.ultravox_universal import UltravoxUniversal
                    self.llm_instance = UltravoxUniversal()
                    await self.llm_instance.initialize()
                    logger.info("✅ Ultravox Universal loaded in-process")

                    # TTS will use HTTP service
                    self.tts_instance = None
                    logger.info("✅ In-process TTS disabled - using HTTP TTS service")

                    logger.info("🚀 In-process modules ready - ultra-low latency enabled!")
                else:
                    logger.warning("⚠️ No GPU available - in-process mode disabled, will use HTTP fallback")
                    self.in_process_mode = False  # Disable in-process mode

            except Exception as e:
                logger.error(f"❌ Failed to load in-process modules: {e}")
                logger.info("   Falling back to HTTP mode...")
                self.in_process_mode = False  # Disable in-process mode on error

        # ==========================================
        # HTTP MODE: Initialize service clients
        # ==========================================
        # Always create HTTP clients (needed for fallback even in in-process mode)
        logger.info("🌐 Initializing HTTP service clients...")

        # Create shared HTTP session
        self.http_session = aiohttp.ClientSession()

        # Create all service clients (using HTTP directly, no communication manager)
        self.clients = create_service_clients(self.config, self.http_session)

        # Initialize each client with shared session
        for name, client in self.clients.items():
            await client.initialize(self.http_session)

        # Create fallback manager with 2-tier failover
        self.fallback_manager = FallbackManager(
            primary_llm=self.clients["llm"],
            secondary_llm=self.clients["external_ultravox"]
        )

        # Health check all services (skip if ORCHESTRATOR_SKIP_HEALTH_CHECKS is set)
        skip_health_checks = os.getenv("ORCHESTRATOR_SKIP_HEALTH_CHECKS", "false").lower() == "true"
        if not self.in_process_mode and not skip_health_checks:
            await self._health_check_services()
        elif skip_health_checks:
            logger.info("🏁 Skipping health checks (startup mode - will run in background)")

        # Run profile-aware warmup
        await self._run_profile_warmup()

        # ==========================================
        # TALKER ABSTRACTION: Create appropriate Talker
        # ==========================================
        logger.info("🎯 Creating Talker (conversation pipeline abstraction)...")
        try:
            # Check GPU availability for Talker decision (gracefully handle torch missing)
            gpu_available = False
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                logger.debug("   torch not available - GPU not available")
                gpu_available = False

            # Create Talker (InternalTalker if GPU, ExternalTalker otherwise)
            self.talker = await TalkerFactory.create_talker(
                gpu_available=gpu_available,
                service_clients=self.clients,
                force_external=not self.in_process_mode  # Force external if not in-process mode
            )

            logger.info(f"✅ Talker ready: {self.talker.name}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to create Talker: {e}")
            logger.info("   Orchestrator will use legacy process_turn() method")
            self.talker = None

        mode_summary = "IN-PROCESS (with HTTP fallback)" if self.in_process_mode else "HTTP (with fallback)"
        logger.info(f"✅ ConversationOrchestrator initialized and ready - Mode: {mode_summary}")

    async def _health_check_services(self):
        """Check health of all downstream services"""
        logger.info("🔍 Checking downstream services health...")

        health_status = {}
        for name, client in self.clients.items():
            try:
                is_healthy = await client.health_check()
                health_status[name] = is_healthy
                status_icon = "✅" if is_healthy else "❌"
                logger.info(f"   {status_icon} {name}: {'healthy' if is_healthy else 'unhealthy'}")
            except Exception as e:
                health_status[name] = False
                logger.warning(f"   ❌ {name}: {e}")

        # Log summary
        healthy_count = sum(1 for v in health_status.values() if v)
        total_count = len(health_status)
        logger.info(f"📊 Services health: {healthy_count}/{total_count} healthy")

        return health_status

    async def _run_profile_warmup(self):
        """Run profile-aware warmup for all services"""
        try:
            logger.info("🔥 Starting warmup...")
            # Warmup functionality removed - services are independent
            # Each service handles its own warmup if needed
            logger.info("✅ Warmup skipped (services are independent)")

            # Log warmup config
            warmup_config = profile_manager.get_warmup_config(active_profile)
            if warmup_config:
                logger.info(
                    f"   STT iterations: {warmup_config.stt.iterations if warmup_config.stt else 0}"
                )
                logger.info(
                    f"   LLM iterations: {warmup_config.llm.iterations if warmup_config.llm else 0}"
                )
                logger.info(
                    f"   TTS iterations: {warmup_config.tts.iterations if warmup_config.tts else 0}"
                )
                logger.info(
                    f"   Mode: {'Parallel' if warmup_config.parallel else 'Sequential'}"
                )

        except Exception as e:
            logger.warning(f"⚠️ Profile warmup failed (non-critical): {e}")
            # Don't fail initialization if warmup fails

    async def process_turn(self,
                          audio_data: bytes,
                          session_id: str,
                          sample_rate: int = 16000,
                          voice_id: Optional[str] = None,
                          force_external_llm: bool = False) -> Dict[str, Any]:
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
        start_time = time.time()
        self.stats["total_turns"] += 1

        try:
            logger.info(f"🎤 Processing conversation turn: session={session_id}, audio={len(audio_data)} bytes")

            # ==========================================
            # STEP 1: Get Session and Scenario Context (OPTIMIZED - PARALLEL LOADING)
            # ==========================================
            # Step 1a: Load session first (required for scenario_id and conversation_id)
            session_data = await self.clients["session"].get_session(session_id)
            scenario_data = None
            # Default system prompt for Q&A when no scenario is provided
            system_prompt = """You are a helpful AI assistant. Your task is to:

1. LISTEN CAREFULLY to the audio and identify the specific question being asked
2. ANSWER ONLY that specific question directly and accurately
3. Provide a concise, factual answer (1-2 sentences maximum)
4. DO NOT ask questions back to the user
5. DO NOT change the topic or discuss unrelated things
6. Focus on accuracy and relevance

Example:
Audio: "Qual é a capital da França?"
Correct response: "A capital da França é Paris."
Incorrect: "Qual é o teu nome?" (asking a different question)
Incorrect: "Paris é uma bela cidade. Você gostaria de saber mais?" (asking follow-up questions)

Listen to the audio, identify the question, and answer it directly."""
            conversation_id = None
            conversation_history = []
            scenario_id = None
            validation_result = None
            user_id = session_id  # Use session_id as user_id for now (can be extracted from session_data if available)
            target_skill = None
            student_profile = None

            if session_data:
                conversation_id = session_data.get("conversation_id")
                scenario_id = session_data.get("scenario_id")
                voice_id = voice_id or session_data.get("voice_id", None)
                user_id = session_data.get("user_id", session_id)  # Try to get user_id from session

                # Step 1b: Load scenario + history + student data in PARALLEL
                tasks = []

                # Add scenario task if scenario_id exists
                if scenario_id:
                    tasks.append(self.clients["scenarios"].get_scenario(scenario_id))
                else:
                    tasks.append(None)  # Placeholder

                # Add history task if conversation_id exists
                if conversation_id:
                    tasks.append(self.clients["conversation_store"].get_context(
                        conversation_id,
                        limit=10
                    ))
                else:
                    tasks.append(None)  # Placeholder

                # Add student profile task (if student_model client exists)
                if "student_model" in self.clients:
                    # NEW: Get detailed CEFR progress instead of just profile
                    tasks.append(self.clients["student_model"].get_cefr_progress(user_id))
                else:
                    tasks.append(None)

                # Add next skill task (if learning_path client exists)
                if "learning_path" in self.clients:
                    tasks.append(self.clients["learning_path"].get_next_skill(user_id))
                else:
                    tasks.append(None)

                # Execute in parallel (asyncio.gather)
                if any(task is not None for task in tasks):
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process scenario result
                    if scenario_id and not isinstance(results[0], Exception) and results[0]:
                        scenario_data = results[0]
                        base_system_prompt = scenario_data.get("system_prompt")
                        logger.info(f"📝 Using scenario: {scenario_data.get('name', scenario_id)}")
                        system_prompt = base_system_prompt
                    elif scenario_id and isinstance(results[0], Exception):
                        logger.warning(f"⚠️ Failed to load scenario: {results[0]}")

                    # Process history result
                    if conversation_id and not isinstance(results[1], Exception) and results[1]:
                        messages = results[1]
                        conversation_history = self._format_conversation_history(messages)
                        logger.info(f"📚 Loaded {len(messages)} previous messages for context")
                    elif conversation_id and isinstance(results[1], Exception):
                        logger.warning(f"⚠️ Failed to load conversation history: {results[1]}")

                    # Process student profile/CEFR result
                    student_cefr_progress = None
                    current_cefr_level = "A1"
                    if "student_model" in self.clients and not isinstance(results[2], Exception) and results[2]:
                        student_cefr_progress = results[2]
                        # Alguns endpoints usam "current_estimated_level", outros "current_level"
                        current_cefr_level = (
                            student_cefr_progress.get("current_estimated_level")
                            or student_cefr_progress.get("current_level")
                            or "A1"
                        )
                        logger.info(f"👤 Loaded student CEFR progress: {current_cefr_level}")

                    # Process next skill result
                    if "learning_path" in self.clients and not isinstance(results[3], Exception) and results[3]:
                        target_skill = results[3]
                        logger.info(f"🎯 Target skill: {target_skill.get('skill_name', target_skill.get('skill_id', 'unknown'))}")

            else:
                logger.warning(f"⚠️ Session {session_id} not found, using defaults")
                voice_id = voice_id or None

            # ==========================================
            # STEP 2: Get User Transcript (STT) - Required for analysis
            # ==========================================
            # We need transcript BEFORE analysis, so we do STT first if not already done
            user_transcript = ""
            
            # Try in-process STT first if enabled
            if self.in_process_mode and self.llm_instance and not force_external_llm:
                try:
                    # Convert audio_data to numpy array
                    import numpy as np
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Get transcript from Ultravox (it does STT internally)
                    # We'll call it just for transcript, then use it for analysis
                    result = await self.llm_instance.process_audio(
                        audio_array=audio_array,
                        sample_rate=sample_rate,
                        system_prompt="Transcribe the audio.",  # Minimal prompt for STT
                        conversation_history=[]
                    )
                    user_transcript = result.get("transcript", "")
                    logger.info(f"📝 Got transcript from in-process: {user_transcript[:50]}...")
                except Exception as e:
                    logger.debug(f"Could not get transcript from in-process: {e}")
            
            # Fallback to HTTP STT if needed
            if not user_transcript and "stt" in self.clients:
                try:
                    stt_result = await self.clients["stt"].transcribe(audio_data, sample_rate)
                    user_transcript = stt_result.get("text", "")
                    logger.info(f"📝 Got transcript from STT service: {user_transcript[:50]}...")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get transcript: {e}")

            # ==========================================
            # STEP 2.5: Analyze Current Turn (BEFORE generating response)
            # ==========================================
            # Analyze user speech to adapt pedagogy BEFORE generating response
            # Call analyze_turn and extract_skills in parallel for better performance
            turn_analysis = None
            skills_extraction = None
            if user_transcript and "speech_grader" in self.clients:
                try:
                    # Get valid skills with context filtering and caching
                    valid_skills = self._get_relevant_skills_for_context(
                        user_text=user_transcript,
                        # Se não tivermos progresso CEFR ainda, assume A1–A2 (diagnóstico inicial simples)
                        cefr_level=current_cefr_level if student_cefr_progress else "A1",
                        context_type="production"
                    )
                    
                    # Call both endpoints in parallel using asyncio.gather
                    if valid_skills:
                        try:
                            turn_analysis_task = self.clients["speech_grader"].analyze_turn(
                                user_text=user_transcript,
                                ai_text=None,  # No AI text yet (we're analyzing BEFORE response)
                                valid_skills=valid_skills
                            )
                            skills_extraction_task = self.clients["speech_grader"].extract_skills(
                                user_text=user_transcript,
                                valid_skills=valid_skills,
                                ai_text=None
                            )
                            
                            # Execute both in parallel
                            turn_analysis, skills_extraction = await asyncio.gather(
                                turn_analysis_task,
                                skills_extraction_task,
                                return_exceptions=True
                            )
                            
                            # Handle exceptions with traceback
                            if isinstance(turn_analysis, Exception):
                                logger.error(f"⚠️ analyze_turn failed: {turn_analysis}")
                                import traceback
                                logger.debug(f"Traceback: {traceback.format_exc()}")
                                turn_analysis = None
                            if isinstance(skills_extraction, Exception):
                                logger.error(f"⚠️ extract_skills failed: {skills_extraction}")
                                import traceback
                                logger.debug(f"Traceback: {traceback.format_exc()}")
                                skills_extraction = None
                            
                            logger.info(f"🔍 Analyzed turn: {len(turn_analysis.get('errors', [])) if turn_analysis else 0} errors, "
                                      f"{len(turn_analysis.get('correct_skills', [])) if turn_analysis else 0} correct skills, "
                                      f"{len(skills_extraction.get('skills', [])) if skills_extraction else 0} extracted skills")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to run parallel analysis: {e}")
                            # Fallback: try analyze_turn alone
                            try:
                                turn_analysis = await self.clients["speech_grader"].analyze_turn(
                                    user_text=user_transcript,
                                    ai_text=None,
                                    valid_skills=valid_skills
                                )
                            except Exception as e2:
                                logger.warning(f"⚠️ Fallback analyze_turn also failed: {e2}")
                                turn_analysis = None
                    else:
                        # No valid_skills, just call analyze_turn
                        turn_analysis = await self.clients["speech_grader"].analyze_turn(
                            user_text=user_transcript,
                            ai_text=None,
                            valid_skills=None
                        )
                        logger.info(f"🔍 Analyzed turn (no valid_skills): {len(turn_analysis.get('errors', []))} errors")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to analyze turn: {e}")
                    turn_analysis = None
                    skills_extraction = None

            # ==========================================
            # STEP 1.6: Combine Results from Parallel Analysis
            # ==========================================
            # Merge results from analyze_turn and extract_skills
            if turn_analysis or skills_extraction:
                # Combine linguistic features: prioritize extract_skills if more complete
                combined_linguistic_features = {}
                if skills_extraction and skills_extraction.get("overall_linguistic_features"):
                    combined_linguistic_features = skills_extraction.get("overall_linguistic_features", {})
                elif turn_analysis:
                    combined_linguistic_features = turn_analysis.get("linguistic_features", {})
                
                # Merge correct_skills: combine from both sources, keeping confidence scores
                combined_skills_with_confidence = {}  # skill_id -> max_confidence
                
                # From turn_analysis (implicit high confidence)
                if turn_analysis:
                    for skill_id in turn_analysis.get("correct_skills", []):
                        combined_skills_with_confidence[skill_id] = 1.0
                
                # From extract_skills (with actual confidence scores)
                if skills_extraction:
                    for skill_data in skills_extraction.get("skills", []):
                        skill_id = skill_data.get("skill_id")
                        confidence = skill_data.get("confidence", 0.0)
                        if skill_id and confidence >= 0.7:
                            # Use max confidence if skill appears in both
                            combined_skills_with_confidence[skill_id] = max(
                                combined_skills_with_confidence.get(skill_id, 0.0),
                                confidence
                            )
                
                combined_correct_skills = list(combined_skills_with_confidence.keys())
                
                # Update turn_analysis with combined data
                if turn_analysis:
                    turn_analysis["linguistic_features"] = combined_linguistic_features
                    turn_analysis["correct_skills"] = list(combined_correct_skills)
                else:
                    # If turn_analysis failed but skills_extraction succeeded, create minimal turn_analysis
                    turn_analysis = {
                        "errors": [],
                        "correct_skills": list(combined_correct_skills),
                        "linguistic_features": combined_linguistic_features,
                        "semantic_skill_mapping": {},
                        "summary": skills_extraction.get("summary", "Skills extracted") if skills_extraction else "Analysis incomplete"
                    }
                
                logger.info(f"📊 Combined results: {len(combined_correct_skills)} total skills, "
                          f"{len(combined_linguistic_features)} linguistic features")
            
            # ==========================================
            # STEP 1.7: Update Knowledge (BEFORE composing prompt)
            # ==========================================
            # Update student knowledge based on analysis so prompt uses updated state
            if turn_analysis and target_skill and "student_model" in self.clients:
                try:
                    # Extract analysis data (now with combined results)
                    linguistic_features = turn_analysis.get("linguistic_features", {})
                    semantic_skill_mapping = turn_analysis.get("semantic_skill_mapping", {})
                    correct_skills = turn_analysis.get("correct_skills", [])
                    errors = turn_analysis.get("errors", [])
                    
                    # Also add high-confidence skills from extract_skills
                    if skills_extraction:
                        for skill_data in skills_extraction.get("skills", []):
                            skill_id = skill_data.get("skill_id")
                            confidence = skill_data.get("confidence", 0.0)
                            if skill_id and confidence >= 0.7:
                                # Add to semantic_skill_mapping if not already present
                                if skill_id not in semantic_skill_mapping:
                                    semantic_skill_mapping[skill_id] = confidence
                    
                    # Process successes with confidence tracking
                    # Merge skills from both sources, keeping max confidence
                    success_skills_with_confidence = {}  # skill_id -> max_confidence
                    
                    # From turn_analysis correct_skills (implicit high confidence)
                    for skill_id in correct_skills:
                        success_skills_with_confidence[skill_id] = 1.0
                    
                    # From semantic_skill_mapping
                    for skill_id, confidence in semantic_skill_mapping.items():
                        if confidence >= 0.7:
                            success_skills_with_confidence[skill_id] = max(
                                success_skills_with_confidence.get(skill_id, 0.0),
                                confidence
                            )
                    
                    # From extract_skills (high confidence only)
                    if skills_extraction:
                        for skill_data in skills_extraction.get("skills", []):
                            skill_id = skill_data.get("skill_id")
                            confidence = skill_data.get("confidence", 0.0)
                            if skill_id and confidence >= 0.7:
                                success_skills_with_confidence[skill_id] = max(
                                    success_skills_with_confidence.get(skill_id, 0.0),
                                    confidence
                                )
                    
                    # Process all success skills
                    for usage_skill_id in success_skills_with_confidence.keys():
                        skill_difficulty = None
                        if self._get_skill_difficulty:
                            try:
                                skill_difficulty = self._get_skill_difficulty(usage_skill_id)
                            except Exception:
                                pass
                        
                        await self.clients["student_model"].assess(
                            user_id=user_id,
                            skill_id=usage_skill_id,
                            correct=True,
                            context={"type": "explicit_success", "source": "pre_response_analysis"},
                            user_text=user_transcript,
                            ai_text=None,
                            difficulty=skill_difficulty,
                            linguistic_features=linguistic_features
                        )
                    
                    # Process errors
                    for error in errors:
                        error_skill_id = error.get("skill_id")
                        if error_skill_id:
                            skill_difficulty = None
                            if self._get_skill_difficulty:
                                try:
                                    skill_difficulty = self._get_skill_difficulty(error_skill_id)
                                except Exception:
                                    pass
                            
                            error_linguistic_features = error.get("linguistic_features", linguistic_features)
                            
                            await self.clients["student_model"].assess(
                                user_id=user_id,
                                skill_id=error_skill_id,
                                correct=False,
                                context={
                                    "type": "error",
                                    "error_type": error.get("error_type"),
                                    "explanation": error.get("explanation"),
                                    "severity": error.get("severity")
                                },
                                user_text=user_transcript,
                                ai_text=None,
                                difficulty=skill_difficulty,
                                linguistic_features=error_linguistic_features
                            )
                    
                    logger.info(f"📊 Updated knowledge: {len(success_skills)} successes, {len(errors)} errors")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update knowledge: {e}")

            # ==========================================
            # STEP 1.7: Compose Pedagogical Prompt (with turn analysis)
            # ==========================================
            if "pedagogical_policy" in self.clients and (scenario_data or student_cefr_progress or target_skill):
                try:
                    # Fetch updated interpretable knowledge state (after analysis)
                    interpretable_state = None
                    if "student_model" in self.clients:
                        try:
                            interpretable_state = await self.clients["student_model"].get_interpretable_knowledge_state(user_id)
                        except Exception as e:
                            logger.debug(f"Could not fetch interpretable knowledge state: {e}")
                    
                    # Build session-level analysis if we have conversation history
                    session_analysis = None
                    if conversation_history and len(conversation_history) >= 2 and "speech_grader" in self.clients:
                        try:
                            # Get recent turn analyses from conversation history
                            # We need to reconstruct session_turns from conversation_history
                            # For now, we'll use the current turn analysis + historical patterns
                            # In a full implementation, we'd store turn analyses in conversation_store
                            
                            # Use interpretable_knowledge_state which already aggregates historical data
                            # This includes linguistic_error_patterns which aggregates last 90 days
                            if interpretable_state and interpretable_state.get("linguistic_error_patterns"):
                                session_analysis = {
                                    "historical_patterns": interpretable_state.get("linguistic_error_patterns"),
                                    "session_context": f"Conversa com {len(conversation_history)} mensagens anteriores"
                                }
                        except Exception as e:
                            logger.debug(f"Could not build session analysis: {e}")
                    
                    # Build prompt context with turn analysis AND session analysis
                    prompt_context_dict = {
                        "scenario": scenario_data,
                        "cefr_level": student_cefr_progress.get("current_estimated_level", "A1") if student_cefr_progress else "A1",
                        "cefr_details": student_cefr_progress.get("cefr_details", {}) if student_cefr_progress else {},
                        "native_language": "pt", # Default or fetch if available in profile
                        "target_skill": target_skill,
                        "mastery_probability": target_skill.get("mastery_probability", 0.0) if target_skill else 0.0,
                        "emotional_state": "neutral",  # TODO: Get from session or analysis
                        "conversation_history": conversation_history,
                        "interpretable_knowledge_state": interpretable_state,
                        "current_turn_analysis": turn_analysis,  # Analysis of current turn
                        "session_analysis": session_analysis  # NEW: Aggregated session-level analysis
                    }
                    
                    # Compose pedagogical prompt
                    prompt_result = await self.clients["pedagogical_policy"].compose_prompt(prompt_context_dict)
                    
                    if prompt_result and prompt_result.get("prompt"):
                        system_prompt = prompt_result["prompt"]
                        logger.info(f"📚 Composed pedagogical prompt (strategy: {prompt_result.get('strategy', 'unknown')})")
                    else:
                        logger.warning("⚠️ Failed to compose pedagogical prompt, using default")
                except Exception as e:
                    logger.warning(f"⚠️ Error composing pedagogical prompt: {e}, using default")
                    # Continue with default prompt

            # ==========================================
            # STEP 3: Call LLM (In-Process or HTTP with Failover)
            # ==========================================
            # Note: If we already got transcript from STT above, we can use text-based LLM call
            # But for simplicity, we'll still use audio-based call (Ultravox handles both)
            llm_result = None
            text_response = None
            llm_used = "unknown"

            # Try in-process first if enabled
            if self.in_process_mode and self.llm_instance and not force_external_llm:
                try:
                    logger.info("⚡ Using in-process LLM (ultra-low latency)...")

                    # Convert audio_data to numpy array
                    import numpy as np
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Log detailed audio pipeline flow
                    logger.debug(f"🔄 Audio Pipeline Flow:")
                    logger.debug(f"   1. Input: {len(audio_data)} bytes (int16 PCM)")
                    logger.debug(f"   2. Decoded: {len(audio_array)} samples")
                    logger.debug(f"   3. Normalized: int16 [-32768, 32767] → float32 [-1.0, 1.0]")
                    logger.debug(f"   4. Array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
                    logger.debug(f"   5. Sample rate: {sample_rate} Hz")
                    logger.debug(f"   6. Duration: {len(audio_array) / sample_rate:.2f}s")

                    # Call Ultravox Universal in-process
                    # system_prompt now includes pedagogical adaptation based on analysis
                    result = await self.llm_instance.process_audio(
                        audio_array=audio_array,
                        sample_rate=sample_rate,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history
                    )

                    text_response = result.get("text", "")
                    # Only update transcript if we didn't get it earlier from STT
                    if not user_transcript:
                        user_transcript = result.get("transcript", "")
                    llm_used = "in_process"

                    logger.info(f"✅ In-process LLM response: {text_response[:100]}...")
                    self.stats["in_process_count"] += 1

                    llm_result = {"success": True, "text": text_response, "llm_used": llm_used}

                except Exception as e:
                    # Wrap exception using hierarchy
                    try:
                        from .utils.exceptions import wrap_exception, LLMError
                        wrapped_error = wrap_exception(e, service_name="llm", operation="process_audio")
                        logger.warning(f"⚠️ In-process LLM failed: {wrapped_error}")
                    except ImportError:
                        logger.warning(f"⚠️ In-process LLM failed: {e}")
                    logger.info("   Falling back to HTTP mode...")
                    self.stats["http_fallback_count"] += 1
                    llm_result = None  # Trigger HTTP fallback

            # Fallback to HTTP if in-process failed or not enabled
            if llm_result is None:
                llm_result = await self.fallback_manager.call_llm_with_failover(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    system_prompt=system_prompt,
                    conversation_id=conversation_id,
                    conversation_history=conversation_history,
                    force_external_llm=force_external_llm
                )

                if not llm_result["success"]:
                    logger.error(f"❌ LLM processing failed: {llm_result.get('error')}")
                    self.stats["failed_turns"] += 1
                    return {
                        "success": False,
                        "error": llm_result.get("error", "LLM processing failed"),
                        "session_id": session_id
                    }

                text_response = llm_result["text"]
                # Only update transcript if we didn't get it earlier from STT
                if not user_transcript:
                    user_transcript = llm_result.get("transcript", "")
                llm_used = llm_result["llm_used"]

                logger.info(f"🤖 LLM ({llm_used}) response: {text_response[:100]}...")

                # Update stats
                if llm_used == "primary":
                    self.stats["primary_llm_count"] += 1
                elif llm_used == "fallback":
                    self.stats["fallback_llm_count"] += 1

            # ==========================================
            # STEP 3: Generate TTS Audio Response (In-Process or HTTP)
            # ==========================================
            audio_response = None

            # Try in-process TTS first if enabled
            if self.in_process_mode and self.tts_instance:
                try:
                    logger.info("⚡ Using in-process TTS (ultra-low latency)...")

                    audio_response = await self.tts_instance.synthesize(
                        text=text_response,
                        voice_id=voice_id
                    )

                    logger.info(f"✅ In-process TTS generated: {len(audio_response)} bytes")

                except Exception as e:
                    # Wrap exception using hierarchy
                    try:
                        from .utils.exceptions import wrap_exception, TTSError
                        wrapped_error = wrap_exception(e, service_name="tts", operation="synthesize")
                        logger.warning(f"⚠️ In-process TTS failed: {wrapped_error}")
                    except ImportError:
                        logger.warning(f"⚠️ In-process TTS failed: {e}")
                    logger.info("   Falling back to HTTP TTS...")
                    audio_response = None  # Trigger HTTP fallback

            # Fallback to HTTP TTS if in-process failed or not enabled
            if audio_response is None:
                try:
                    # Try local TTS first
                    audio_response = await self.clients["tts"].synthesize(
                        text=text_response,
                        voice_id=voice_id
                    )
                    logger.info(f"🔊 TTS (local) generated: {len(audio_response)} bytes")
                except ServiceClientError as e:
                    # Wrap exception using hierarchy
                    try:
                        from .utils.exceptions import wrap_exception, TTSError
                        wrapped_error = wrap_exception(e, service_name="tts", operation="synthesize")
                        logger.warning(f"⚠️ Local TTS failed: {wrapped_error}")
                    except ImportError:
                        logger.warning(f"⚠️ Local TTS failed: {e}")
                    logger.info("   Trying external TTS (HuggingFace)...")

                    # Fallback to external TTS
                    try:
                        audio_response = await self.clients["tts"].synthesize(
                            text=text_response,
                            voice="af_heart",  # Default HF voice
                            format="wav"
                        )
                        logger.info(f"🔊 TTS (external) generated: {len(audio_response)} bytes")
                    except ServiceClientError as e2:
                        # Wrap exception using hierarchy
                        try:
                            from .utils.exceptions import wrap_exception, TTSError
                            wrapped_error = wrap_exception(e2, service_name="tts", operation="synthesize_external")
                            logger.error(f"❌ External TTS also failed: {wrapped_error}")
                        except ImportError:
                            logger.error(f"❌ External TTS also failed: {e2}")
                        # Return text-only response if all TTS fails
                        audio_response = None

            # ==========================================
            # STEP 5 + 6: Save Conversation Turn + Update Session (OPTIMIZED - PARALLEL)
            # ==========================================
            # Both operations are independent and can run in parallel (30ms → 15ms)
            save_tasks = []

            # Add save turn task
            if conversation_id:
                save_tasks.append(
                    self.clients["conversation_store"].add_turn(
                        conversation_id=conversation_id,
                        user_audio=audio_data,
                        user_text=user_transcript,
                        ai_text=text_response,
                        ai_audio=audio_response
                    )
                )

            # Add update session task
            if session_data:
                save_tasks.append(
                    self.clients["session"].update_session_llm(session_id, llm_used)
                )

            # Execute save operations in parallel
            if save_tasks:
                save_results = await asyncio.gather(*save_tasks, return_exceptions=True)

                # Log results
                task_idx = 0
                if conversation_id:
                    if isinstance(save_results[task_idx], Exception):
                        logger.warning(f"⚠️ Failed to save turn: {save_results[task_idx]}")
                    else:
                        logger.info(f"💾 Turn saved to conversation {conversation_id}")
                    task_idx += 1

                if session_data:
                    if isinstance(save_results[task_idx], Exception):
                        logger.warning(f"⚠️ Failed to update session: {save_results[task_idx]}")
                    else:
                        logger.debug(f"✅ Session {session_id} updated with LLM: {llm_used}")

            # ==========================================
            # STEP 6.5: Background Final Knowledge Update (Optional - for reinforcement)
            # ==========================================
            # Note: Main analysis and update already done BEFORE response generation
            # This is just for final reinforcement/validation if needed
            # (Analysis is now done in STEP 2.5, before prompt composition)

            # ==========================================
            # STEP 6: Return Response
            # ==========================================
            processing_time = time.time() - start_time
            self.stats["successful_turns"] += 1
            self.stats["total_processing_time"] += processing_time

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
                    "has_tts": audio_response is not None
                }
            }

            logger.info(f"✅ Turn completed in {processing_time:.2f}s using {llm_used} LLM")
            return response

        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}", exc_info=True)
            self.stats["failed_turns"] += 1
            return {
                "success": False,
                "error": f"Orchestration failed: {str(e)}",
                "session_id": session_id
            }

    def _format_conversation_history(self, messages: list) -> list:
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
        self,
        user_id: str,
        target_skill: Dict[str, Any],
        user_text: str,
        ai_text: str
    ):
        """
        Background task: Analyze turn and update student knowledge
        
        This runs asynchronously after the main response is sent,
        so it doesn't block the conversation flow.
        
        Args:
            user_id: User ID
            target_skill: Target skill being practiced
            user_text: User's input text
            ai_text: AI's response text
        """
        try:
            logger.info(f"🔍 Background: Analyzing turn for user {user_id}, skill {target_skill.get('skill_id')}")
            
            # Step 1: Get valid skills from SKILL_CEFR_MAP for SINKT semantic tagging
            valid_skills = []
            try:
                from src.services.student_model.skill_registry import SKILL_CEFR_MAP
                for level_skills in SKILL_CEFR_MAP.values():
                    valid_skills.extend(level_skills)
            except Exception as e:
                logger.warning(f"Could not fetch valid skills: {e}")
            
            # Step 2: Analyze errors AND successes using speech_grader (with SINKT and linguistic features)
            analysis = await self.clients["speech_grader"].analyze_turn(
                user_text=user_text,
                ai_text=ai_text,
                valid_skills=valid_skills if valid_skills else None
            )
            
            skill_id = target_skill.get("skill_id")
            
            # ==================================================================================
            # NEW LOGIC: Process granular skill updates (Successes & Errors)
            # ==================================================================================
            
            # Extract linguistic features and semantic mapping from analysis
            linguistic_features = analysis.get("linguistic_features", {})
            semantic_skill_mapping = analysis.get("semantic_skill_mapping", {})
            correct_skills = analysis.get("correct_skills", [])
            
            # 1. Process explicit successes identified by LLM (correct_skills + semantic mapping)
            # Combine correct_skills from analysis with high-confidence semantic mappings
            success_skills = set(correct_skills)
            for skill_id, confidence in semantic_skill_mapping.items():
                if confidence >= 0.7:  # High confidence semantic match
                    success_skills.add(skill_id)
            
            for usage_skill_id in success_skills:
                # Get difficulty for this skill
                try:
                    from src.services.student_model.skill_registry import get_skill_difficulty
                    skill_difficulty = get_skill_difficulty(usage_skill_id)
                except Exception:
                    skill_difficulty = None
                
                await self.clients["student_model"].assess(
                    user_id=user_id,
                    skill_id=usage_skill_id,
                    correct=True,
                    context={
                        "type": "explicit_success",
                        "source": "diagnostic_analysis"
                    },
                    user_text=user_text,
                    ai_text=ai_text,
                    difficulty=skill_difficulty,
                    linguistic_features=linguistic_features
                )
                logger.info(f"✅ Registered SUCCESS for skill {usage_skill_id}")

            # 2. Process errors identified by LLM
            errors = analysis.get("errors", [])
            error_skills_processed = set()
            
            for error in errors:
                error_skill_id = error.get("skill_id")
                if error_skill_id:
                    # Get difficulty for this skill
                    try:
                        from src.services.student_model.skill_registry import get_skill_difficulty
                        skill_difficulty = get_skill_difficulty(error_skill_id)
                    except Exception:
                        skill_difficulty = None
                    
                    # Extract linguistic features from error if available
                    error_linguistic_features = error.get("linguistic_features", linguistic_features)
                    
                    await self.clients["student_model"].assess(
                        user_id=user_id,
                        skill_id=error_skill_id,
                        correct=False,
                        context={
                            "type": "error",
                            "error_type": error.get("error_type"),
                            "explanation": error.get("explanation"),
                            "severity": error.get("severity")
                        },
                        user_text=user_text,
                        ai_text=ai_text,
                        difficulty=skill_difficulty,
                        linguistic_features=error_linguistic_features
                    )
                    error_skills_processed.add(error_skill_id)
                    logger.info(f"❌ Registered ERROR for skill {error_skill_id}")

            # 3. Handle Target Skill Logic (fallback/reinforcement)
            # If the target skill was NOT mentioned in errors OR successes explicitly,
            # we need to infer its status.
            if skill_id:
                # Was it marked as error?
                is_error = skill_id in error_skills_processed
                
                # Was it marked as success?
                is_success = skill_id in success_skills
                
                if not is_error and not is_success:
                    # Inferred success: If user spoke and didn't make a mistake on the target skill,
                    # AND the analysis didn't explicitly flag it, we can assume implicit success 
                    # IF the analysis says "target_skill_correct" (legacy flag) or if no errors found at all.
                    
                    # Fallback to legacy logic
                    target_skill_correct = analysis.get("target_skill_correct", False)
                    if not errors: 
                        target_skill_correct = True
                    
                    if target_skill_correct:
                        # Get difficulty for target skill
                        try:
                            from src.services.student_model.skill_registry import get_skill_difficulty
                            skill_difficulty = get_skill_difficulty(skill_id)
                        except Exception:
                            skill_difficulty = None
                        
                        await self.clients["student_model"].assess(
                            user_id=user_id,
                            skill_id=skill_id,
                            correct=True,
                            context={"type": "implicit_success"},
                            user_text=user_text,
                            ai_text=ai_text,
                            difficulty=skill_difficulty,
                            linguistic_features=linguistic_features
                        )
                        logger.info(f"✅ Registered IMPLICIT SUCCESS for target skill {skill_id}")

        except Exception as e:
            logger.error(f"❌ Error in background analysis: {e}", exc_info=True)
            # Don't raise - this is a background task, errors shouldn't affect main flow

    async def get_services_health(self) -> Dict[str, bool]:
        """Get health status of all services"""
        return await self._health_check_services()

    async def get_fallback_health(self) -> Dict[str, bool]:
        """Get health status of LLM failover components"""
        return await self.fallback_manager.health_check()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics including controller integration metrics

        Returns:
            Dict with processing stats, controller integration, and data flow info
        """
        avg_time = (self.stats["total_processing_time"] / self.stats["total_turns"]
                   if self.stats["total_turns"] > 0 else 0)

        return {
            **self.stats,
            "average_processing_time_ms": int(avg_time * 1000),
            "success_rate": (self.stats["successful_turns"] / self.stats["total_turns"]
                           if self.stats["total_turns"] > 0 else 0),
            "primary_llm_rate": (self.stats["primary_llm_count"] / self.stats["total_turns"]
                               if self.stats["total_turns"] > 0 else 0),

            # Controller Integration Metrics
            "controller_integration": {
                "entry_points": [
                    "API Gateway (POST /process) - port 8888",
                    "WebRTC (POST /process) - port 8020",
                    "WebSocket (Socket.IO audio event) - port 8022",
                    "REST Polling (POST /api/session/{id}/audio) - port 8600"
                ],
                "controller_type": "ConversationController",
                "validation_format": "Ultravox LLM Format",
                "requirements": {
                    "audio_format": "Base64 encoded int16 PCM",
                    "sample_rate": "16000 Hz (recommended)",
                    "minimum_duration": "40ms (640 samples @ 16kHz)",
                    "maximum_size": "50 MB"
                }
            },

            # Data Flow Documentation
            "data_flow": {
                "description": "Audio data transformation pipeline",
                "stages": [
                    "1. Entry Point: Base64 string received from client",
                    "2. Controller: Validates Base64 format, sample rate, size",
                    "3. Controller: Decodes Base64 → raw bytes (int16 PCM)",
                    "4. Orchestrator: Converts bytes → numpy float32 array",
                    "5. Orchestrator: Normalizes int16 [-32768, 32767] → float32 [-1.0, 1.0]",
                    "6. Ultravox LLM: Processes numpy array with <|audio|> placeholder",
                    "7. Ultravox LLM: Returns text transcript + AI response",
                    "8. TTS: Converts AI text → audio bytes",
                    "9. Response: Returns {transcript, text, audio} to client"
                ],
                "critical_note": "Controllers ONLY validate format. Audio conversion happens in Orchestrator."
            },

            # Backend Mode Info
            "backend_mode": {
                "mode": "IN-PROCESS (with HTTP fallback)" if self.in_process_mode else "HTTP (with fallback)",
                "in_process_enabled": self.in_process_mode,
                "gpu_available": self.gpu_available,
                "in_process_calls": self.stats["in_process_count"],
                "http_fallback_calls": self.stats["http_fallback_count"]
            }
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            "total_turns": 0,
            "successful_turns": 0,
            "failed_turns": 0,
            "primary_llm_count": 0,
            "fallback_llm_count": 0,
            "in_process_count": 0,
            "http_fallback_count": 0,
            "total_processing_time": 0.0
        }
        logger.info("📊 Statistics reset")

    async def process_text_conversation(self,
                                       message: str,
                                       session_id: str,
                                       voice_id: Optional[str] = None,
                                       scenario_id_override: Optional[str] = None) -> Dict[str, Any]:
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
        self.stats["total_turns"] += 1

        try:
            print(f"[ORCHESTRATOR_ENGINE] 💬 Processing text conversation: session={session_id}, message={message[:50]}...", flush=True)
            logger.info(f"💬 Processing text conversation: session={session_id}, message={message[:50]}...")

            # ==========================================
            # STEP 1: Get Session and Scenario Context
            # ==========================================
            session_data = await self.clients["session"].get_session(session_id)
            system_prompt = """You are a helpful AI assistant. Your task is to:

1. LISTEN CAREFULLY to the user's message and identify the specific question or topic
2. ANSWER directly and accurately
3. Provide a concise, helpful response
4. DO NOT ask unrelated questions
5. Focus on being helpful and accurate

Respond naturally in a conversational tone."""
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
                            "system_prompt": base_system_prompt
                        }

                        # Use system prompt (structured validation will happen in LLM call)
                        system_prompt = base_system_prompt

                # Get conversation history for context
                if conversation_id:
                    messages = await self.clients["conversation_store"].get_context(
                        conversation_id,
                        limit=10
                    )
                    # Convert to format expected by LLM
                    conversation_history = self._format_conversation_history(messages)
                    logger.info(f"📚 Loaded {len(messages)} previous messages for context")

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
                        conversation_id=conversation_id,
                        session_id=session_id
                    )
                    created_session_id = session_result.get("id")
                    logger.info(f"✅ Created new session: {created_session_id}")

                except Exception as e:
                    # Session/Conversation services are optional - continue without them
                    logger.warning(f"⚠️  Session/Conversation services unavailable - continuing without history")
                    logger.debug(f"   Error details: {e}")
                    # Continue without conversation_id - won't save history but conversation will still work

                # Get scenario data if scenario_id is provided
                if scenario_id:
                    logger.info(f"🔍 DEBUG: Fetching scenario (no session) with ID: {scenario_id}")
                    scenario_data = await self.clients["scenarios"].get_scenario(scenario_id)
                    logger.info(f"🔍 DEBUG: scenario_data (no session) = {scenario_data}")
                    if scenario_data:
                        base_system_prompt = scenario_data.get("system_prompt")
                        logger.info(f"📝 Using scenario (no session): {scenario_data.get('name', scenario_id)}")

                        # Build scenario context for structured LLM
                        scenario_context = {
                            "type": scenario_data.get("type", "conversation"),
                            "expected_topics": scenario_data.get("expected_topics", []),
                            "ai_role": scenario_data.get("ai_role", "assistant"),
                            "user_role": scenario_data.get("user_role", "user"),
                            "language": scenario_data.get("language", "pt-BR"),
                            "system_prompt": base_system_prompt
                        }

                        # Use system prompt (structured validation will happen in LLM call)
                        system_prompt = base_system_prompt

            # ==========================================
            # STEP 2: Call LLM (Structured with validation OR simple text)
            # ==========================================
            print(f"[DEBUG] Step 2: scenario_id={scenario_id}, session_data={session_data is not None}", flush=True)
            validation_metrics = None

            # DEBUG: Log scenario_id and scenario_context existence
            print(f"[DEBUG] Checking scenario_context in locals: {'scenario_context' in locals()}", flush=True)
            logger.info(f"🔍 DEBUG: scenario_id={scenario_id}, scenario_context_exists={'scenario_context' in locals()}")
            if 'scenario_context' in locals():
                logger.info(f"🔍 DEBUG: scenario_context keys={list(scenario_context.keys())}")

            # If scenario exists, use structured LLM (validation + response in one call)
            if scenario_id and 'scenario_context' in locals():
                try:
                    logger.info("🤖 Using structured LLM (validation + response in single call)...")

                    # Initialize structured LLM client (lazy init)
                    if not hasattr(self, 'structured_llm'):
                        from .structured_llm_client import StructuredLLMClient
                        self.structured_llm = StructuredLLMClient()
                        await self.structured_llm.initialize(self.http_session)

                    # Single LLM call with structured output (Pydantic validated!)
                    validated_response = await self.structured_llm.generate_with_validation(
                        user_message=message,
                        scenario_context=scenario_context,
                        conversation_history=conversation_history
                    )

                    # Extract text response and validation
                    text_response = validated_response.assistant_response
                    validation_metrics = {
                        "coherence_score": validated_response.validation.coherence_score,
                        "in_scope": validated_response.validation.in_scope,
                        "should_redirect": validated_response.validation.should_redirect,
                        "found_topics": validated_response.validation.found_topics,
                        "missing_topics": validated_response.validation.missing_topics,
                        "reason": validated_response.validation.reason
                    }

                    logger.info(
                        f"✅ Structured LLM response: "
                        f"coherence={validation_metrics['coherence_score']:.2f}, "
                        f"in_scope={validation_metrics['in_scope']}, "
                        f"redirect={validation_metrics['should_redirect']}"
                    )

                    self.stats["primary_llm_count"] += 1

                except Exception as e:
                    logger.warning(f"⚠️ Structured LLM failed: {e}, falling back to simple LLM")
                    # Fallback to simple external_llm if structured fails
                    text_response = await self.clients["llm"].generate(
                        text=message,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history
                    )
                    logger.info(f"🤖 LLM response (fallback): {text_response[:100]}...")
                    self.stats["primary_llm_count"] += 1

            # No scenario - use simple external_llm
            else:
                try:
                    # Call external LLM service using generate() method
                    text_response = await self.clients["llm"].generate(
                        text=message,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history
                    )

                    logger.info(f"🤖 LLM response: {text_response[:100]}...")
                    self.stats["primary_llm_count"] += 1

                except ServiceClientError as e:
                    logger.error(f"❌ External LLM failed: {e}")
                    self.stats["failed_turns"] += 1
                    return {
                        "success": False,
                        "error": f"LLM processing failed: {str(e)}",
                        "response": "",
                        "session_id": session_id
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
                        format="opus"
                    )

                    # Convert to base64 for JSON transport
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    logger.info(f"✅ Audio (local TTS) generated: {len(audio_bytes)} bytes → {len(audio_base64)} base64 chars")

                except ServiceClientError as e:
                    logger.warning(f"⚠️  Local TTS service unavailable - trying external TTS")
                    logger.debug(f"   Local TTS error: {e}")

                    # Fallback to external TTS (HuggingFace)
                    try:
                        audio_bytes = await self.clients["tts"].synthesize(
                            text=text_response,
                            voice="af_heart",
                            format="wav"
                        )

                        # Convert to base64 for JSON transport
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        logger.info(f"✅ Audio (external TTS) generated: {len(audio_bytes)} bytes → {len(audio_base64)} base64 chars")

                    except ServiceClientError as e2:
                        logger.warning(f"⚠️  External TTS also failed - continuing without audio")
                        logger.debug(f"   External TTS error: {e2}")
                        # Continue without audio - don't fail the whole request

                except Exception as e:
                    logger.warning(f"⚠️  TTS service unavailable - continuing without audio")
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
                        ai_audio=None  # No audio in text mode
                    )
                    logger.info(f"💾 Turn saved to conversation {conversation_id}")
                except ServiceClientError as e:
                    logger.warning(f"⚠️  Failed to save turn: {e}")

            # ==========================================
            # STEP 4: Return Response
            # ==========================================
            processing_time = time.time() - start_time
            self.stats["successful_turns"] += 1
            self.stats["total_processing_time"] += processing_time

            # Get updated message count
            messages_count = len(conversation_history) + 2  # +2 for current turn

            response = {
                "success": True,
                "response": text_response,
                "session_id": session_id,
                "audio": audio_base64,  # Opus 24kHz base64 encoded (if voice_id provided)
                "context_size": len(conversation_history),
                "messages_count": messages_count,
                "metrics": {
                    "processing_time_ms": processing_time * 1000,
                    "timestamp": time.time()
                }
            }

            # Add validation metrics if available (from structured LLM)
            if validation_metrics:
                response["validation"] = validation_metrics

            return response

        except Exception as e:
            logger.error(f"❌ Unexpected error in text conversation: {e}")
            logger.exception("Full traceback:")
            self.stats["failed_turns"] += 1
            return {
                "success": False,
                "error": f"Internal error: {str(e)}",
                "response": "",
                "session_id": session_id
            }

    async def process_turn_structured(
        self,
        audio_data: bytes,
        session_id: str,
        sample_rate: int = 16000,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
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

        from datetime import datetime

        start_time = time.time()
        self.stats["total_turns"] += 1

        try:
            logger.info(f"🎤 Processing structured turn: session={session_id}, audio={len(audio_data)} bytes")

            # ==========================================
            # STEP 1: STT (External Groq Whisper)
            # ==========================================
            logger.info("📝 Transcribing audio with External STT (Groq Whisper)...")

            try:
                transcribed_text = await self.clients["stt"].transcribe(
                    audio_data=audio_data,
                    language="pt"  # or auto-detect from scenario
                )
                logger.info(f"✅ Transcribed: {transcribed_text[:100]}...")
            except ServiceClientError as e:
                logger.error(f"❌ STT failed: {e}")
                self.stats["failed_turns"] += 1
                return {
                    "success": False,
                    "error": f"STT failed: {e}",
                    "session_id": session_id
                }

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
                        "system_prompt": scenario_data.get("system_prompt", "")
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
                    "system_prompt": "You are a helpful AI assistant."
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
            if not hasattr(self, 'structured_llm'):
                from .structured_llm_client import StructuredLLMClient
                self.structured_llm = StructuredLLMClient()
                await self.structured_llm.initialize(self.http_session)

            try:
                # Single LLM call with structured output (Pydantic validated!)
                validated_response = await self.structured_llm.generate_with_validation(
                    user_message=transcribed_text,
                    scenario_context=scenario_context,
                    conversation_history=conversation_history
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
                self.stats["failed_turns"] += 1
                return {
                    "success": False,
                    "error": f"LLM processing failed: {e}",
                    "session_id": session_id
                }

            # ==========================================
            # STEP 4: TTS (HTTP Service)
            # ==========================================
            logger.info("🔊 Generating audio with TTS service...")

            audio_response = None
            try:
                audio_response = await self.clients["tts"].synthesize(
                    text=assistant_text,
                    voice_id=voice_id
                )
                logger.info(f"✅ TTS generated: {len(audio_response)} bytes")
            except ServiceClientError as e:
                logger.warning(f"⚠️ Local TTS failed: {e}, trying external...")

                # Fallback to external TTS
                try:
                    audio_response = await self.clients["external_tts"].synthesize(
                        text=assistant_text,
                        voice="af_heart",
                        format="wav"
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
                        ai_audio=audio_response
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
                            "validated_at": datetime.utcnow().isoformat()
                        }
                    )
                    logger.info("📊 Scenario state updated")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update scenario state: {e}")

            # ==========================================
            # STEP 7: Return Response
            # ==========================================
            processing_time = time.time() - start_time
            self.stats["successful_turns"] += 1
            self.stats["total_processing_time"] += processing_time

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
                    "reason": validation.reason
                },
                "metadata": {
                    "sentiment": metadata.sentiment if metadata else None,
                    "intent": metadata.intent if metadata else None,
                    "language_quality": metadata.language_quality if metadata else None,
                    "confidence": metadata.confidence if metadata else None
                },
                "metrics": {
                    "processing_time_ms": int(processing_time * 1000),
                    "stt_provider": "external_groq",
                    "llm_provider": "structured_groq_pydantic",
                    "tts_provider": "http_service",
                    "has_audio": audio_response is not None
                }
            }

        except Exception as e:
            logger.error(f"❌ Structured turn error: {e}", exc_info=True)
            self.stats["failed_turns"] += 1
            return {
                "success": False,
                "error": f"Orchestration failed: {str(e)}",
                "session_id": session_id
            }

    async def process_turn_with_talker(
        self,
        audio_data: bytes,
        session_id: str,
        sample_rate: int = 16000,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process conversation turn using Talker abstraction (SIMPLIFIED VERSION)

        This method delegates the audio→text→audio pipeline to the appropriate
        Talker (InternalTalker or ExternalTalker), making the orchestration
        much simpler and cleaner.

        Args:
            audio_data: Input audio bytes
            session_id: Session identifier
            sample_rate: Audio sample rate (default 16000)
            voice_id: TTS voice ID (optional)

        Returns:
            Dict with success, text, audio, transcript, talker, and metrics
        """
        return await process_turn_with_talker(
            self, audio_data, session_id, sample_rate, voice_id
        )

    # ============================================================================
    # STREAMING METHODS WITH ADAPTIVE PARAMETERS (Phase 2: Streaming JSON)
    # ============================================================================

    async def process_turn_streaming(
        self,
        audio_data: bytes,
        session_id: str,
        sample_rate: int = 16000,
        voice_id: Optional[str] = None
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
                        conversation_id,
                        limit=5
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
                    is_final=True
                )
                return

            # ==========================================
            # STEP 3: Stream LLM response (with chunks)
            # ==========================================
            sequence = 0
            llm_start = time.time()

            async for text_chunk in self._stream_llm_response(
                user_transcript,
                conversation_history,
                session_data
            ):
                llm_text_buffer += text_chunk
                sequence += 1

                # Yield text chunk
                yield StreamingJSONEvent(
                    event="text_chunk",
                    sequence=sequence,
                    data=text_chunk,
                    timestamp=time.time()
                )

            llm_duration = time.time() - llm_start
            logger.info(f"✅ LLM: Generated {len(llm_text_buffer)} chars ({llm_duration*1000:.0f}ms)")

            # ==========================================
            # STEP 4: Analyze response (async, in-parallel)
            # ==========================================
            analysis = await self._generate_analysis(
                user_transcript,
                llm_text_buffer
            )
            sequence += 1

            yield StreamingJSONEvent(
                event="analysis",
                sequence=sequence,
                data=analysis.model_dump(),
                timestamp=time.time()
            )

            # ==========================================
            # STEP 5: Generate adaptive instructions
            # ==========================================
            adaptive_instructions = await self._generate_adaptive_instructions(
                user_transcript,
                llm_text_buffer,
                analysis
            )
            sequence += 1

            yield StreamingJSONEvent(
                event="adaptive_instructions",
                sequence=sequence,
                data=adaptive_instructions.model_dump(),
                timestamp=time.time()
            )

            # ==========================================
            # STEP 6: Error correction detection
            # ==========================================
            error_correction = await self._detect_error_patterns(
                user_transcript,
                llm_text_buffer
            )
            if error_correction.correction_needed or error_correction.pattern_detected:
                sequence += 1
                yield StreamingJSONEvent(
                    event="error_correction",
                    sequence=sequence,
                    data=error_correction.model_dump(),
                    timestamp=time.time()
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
                    "session_id": session_id
                },
                timestamp=time.time(),
                is_final=True
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
                is_final=True
            )

    async def _stream_llm_response(self, user_input: str, history: list, session_data: dict):
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
            yield f"[Erro no LLM: {str(e)}]"

    async def _generate_analysis(self, user_input: str, llm_output: str):
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
            elif len(llm_output) < 50:
                response_type = "confirmation"
            elif any(word in llm_output.lower() for word in ["sugestão", "recomendo", "tente"]):
                response_type = "suggestion"

            # Detect theme from user input
            theme = "general"
            theme_keywords = {
                "transportation": ["táxi", "uber", "transporte", "ônibus", "carro", "car"],
                "weather": ["tempo", "chuva", "sol", "temperatura"],
                "food": ["comida", "restaurante", "pizza", "café"],
                "travel": ["viagem", "hotel", "passagem", "destino"]
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
                confidence=0.85
            )

        except Exception as e:
            logger.error(f"⚠️ Analysis generation failed: {e}")
            from src.core.shared.models.response_models import ConversationAnalysis

            return ConversationAnalysis(
                response_type="other",
                theme="general",
                tone="helpful",
                confidence=0.5
            )

    async def _generate_adaptive_instructions(self, user_input: str, llm_output: str, analysis):
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
            estimated_turns = 3
            if len(llm_output) > 100:
                estimated_turns = 2  # Long response, likely nearing resolution
            elif len(user_input) > 100:
                estimated_turns = 4  # Complex question, might need more

            # Generate prefix for next prompt
            next_prompt_prefix = f"""Usuário anterior perguntou: {user_input[:50]}...
Você respondeu sobre: {analysis.theme}
Tom a manter: {analysis.tone}
Próximos tópicos esperados: {', '.join(expected_topics)}

Próxima resposta:"""

            return AdaptiveInstructions(
                next_prompt_prefix=next_prompt_prefix,
                tone_adjustment="maintain_current",
                verbosity="medium" if len(llm_output) > 100 else "concise",
                expected_next_topics=expected_topics,
                estimated_turns_remaining=estimated_turns
            )

        except Exception as e:
            logger.error(f"⚠️ Adaptive instructions generation failed: {e}")
            from src.core.shared.models.response_models import AdaptiveInstructions

            return AdaptiveInstructions(
                next_prompt_prefix="Continue the conversation naturally.",
                tone_adjustment="maintain_current",
                verbosity="medium",
                expected_next_topics=["clarification"],
                estimated_turns_remaining=3
            )

    async def _detect_error_patterns(self, user_input: str, llm_output: str):
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
            if "location" in user_input.lower() and len(llm_output) < 20:
                pattern_detected = "location_format_error"
                suggested_clarification = "Pode detalhar o endereço? (rua, número, bairro)"

            # Detect if user seems confused
            elif any(word in user_input.lower() for word in ["?", "hã", "o que", "como"]):
                if len(llm_output) < 50:
                    pattern_detected = "possible_confusion"
                    suggested_clarification = "Deixe-me explicar melhor..."

            return ErrorCorrection(
                pattern_detected=pattern_detected,
                suggested_clarification=suggested_clarification,
                correction_needed=pattern_detected is not None,
                correction_urgency="medium" if pattern_detected else "low"
            )

        except Exception as e:
            logger.error(f"⚠️ Error pattern detection failed: {e}")
            from src.core.shared.models.response_models import ErrorCorrection

            return ErrorCorrection(
                pattern_detected=None,
                correction_needed=False
            )

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up ConversationOrchestrator...")
        if self.http_session:
            await self.http_session.close()

        # Cleanup structured LLM client
        if hasattr(self, 'structured_llm'):
            await self.structured_llm.cleanup()

        # Cleanup Talker
        if self.talker:
            await self.talker.cleanup()

        logger.info("✅ Cleanup complete")
