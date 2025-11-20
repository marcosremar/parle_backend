#!/usr/bin/env python3
"""
Fallback Manager - Manages LLM failover using circuit breaker pattern
Coordinates failover between Primary LLM (Ultravox) and Fallback LLM (Groq)
"""

import logging
from typing import Dict, Any, Optional, Tuple, Callable
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .utils.pipeline.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from config.settings import get_pipeline_failover_settings
from .service_clients import LLMClient, ExternalLLMClient, STTClient, ExternalUltravoxClient, ServiceClientError

logger = logging.getLogger(__name__)


class FallbackManager:
    """
    Manages automatic failover between primary and secondary LLM

    Features:
    - 2-tier circuit breaker pattern for automatic failover
    - Ultravox (primary) → External Ultravox (secondary)
    - Preserves conversation context across failover
    - Tracks which LLM is active
    - Automatic recovery when primary comes back
    """

    def __init__(self,
                 primary_llm: LLMClient,
                 secondary_llm: ExternalUltravoxClient):
        """
        Initialize failover manager with 2-tier fallback

        Args:
            primary_llm: Ultravox LLM client (integrated STT + LLM) - GPU-based
            secondary_llm: External Ultravox client (Groq STT + LLM) - Cloud-based, Ultravox-compatible
        """
        self.primary_llm = primary_llm
        self.secondary_llm = secondary_llm

        # Initialize circuit breaker with settings from config
        try:
            settings = get_pipeline_failover_settings()
            self.circuit_breaker = CircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=settings.failure_threshold,
                    recovery_timeout=settings.recovery_timeout,
                    half_open_max_calls=settings.half_open_max_calls,
                    primary_timeout=settings.primary_timeout,
                    fallback_timeout=settings.fallback_timeout
                )
            )
            logger.info("🔌 Circuit breaker initialized for automatic LLM failover")
            logger.info(f"   Failure threshold: {settings.failure_threshold}")
            logger.info(f"   Recovery timeout: {settings.recovery_timeout}s")
        except Exception as e:
            logger.error(f"❌ Failed to initialize circuit breaker: {e}")
            # Create default circuit breaker
            self.circuit_breaker = CircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=30.0,
                    half_open_max_calls=1,
                    primary_timeout=10.0,
                    fallback_timeout=10.0
                )
            )

    async def call_llm_with_failover(self,
                                     audio_data: bytes,
                                     sample_rate: int,
                                     system_prompt: Optional[str] = None,
                                     conversation_id: Optional[str] = None,
                                     conversation_history: Optional[list] = None,
                                     force_external_llm: bool = False) -> Dict[str, Any]:
        """
        Call LLM with automatic 2-tier failover
        Preserves conversation context during failover for seamless experience

        Failover chain:
        1. Primary: Ultravox (GPU-based, integrated STT + LLM)
        2. Secondary: External Ultravox (Groq STT + LLM, same interface)

        Args:
            audio_data: Audio bytes to process
            sample_rate: Audio sample rate
            system_prompt: Optional system prompt for LLM
            conversation_id: Optional conversation ID for context
            conversation_history: Optional conversation history
            force_external_llm: Force use of external LLM (skip primary) for benchmarking

        Returns:
            Dict with:
                - success: bool
                - text: Response text
                - transcript: User input transcript (if available)
                - llm_used: "primary" or "fallback"
                - circuit_state: Current circuit breaker state
        """

        async def primary_fn(context: Dict[str, Any]) -> Dict[str, Any]:
            """
            Primary LLM function (GPU Ultravox with integrated STT)
            """
            logger.info("🎯 Trying primary LLM (GPU Ultravox)...")

            result = await self.primary_llm.process_audio(
                audio_data=context["audio_data"],
                sample_rate=context["sample_rate"],
                system_prompt=context.get("system_prompt")
            )

            return {
                "text": result["text"],
                "transcript": result.get("transcript", ""),
                "metadata": result.get("metadata", {}),
                "llm_tier": "primary"
            }

        async def secondary_fn(context: Dict[str, Any]) -> Dict[str, Any]:
            """
            Secondary LLM function (External Ultravox - Groq STT + LLM)
            Same interface as primary, just cloud-based
            """
            logger.info("🔄 Primary failed, trying secondary LLM (External Ultravox)...")

            result = await self.secondary_llm.process_audio(
                audio_data=context["audio_data"],
                sample_rate=context["sample_rate"],
                system_prompt=context.get("system_prompt")
            )

            return {
                "text": result["text"],
                "transcript": result.get("transcript", ""),
                "metadata": result.get("metadata", {}),
                "llm_tier": "secondary"
            }

        # Call with 2-tier fallback: try primary, then secondary
        context = {
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "system_prompt": system_prompt,
            "conversation_id": conversation_id,
            "conversation_history": conversation_history
        }

        # Try each tier in order (or skip primary if force_external_llm=True)
        try:
            # If force_external_llm, skip primary and go directly to secondary
            if force_external_llm:
                logger.info("🔀 force_external_llm=True, skipping primary LLM and using External Ultravox...")
                result = await secondary_fn(context)
                logger.info("✅ External LLM (forced) succeeded")
                llm_tier = result.pop("llm_tier", "fallback")

                return {
                    "success": True,
                    "text": result["text"],
                    "transcript": result.get("transcript", ""),
                    "llm_used": "fallback",  # Always return "fallback" for secondary
                    "circuit_state": self.get_circuit_state(),
                    "metadata": result.get("metadata", {})
                }

            # Normal failover: Try primary first
            try:
                result = await primary_fn(context)
                logger.info("✅ Primary LLM succeeded")
                llm_tier = result.pop("llm_tier", "primary")

                return {
                    "success": True,
                    "text": result["text"],
                    "transcript": result.get("transcript", ""),
                    "llm_used": llm_tier,
                    "circuit_state": self.get_circuit_state(),
                    "metadata": result.get("metadata", {})
                }
            except Exception as primary_error:
                logger.warning(f"⚠️ Primary LLM failed: {primary_error}")

                # Try secondary
                try:
                    result = await secondary_fn(context)
                    logger.info("✅ Secondary LLM succeeded (External Ultravox)")
                    llm_tier = result.pop("llm_tier", "fallback")  # Changed from "secondary" to "fallback"

                    return {
                        "success": True,
                        "text": result["text"],
                        "transcript": result.get("transcript", ""),
                        "llm_used": "fallback",  # Always return "fallback" for secondary
                        "circuit_state": self.get_circuit_state(),
                        "metadata": result.get("metadata", {})
                    }
                except Exception as secondary_error:
                    # Both tiers failed
                    logger.error(f"❌ Both LLM tiers failed: Primary={primary_error}, Secondary={secondary_error}")
                    raise Exception(f"Both LLM tiers failed. Last error: {secondary_error}")

        except Exception as e:
            logger.error(f"❌ Complete LLM failover chain failed: {e}")
            return {
                "success": False,
                "error": f"All LLMs failed: {str(e)}",
                "llm_used": "failed",
                "circuit_state": self.get_circuit_state()
            }

    def get_circuit_state(self) -> Dict[str, Any]:
        """
        Get current circuit breaker state

        Returns:
            Dict with circuit breaker stats
        """
        state = self.circuit_breaker.get_state()
        return {
            "state": state.get("state", "unknown"),
            "failure_count": state.get("failure_count", 0),
            "success_count": state.get("success_count", 0),
            "last_failure_time": state.get("last_failure_time"),
            "using_fallback": state.get("state") in ["open", "half_open"]
        }

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of both LLM tiers

        Returns:
            Dict with health status of each LLM tier
        """
        health = {}

        try:
            health["primary_llm"] = await self.primary_llm.health_check()
        except asyncio.TimeoutError:
            logger.warning("Primary LLM health check timeout")
            health["primary_llm"] = False
        except (ConnectionError, AttributeError) as e:
            logger.warning(f"Primary LLM health check failed: {e}")
            health["primary_llm"] = False
        except Exception as e:
            logger.error(f"Unexpected error checking primary LLM health: {e}")
            health["primary_llm"] = False

        try:
            health["secondary_llm"] = await self.secondary_llm.health_check()
        except asyncio.TimeoutError:
            logger.warning("Secondary LLM health check timeout")
            health["secondary_llm"] = False
        except (ConnectionError, AttributeError) as e:
            logger.warning(f"Secondary LLM health check failed: {e}")
            health["secondary_llm"] = False
        except Exception as e:
            logger.error(f"Unexpected error checking secondary LLM health: {e}")
            health["secondary_llm"] = False

        return health

    def reset_circuit_breaker(self):
        """Reset circuit breaker state (for testing/manual recovery)"""
        self.circuit_breaker.reset()
        logger.info("🔄 Circuit breaker manually reset")
