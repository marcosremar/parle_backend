"""
LLM Strategy Pattern

Eliminates if/else chains for in-process vs HTTP LLM processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

from ..constants import DEFAULT_SAMPLE_RATE, AUDIO_NORMALIZATION_DIVISOR, LLMProvider

logger = logging.getLogger(__name__)


class LLMStrategy(ABC):
    """Abstract base class for LLM processing strategies."""

    @abstractmethod
    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int,
        system_prompt: str,
        conversation_history: List[Dict[str, Any]],
        conversation_id: Optional[str],
        force_external_llm: bool
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Process audio and generate text response.
        
        Args:
            audio_data: Input audio bytes
            sample_rate: Audio sample rate
            system_prompt: System prompt for LLM
            conversation_history: Conversation history
            conversation_id: Optional conversation ID
            force_external_llm: Force external LLM
            
        Returns:
            Tuple of (text_response, llm_used, llm_result_dict)
        """
        pass


class InProcessLLMStrategy(LLMStrategy):
    """In-process LLM strategy using direct module calls."""

    def __init__(
        self,
        llm_instance: Any,
        stats_tracker: Any
    ) -> None:
        """
        Initialize in-process LLM strategy.
        
        Args:
            llm_instance: In-process LLM instance
            stats_tracker: Stats tracker for metrics
        """
        self.llm_instance = llm_instance
        self.stats_tracker = stats_tracker

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int,
        system_prompt: str,
        conversation_history: List[Dict[str, Any]],
        conversation_id: Optional[str],
        force_external_llm: bool
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Process audio using in-process LLM."""
        try:
            logger.info("⚡ Using in-process LLM (ultra-low latency)...")
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / AUDIO_NORMALIZATION_DIVISOR
            
            result = await self.llm_instance.process_audio(
                audio_array=audio_array,
                sample_rate=sample_rate,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            
            text_response = result.get("text", "")
            llm_used = LLMProvider.IN_PROCESS
            
            logger.info(f"✅ In-process LLM response: {text_response[:100]}...")
            self.stats_tracker.increment_in_process_count()
            
            return text_response, llm_used, {
                "success": True,
                "text": text_response,
                "llm_used": llm_used,
                "transcript": result.get("transcript", "")
            }
        except Exception as e:
            logger.warning(f"⚠️ In-process LLM failed: {e}")
            logger.info("   Falling back to HTTP mode...")
            self.stats_tracker.increment_http_fallback_count()
            raise  # Let fallback strategy handle it


class HTTPLLMStrategy(LLMStrategy):
    """HTTP LLM strategy using fallback manager."""

    def __init__(
        self,
        fallback_manager: Any,
        stats_tracker: Any
    ) -> None:
        """
        Initialize HTTP LLM strategy.
        
        Args:
            fallback_manager: Fallback manager for LLM failover
            stats_tracker: Stats tracker for metrics
        """
        self.fallback_manager = fallback_manager
        self.stats_tracker = stats_tracker

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int,
        system_prompt: str,
        conversation_history: List[Dict[str, Any]],
        conversation_id: Optional[str],
        force_external_llm: bool
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Process audio using HTTP LLM with failover."""
        llm_result = await self.fallback_manager.call_llm_with_failover(
            audio_data=audio_data,
            sample_rate=sample_rate,
            system_prompt=system_prompt,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            force_external_llm=force_external_llm
        )
        
        text_response = llm_result["text"]
        llm_used = llm_result["llm_used"]
        
        # Update stats
        if llm_used == LLMProvider.PRIMARY:
            self.stats_tracker.increment_primary_llm_count()
        elif llm_used == LLMProvider.FALLBACK:
            self.stats_tracker.increment_fallback_llm_count()
        
        logger.info(f"🤖 LLM ({llm_used}) response: {text_response[:100]}...")
        
        return text_response, llm_used, llm_result


class LLMStrategyFactory:
    """Factory for creating appropriate LLM strategy."""

    @staticmethod
    def create_strategy(
        in_process_mode: bool,
        llm_instance: Optional[Any],
        fallback_manager: Any,
        stats_tracker: Any
    ) -> LLMStrategy:
        """
        Create appropriate LLM strategy based on configuration.
        
        Args:
            in_process_mode: Whether in-process mode is enabled
            llm_instance: Optional in-process LLM instance
            fallback_manager: Fallback manager
            stats_tracker: Stats tracker
            
        Returns:
            LLMStrategy instance
        """
        if in_process_mode and llm_instance:
            return InProcessLLMStrategy(llm_instance, stats_tracker)
        else:
            return HTTPLLMStrategy(fallback_manager, stats_tracker)
