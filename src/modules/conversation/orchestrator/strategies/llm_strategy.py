"""
LLM Strategy Pattern

HTTP-based LLM processing using fallback manager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

from ..constants import AUDIO_NORMALIZATION_DIVISOR, LLMProvider

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
    """Factory for creating LLM strategy (always HTTP-based)."""

    @staticmethod
    def create_strategy(
        fallback_manager: Any,
        stats_tracker: Any,
        **kwargs  # Accept but ignore legacy parameters (in_process_mode, llm_instance)
    ) -> LLMStrategy:
        """
        Create LLM strategy (always HTTP-based).
        
        Args:
            fallback_manager: Fallback manager
            stats_tracker: Stats tracker
            **kwargs: Ignored (for backward compatibility)
            
        Returns:
            HTTPLLMStrategy instance
        """
        return HTTPLLMStrategy(fallback_manager, stats_tracker)
