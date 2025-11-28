"""
Stats Tracker Engine

Centralized statistics tracking for the orchestrator.
"""

from __future__ import annotations

from typing import Dict, Any
from ..constants import StatsKey


class StatsTracker:
    """Tracks orchestrator statistics and metrics."""

    def __init__(self) -> None:
        """Initialize stats tracker with default values."""
        self.stats: Dict[str, Any] = {
            StatsKey.TOTAL_TURNS: 0,
            StatsKey.SUCCESSFUL_TURNS: 0,
            StatsKey.FAILED_TURNS: 0,
            StatsKey.PRIMARY_LLM_COUNT: 0,
            StatsKey.FALLBACK_LLM_COUNT: 0,
            StatsKey.IN_PROCESS_COUNT: 0,
            StatsKey.HTTP_FALLBACK_COUNT: 0,
            StatsKey.TOTAL_PROCESSING_TIME: 0.0
        }

    def increment_total_turns(self) -> None:
        """Increment total turns counter."""
        self.stats[StatsKey.TOTAL_TURNS] += 1

    def increment_successful_turns(self) -> None:
        """Increment successful turns counter."""
        self.stats[StatsKey.SUCCESSFUL_TURNS] += 1

    def increment_failed_turns(self) -> None:
        """Increment failed turns counter."""
        self.stats[StatsKey.FAILED_TURNS] += 1

    def increment_primary_llm_count(self) -> None:
        """Increment primary LLM usage counter."""
        self.stats[StatsKey.PRIMARY_LLM_COUNT] += 1

    def increment_fallback_llm_count(self) -> None:
        """Increment fallback LLM usage counter."""
        self.stats[StatsKey.FALLBACK_LLM_COUNT] += 1

    def increment_in_process_count(self) -> None:
        """Increment in-process call counter."""
        self.stats[StatsKey.IN_PROCESS_COUNT] += 1

    def increment_http_fallback_count(self) -> None:
        """Increment HTTP fallback counter."""
        self.stats[StatsKey.HTTP_FALLBACK_COUNT] += 1

    def add_processing_time(self, time_seconds: float) -> None:
        """Add processing time to total."""
        self.stats[StatsKey.TOTAL_PROCESSING_TIME] += time_seconds

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with all statistics
        """
        return self.stats.copy()

    def get_average_processing_time_ms(self) -> int:
        """
        Get average processing time in milliseconds.
        
        Returns:
            Average processing time in ms, or 0 if no turns processed
        """
        if self.stats[StatsKey.TOTAL_TURNS] > 0:
            avg_time = (self.stats[StatsKey.TOTAL_PROCESSING_TIME] / 
                       self.stats[StatsKey.TOTAL_TURNS])
            return int(avg_time * 1000)
        return 0

    def get_success_rate(self) -> float:
        """
        Get success rate as a ratio.
        
        Returns:
            Success rate between 0.0 and 1.0
        """
        if self.stats[StatsKey.TOTAL_TURNS] > 0:
            return (self.stats[StatsKey.SUCCESSFUL_TURNS] / 
                   self.stats[StatsKey.TOTAL_TURNS])
        return 0.0

    def get_primary_llm_rate(self) -> float:
        """
        Get primary LLM usage rate as a ratio.
        
        Returns:
            Primary LLM rate between 0.0 and 1.0
        """
        if self.stats[StatsKey.TOTAL_TURNS] > 0:
            return (self.stats[StatsKey.PRIMARY_LLM_COUNT] / 
                   self.stats[StatsKey.TOTAL_TURNS])
        return 0.0

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.stats = {
            StatsKey.TOTAL_TURNS: 0,
            StatsKey.SUCCESSFUL_TURNS: 0,
            StatsKey.FAILED_TURNS: 0,
            StatsKey.PRIMARY_LLM_COUNT: 0,
            StatsKey.FALLBACK_LLM_COUNT: 0,
            StatsKey.IN_PROCESS_COUNT: 0,
            StatsKey.HTTP_FALLBACK_COUNT: 0,
            StatsKey.TOTAL_PROCESSING_TIME: 0.0
        }
