"""
MetricsCalculator - Calculate tokens/s and throughput correctly
Tracks metrics per turn and across scenarios
"""

import time
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn"""
    turn_number: int
    latency_ms: float
    first_token_latency_ms: Optional[float]
    tokens_generated: int
    tokens_per_second: float

    # Stage breakdown
    tts_generation_ms: float  # TTS for question
    stt_transcription_ms: float  # Whisper STT
    llm_processing_ms: float  # Ultravox processing
    tts_synthesis_ms: float  # TTS for response

    # Memory and context
    memory_usage_mb: float
    context_tokens: int
    context_retention_score: float

    # Response quality
    response_text: str
    response_language: str
    voice_used: str


@dataclass
class ScenarioMetrics:
    """Metrics for a complete scenario"""
    scenario_id: str
    scenario_name: str
    total_turns: int
    turns: List[TurnMetrics] = field(default_factory=list)

    # Aggregate metrics
    average_latency_ms: float = 0
    latency_increase_rate: float = 0  # % increase per turn
    total_tokens: int = 0
    average_tokens_per_second: float = 0

    # Context effectiveness
    context_coherence_score: float = 0
    memory_effectiveness: float = 0

    # Performance consistency
    latency_variance: float = 0
    throughput_variance: float = 0


class MetricsCalculator:
    """
    Calculate and track metrics for conversation testing
    Properly calculates tokens/second and throughput
    """

    def __init__(self):
        self.scenarios: Dict[str, ScenarioMetrics] = {}
        self.current_scenario: Optional[str] = None

    def start_scenario(self, scenario_id: str, scenario_name: str) -> None:
        """Start tracking a new scenario"""
        self.current_scenario = scenario_id
        self.scenarios[scenario_id] = ScenarioMetrics(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            total_turns=0
        )
        logger.info(f"📊 Started metrics tracking for scenario: {scenario_name}")

    def record_turn(
        self,
        turn_number: int,
        start_time: float,
        end_time: float,
        first_token_time: Optional[float],
        response_text: str,
        stage_timings: Dict[str, float],
        memory_usage_mb: float = 0,
        context_tokens: int = 0
    ) -> TurnMetrics:
        """
        Record metrics for a single turn

        Args:
            turn_number: Turn number in conversation
            start_time: Request start timestamp
            end_time: Response complete timestamp
            first_token_time: Time when first token received
            response_text: Generated response text
            stage_timings: Breakdown by processing stage
            memory_usage_mb: Current memory usage
            context_tokens: Tokens in context
        """

        if not self.current_scenario:
            raise ValueError("No scenario started")

        # Calculate latencies
        total_latency_ms = (end_time - start_time) * 1000
        first_token_latency_ms = None
        if first_token_time:
            first_token_latency_ms = (first_token_time - start_time) * 1000

        # Calculate tokens (estimate if not provided)
        tokens_generated = self._estimate_tokens(response_text)

        # Calculate tokens per second
        generation_time_s = end_time - start_time
        if generation_time_s > 0:
            tokens_per_second = tokens_generated / generation_time_s
        else:
            tokens_per_second = 0

        # Extract stage timings
        turn_metrics = TurnMetrics(
            turn_number=turn_number,
            latency_ms=total_latency_ms,
            first_token_latency_ms=first_token_latency_ms,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            tts_generation_ms=stage_timings.get('tts_generation_ms', 0),
            stt_transcription_ms=stage_timings.get('stt_transcription_ms', 0),
            llm_processing_ms=stage_timings.get('llm_processing_ms', 0),
            tts_synthesis_ms=stage_timings.get('tts_synthesis_ms', 0),
            memory_usage_mb=memory_usage_mb,
            context_tokens=context_tokens,
            context_retention_score=0,  # Will be calculated separately
            response_text=response_text,
            response_language=self._detect_language(response_text),
            voice_used=""  # Will be set by runner
        )

        # Add to scenario
        scenario = self.scenarios[self.current_scenario]
        scenario.turns.append(turn_metrics)
        scenario.total_turns += 1

        logger.info(
            f"📈 Turn {turn_number}: "
            f"Latency={total_latency_ms:.0f}ms, "
            f"Tokens={tokens_generated}, "
            f"Throughput={tokens_per_second:.1f} tok/s"
        )

        return turn_metrics

    def calculate_scenario_metrics(self, scenario_id: str) -> ScenarioMetrics:
        """Calculate aggregate metrics for a scenario"""

        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")

        scenario = self.scenarios[scenario_id]

        if not scenario.turns:
            return scenario

        # Calculate average latency
        latencies = [turn.latency_ms for turn in scenario.turns]
        scenario.average_latency_ms = statistics.mean(latencies)
        scenario.latency_variance = statistics.variance(latencies) if len(latencies) > 1 else 0

        # Calculate latency increase rate
        if len(latencies) > 1:
            first_latency = latencies[0]
            last_latency = latencies[-1]
            if first_latency > 0:
                scenario.latency_increase_rate = ((last_latency - first_latency) / first_latency) * 100

        # Calculate total tokens and average throughput
        scenario.total_tokens = sum(turn.tokens_generated for turn in scenario.turns)
        throughputs = [turn.tokens_per_second for turn in scenario.turns if turn.tokens_per_second > 0]
        if throughputs:
            scenario.average_tokens_per_second = statistics.mean(throughputs)
            scenario.throughput_variance = statistics.variance(throughputs) if len(throughputs) > 1 else 0

        # Log summary
        logger.info(
            f"📊 Scenario '{scenario.scenario_name}' Summary:\n"
            f"   Turns: {scenario.total_turns}\n"
            f"   Avg Latency: {scenario.average_latency_ms:.0f}ms\n"
            f"   Latency Increase: {scenario.latency_increase_rate:.1f}%\n"
            f"   Total Tokens: {scenario.total_tokens}\n"
            f"   Avg Throughput: {scenario.average_tokens_per_second:.1f} tok/s"
        )

        return scenario

    def evaluate_context_retention(
        self,
        scenario_id: str,
        expected_contexts: List[List[str]]
    ) -> float:
        """
        Evaluate how well context is retained across turns

        Args:
            scenario_id: Scenario to evaluate
            expected_contexts: List of expected context items per turn

        Returns:
            Context retention score (0.0-1.0)
        """

        if scenario_id not in self.scenarios:
            return 0.0

        scenario = self.scenarios[scenario_id]
        scores = []

        for i, turn in enumerate(scenario.turns):
            if i >= len(expected_contexts):
                break

            expected = expected_contexts[i]
            response_lower = turn.response_text.lower()

            # Check how many expected context items are mentioned
            mentioned = sum(
                1 for item in expected
                if item.lower() in response_lower
            )

            score = mentioned / len(expected) if expected else 1.0
            turn.context_retention_score = score
            scores.append(score)

        overall_score = statistics.mean(scores) if scores else 0.0
        scenario.context_coherence_score = overall_score

        logger.info(f"🧠 Context retention score: {overall_score:.2f}")
        return overall_score

    def get_comparison_metrics(self, scenario_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple scenarios"""

        comparison = {
            'scenarios': [],
            'best_latency': None,
            'best_throughput': None,
            'most_consistent': None
        }

        for scenario_id in scenario_ids:
            if scenario_id not in self.scenarios:
                continue

            scenario = self.scenarios[scenario_id]
            comparison['scenarios'].append({
                'id': scenario_id,
                'name': scenario.scenario_name,
                'avg_latency': scenario.average_latency_ms,
                'avg_throughput': scenario.average_tokens_per_second,
                'latency_variance': scenario.latency_variance,
                'context_score': scenario.context_coherence_score
            })

        # Find best performers
        if comparison['scenarios']:
            comparison['best_latency'] = min(
                comparison['scenarios'],
                key=lambda x: x['avg_latency']
            )['name']

            comparison['best_throughput'] = max(
                comparison['scenarios'],
                key=lambda x: x['avg_throughput']
            )['name']

            comparison['most_consistent'] = min(
                comparison['scenarios'],
                key=lambda x: x['latency_variance']
            )['name']

        return comparison

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics as dictionary"""
        return {
            scenario_id: {
                'name': scenario.scenario_name,
                'total_turns': scenario.total_turns,
                'average_latency_ms': scenario.average_latency_ms,
                'latency_increase_rate': scenario.latency_increase_rate,
                'total_tokens': scenario.total_tokens,
                'average_tokens_per_second': scenario.average_tokens_per_second,
                'context_coherence_score': scenario.context_coherence_score,
                'turns': [
                    {
                        'turn': turn.turn_number,
                        'latency_ms': turn.latency_ms,
                        'tokens': turn.tokens_generated,
                        'tokens_per_second': turn.tokens_per_second,
                        'stages': {
                            'tts_generation': turn.tts_generation_ms,
                            'stt_transcription': turn.stt_transcription_ms,
                            'llm_processing': turn.llm_processing_ms,
                            'tts_synthesis': turn.tts_synthesis_ms
                        }
                    }
                    for turn in scenario.turns
                ]
            }
            for scenario_id, scenario in self.scenarios.items()
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text"""
        if not text:
            return 0

        # Simple estimation: ~1.3 tokens per word for Portuguese/English
        words = text.split()
        return int(len(words) * 1.3)

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        if not text:
            return "unknown"

        # Portuguese indicators
        pt_indicators = [
            'ção', 'ões', 'ão', 'ém', 'você', 'não',
            'está', 'isso', 'para', 'muito', 'sim'
        ]

        # Count Portuguese indicators
        text_lower = text.lower()
        pt_count = sum(1 for ind in pt_indicators if ind in text_lower)

        return "pt" if pt_count >= 2 else "en"