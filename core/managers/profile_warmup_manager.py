"""
Profile-Aware Warmup Manager
Extends WarmupManager with GPU profile-specific warmup strategies
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .warmup_manager import WarmupManager, WarmupResult, WarmupStatus
from .gpu_profile_manager import get_gpu_profile_manager, WarmupConfig
from .metrics import get_metrics_collector
from .structured_logger import get_logger

logger = logging.getLogger(__name__)
structured_logger = get_logger("ProfileWarmupManager")
metrics = get_metrics_collector()


@dataclass
class ProfileWarmupMetrics:
    """Extended warmup metrics with profile-specific data"""
    module: str
    profile_id: str
    status: WarmupStatus
    total_time_ms: float
    iterations_completed: int
    iterations_planned: int
    per_iteration_metrics: List[Dict[str, Any]]
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    error: Optional[str] = None


class ProfileWarmupManager(WarmupManager):
    """
    Profile-aware warmup manager that adapts warmup strategy based on active GPU profile

    Features:
    - Variable iterations per profile (minimal_gpu: 15 LLM iters, full_gpu: 10 iters)
    - Variable batch sizes and sequence lengths
    - Parallel vs sequential warmup based on profile
    - Audio duration testing for STT
    - Text length testing for TTS
    - Voice variation testing
    """

    def __init__(self, profile_id: Optional[str] = None):
        """
        Initialize profile-aware warmup manager

        Args:
            profile_id: GPU profile to use (defaults to active profile)
        """
        # Get profile manager
        self.profile_manager = get_gpu_profile_manager()

        # Get warmup config from profile
        self.profile_id = profile_id or self.profile_manager.active_profile
        self.warmup_config = self.profile_manager.get_warmup_config(self.profile_id)

        if not self.warmup_config:
            raise ValueError(f"No warmup config found for profile '{self.profile_id}'")

        # Initialize parent with profile-specific iterations
        stt_iterations = self.warmup_config.stt.iterations if self.warmup_config.stt else 0
        llm_iterations = self.warmup_config.llm.iterations if self.warmup_config.llm else 0
        tts_iterations = self.warmup_config.tts.iterations if self.warmup_config.tts else 0

        super().__init__(
            stt_iterations=stt_iterations,
            llm_iterations=llm_iterations,
            tts_iterations=tts_iterations,
            timeout_seconds=self.warmup_config.timeout_seconds
        )

        # Profile-specific metrics
        self.profile_metrics: Dict[str, ProfileWarmupMetrics] = {}

        structured_logger.info(
            f"🎯 ProfileWarmupManager initialized for profile '{self.profile_id}'",
            metadata={
                'profile_id': self.profile_id,
                'stt_iterations': stt_iterations,
                'llm_iterations': llm_iterations,
                'tts_iterations': tts_iterations,
                'parallel': self.warmup_config.parallel
            }
        )

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile from list of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        return sorted_values[min(index, len(sorted_values) - 1)]

    async def warmup_stt(self, stt_module) -> WarmupResult:
        """
        Profile-aware STT warmup with variable audio durations

        Tests different audio lengths to warm up all processing paths
        """
        if not self.warmup_config.stt or self.warmup_config.stt.iterations == 0:
            structured_logger.info("⏭️ Skipping STT warmup (disabled in profile)")
            return WarmupResult(
                module='stt',
                status=WarmupStatus.COMPLETED,
                time_ms=0,
                iterations=0
            )

        structured_logger.info(
            f"🎤 Starting profile-aware STT warmup ({self.warmup_config.stt.iterations} iterations)"
        )
        self.warmup_status['stt'] = WarmupStatus.IN_PROGRESS

        start_time = time.time()
        latencies = []
        per_iteration_metrics = []

        # Get test parameters from profile
        test_durations = self.warmup_config.stt.test_audio_durations or [1, 3, 5]
        batch_sizes = self.warmup_config.stt.batch_sizes or [1]
        languages = self.warmup_config.stt.languages or ['pt']

        try:
            with metrics.measure_time("warmup", "profile_stt_total"):
                for i in range(self.warmup_config.stt.iterations):
                    iter_start = time.time()

                    # Cycle through test parameters
                    duration = test_durations[i % len(test_durations)]
                    batch_size = batch_sizes[i % len(batch_sizes)]
                    language = languages[i % len(languages)]

                    # Create synthetic audio
                    sample_rate = 16000
                    audio = np.random.randn(int(sample_rate * duration)) * 0.01
                    audio = audio.astype(np.float32)

                    from src.services.external_stt.transcription.base import STTRequest

                    request = STTRequest(
                        audio=audio,
                        sample_rate=sample_rate,
                        language=language,
                        metadata={
                            'warmup': True,
                            'profile': self.profile_id,
                            'iteration': i + 1,
                            'duration_s': duration,
                            'batch_size': batch_size
                        }
                    )

                    with metrics.measure_time("warmup", f"profile_stt_iter_{i+1}"):
                        response = await stt_module.transcribe_batch(request)

                    iter_time = (time.time() - iter_start) * 1000
                    latencies.append(iter_time)

                    per_iteration_metrics.append({
                        'iteration': i + 1,
                        'duration_s': duration,
                        'batch_size': batch_size,
                        'language': language,
                        'latency_ms': iter_time
                    })

                    structured_logger.info(
                        f"  ✓ STT warmup {i+1}/{self.warmup_config.stt.iterations}: "
                        f"{iter_time:.0f}ms (audio: {duration}s, lang: {language})"
                    )

                    if i < self.warmup_config.stt.iterations - 1:
                        await asyncio.sleep(0.1)

            # Calculate metrics
            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)

            self.warmup_status['stt'] = WarmupStatus.COMPLETED
            self.modules_ready['stt'] = True

            # Store profile-specific metrics
            self.profile_metrics['stt'] = ProfileWarmupMetrics(
                module='stt',
                profile_id=self.profile_id,
                status=WarmupStatus.COMPLETED,
                total_time_ms=total_time,
                iterations_completed=len(latencies),
                iterations_planned=self.warmup_config.stt.iterations,
                per_iteration_metrics=per_iteration_metrics,
                avg_latency_ms=avg_latency,
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                p50_latency_ms=self._calculate_percentile(latencies, 50),
                p95_latency_ms=self._calculate_percentile(latencies, 95)
            )

            result = WarmupResult(
                module='stt',
                status=WarmupStatus.COMPLETED,
                time_ms=total_time,
                iterations=self.warmup_config.stt.iterations,
                metrics={
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'p50_latency_ms': self._calculate_percentile(latencies, 50),
                    'p95_latency_ms': self._calculate_percentile(latencies, 95),
                    'test_durations': test_durations,
                    'languages_tested': list(set(languages)),
                    'latencies': latencies
                }
            )

            self.warmup_results['stt'] = result

            structured_logger.info(
                f"✅ Profile-aware STT warmup complete! "
                f"Avg: {avg_latency:.0f}ms, P95: {self._calculate_percentile(latencies, 95):.0f}ms"
            )

            return result

        except Exception as e:
            self.warmup_status['stt'] = WarmupStatus.FAILED
            self.modules_ready['stt'] = False

            result = WarmupResult(
                module='stt',
                status=WarmupStatus.FAILED,
                time_ms=(time.time() - start_time) * 1000,
                iterations=len(latencies),
                error=str(e)
            )

            self.warmup_results['stt'] = result
            structured_logger.error("❌ Profile-aware STT warmup failed", exception=e)
            raise

    async def warmup_llm(self, llm_module) -> WarmupResult:
        """
        Profile-aware LLM warmup with variable sequence lengths and batch sizes

        Tests different sequence lengths to warm up all KV cache sizes
        """
        if not self.warmup_config.llm or self.warmup_config.llm.iterations == 0:
            structured_logger.info("⏭️ Skipping LLM warmup (disabled in profile)")
            return WarmupResult(
                module='llm',
                status=WarmupStatus.COMPLETED,
                time_ms=0,
                iterations=0
            )

        structured_logger.info(
            f"🤖 Starting profile-aware LLM warmup ({self.warmup_config.llm.iterations} iterations)"
        )
        self.warmup_status['llm'] = WarmupStatus.IN_PROGRESS

        start_time = time.time()
        latencies = []
        per_iteration_metrics = []

        # Get test parameters from profile
        sequence_lengths = self.warmup_config.llm.sequence_lengths or [64, 128, 256]
        batch_sizes = self.warmup_config.llm.batch_sizes or [1]
        with_audio = self.warmup_config.llm.with_audio

        try:
            with metrics.measure_time("warmup", "profile_llm_total"):
                for i in range(self.warmup_config.llm.iterations):
                    iter_start = time.time()

                    # Cycle through test parameters
                    seq_len = sequence_lengths[i % len(sequence_lengths)]
                    batch_size = batch_sizes[i % len(batch_sizes)]

                    # Create prompt with target sequence length
                    base_prompt = "Responda de forma concisa: "
                    padding = "contexto " * max(0, (seq_len - len(base_prompt)) // 10)
                    prompt = base_prompt + padding

                    # Add audio if configured
                    audio_data = None
                    if with_audio:
                        sample_rate = 16000
                        duration = 0.5
                        audio = np.random.randn(int(sample_rate * duration)) * 0.01
                        audio = audio.astype(np.float32)
                        prompt = f"<|audio|>\n{prompt}"
                        audio_data = {
                            'audio_tuple': (audio, sample_rate),
                            'audio_shape': audio.shape,
                            'sample_rate': sample_rate
                        }

                    from src.services.llm.evaluators.base import LLMRequest

                    request = LLMRequest(
                        prompt=prompt,
                        max_tokens=20,
                        temperature=0.1,
                        metadata={
                            'warmup': True,
                            'profile': self.profile_id,
                            'iteration': i + 1,
                            'sequence_length': seq_len,
                            'batch_size': batch_size,
                            'with_audio': with_audio,
                            'audio_data': audio_data
                        }
                    )

                    with metrics.measure_time("warmup", f"profile_llm_iter_{i+1}"):
                        response = await llm_module.generate(request)

                    iter_time = (time.time() - iter_start) * 1000
                    latencies.append(iter_time)

                    per_iteration_metrics.append({
                        'iteration': i + 1,
                        'sequence_length': seq_len,
                        'batch_size': batch_size,
                        'with_audio': with_audio,
                        'latency_ms': iter_time
                    })

                    structured_logger.info(
                        f"  ✓ LLM warmup {i+1}/{self.warmup_config.llm.iterations}: "
                        f"{iter_time:.0f}ms (seq_len: {seq_len}, batch: {batch_size})"
                    )

                    # Longer pause on first iteration (CUDA compilation)
                    if i == 0:
                        await asyncio.sleep(1.0)
                    elif i < self.warmup_config.llm.iterations - 1:
                        await asyncio.sleep(0.2)

            # Calculate metrics
            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)

            self.warmup_status['llm'] = WarmupStatus.COMPLETED
            self.modules_ready['llm'] = True

            # Store profile-specific metrics
            self.profile_metrics['llm'] = ProfileWarmupMetrics(
                module='llm',
                profile_id=self.profile_id,
                status=WarmupStatus.COMPLETED,
                total_time_ms=total_time,
                iterations_completed=len(latencies),
                iterations_planned=self.warmup_config.llm.iterations,
                per_iteration_metrics=per_iteration_metrics,
                avg_latency_ms=avg_latency,
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                p50_latency_ms=self._calculate_percentile(latencies, 50),
                p95_latency_ms=self._calculate_percentile(latencies, 95)
            )

            result = WarmupResult(
                module='llm',
                status=WarmupStatus.COMPLETED,
                time_ms=total_time,
                iterations=self.warmup_config.llm.iterations,
                metrics={
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'p50_latency_ms': self._calculate_percentile(latencies, 50),
                    'p95_latency_ms': self._calculate_percentile(latencies, 95),
                    'first_iter_ms': latencies[0],
                    'sequence_lengths_tested': sequence_lengths,
                    'with_audio': with_audio,
                    'latencies': latencies
                }
            )

            self.warmup_results['llm'] = result

            structured_logger.info(
                f"✅ Profile-aware LLM warmup complete! "
                f"Avg: {avg_latency:.0f}ms, P95: {self._calculate_percentile(latencies, 95):.0f}ms "
                f"(1st: {latencies[0]:.0f}ms)"
            )

            return result

        except Exception as e:
            self.warmup_status['llm'] = WarmupStatus.FAILED
            self.modules_ready['llm'] = False

            result = WarmupResult(
                module='llm',
                status=WarmupStatus.FAILED,
                time_ms=(time.time() - start_time) * 1000,
                iterations=len(latencies),
                error=str(e)
            )

            self.warmup_results['llm'] = result
            structured_logger.error("❌ Profile-aware LLM warmup failed", exception=e)
            raise

    async def warmup_tts(self, tts_module) -> WarmupResult:
        """
        Profile-aware TTS warmup with variable text lengths and voices

        Tests different text lengths and voice models
        """
        if not self.warmup_config.tts or self.warmup_config.tts.iterations == 0:
            structured_logger.info("⏭️ Skipping TTS warmup (disabled in profile)")
            return WarmupResult(
                module='tts',
                status=WarmupStatus.COMPLETED,
                time_ms=0,
                iterations=0
            )

        structured_logger.info(
            f"🔊 Starting profile-aware TTS warmup ({self.warmup_config.tts.iterations} iterations)"
        )
        self.warmup_status['tts'] = WarmupStatus.IN_PROGRESS

        start_time = time.time()
        latencies = []
        per_iteration_metrics = []

        # Get test parameters from profile
        text_lengths = self.warmup_config.tts.text_lengths or [10, 30, 50]
        voices = self.warmup_config.tts.voices or ['af_sky']
        batch_sizes = self.warmup_config.tts.batch_sizes or [1]

        # Text templates for different lengths
        text_templates = {
            10: "Olá mundo",
            30: "Sistema de síntese de voz inicializado com sucesso",
            50: "Este é um teste de aquecimento do sistema de conversão de texto para fala",
            100: "O sistema de síntese de voz está sendo aquecido para garantir latências otimizadas durante a execução em produção",
            200: "O processo de aquecimento do sistema de síntese de voz é fundamental para carregar os modelos na memória GPU e compilar os kernels CUDA necessários para uma execução eficiente e com baixa latência"
        }

        try:
            with metrics.measure_time("warmup", "profile_tts_total"):
                for i in range(self.warmup_config.tts.iterations):
                    iter_start = time.time()

                    # Cycle through test parameters
                    text_len = text_lengths[i % len(text_lengths)]
                    voice = voices[i % len(voices)]
                    batch_size = batch_sizes[i % len(batch_sizes)]

                    # Get text of appropriate length
                    text = text_templates.get(text_len, "Teste de aquecimento")

                    from src.services.tts.kokoro.base import TTSRequest

                    request = TTSRequest(
                        text=text,
                        voice_id=voice,
                        speed=1.0,
                        metadata={
                            'warmup': True,
                            'profile': self.profile_id,
                            'iteration': i + 1,
                            'text_length': text_len,
                            'batch_size': batch_size
                        }
                    )

                    with metrics.measure_time("warmup", f"profile_tts_iter_{i+1}"):
                        response = await tts_module.synthesize_advanced(request)

                    iter_time = (time.time() - iter_start) * 1000
                    latencies.append(iter_time)

                    per_iteration_metrics.append({
                        'iteration': i + 1,
                        'text_length': text_len,
                        'voice': voice,
                        'batch_size': batch_size,
                        'latency_ms': iter_time
                    })

                    structured_logger.info(
                        f"  ✓ TTS warmup {i+1}/{self.warmup_config.tts.iterations}: "
                        f"{iter_time:.0f}ms (text_len: {text_len}, voice: {voice})"
                    )

                    if i < self.warmup_config.tts.iterations - 1:
                        await asyncio.sleep(0.1)

            # Calculate metrics
            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)

            self.warmup_status['tts'] = WarmupStatus.COMPLETED
            self.modules_ready['tts'] = True

            # Store profile-specific metrics
            self.profile_metrics['tts'] = ProfileWarmupMetrics(
                module='tts',
                profile_id=self.profile_id,
                status=WarmupStatus.COMPLETED,
                total_time_ms=total_time,
                iterations_completed=len(latencies),
                iterations_planned=self.warmup_config.tts.iterations,
                per_iteration_metrics=per_iteration_metrics,
                avg_latency_ms=avg_latency,
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                p50_latency_ms=self._calculate_percentile(latencies, 50),
                p95_latency_ms=self._calculate_percentile(latencies, 95)
            )

            result = WarmupResult(
                module='tts',
                status=WarmupStatus.COMPLETED,
                time_ms=total_time,
                iterations=self.warmup_config.tts.iterations,
                metrics={
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'p50_latency_ms': self._calculate_percentile(latencies, 50),
                    'p95_latency_ms': self._calculate_percentile(latencies, 95),
                    'text_lengths_tested': text_lengths,
                    'voices_tested': list(set(voices)),
                    'latencies': latencies
                }
            )

            self.warmup_results['tts'] = result

            structured_logger.info(
                f"✅ Profile-aware TTS warmup complete! "
                f"Avg: {avg_latency:.0f}ms, P95: {self._calculate_percentile(latencies, 95):.0f}ms"
            )

            return result

        except Exception as e:
            self.warmup_status['tts'] = WarmupStatus.FAILED
            self.modules_ready['tts'] = False

            result = WarmupResult(
                module='tts',
                status=WarmupStatus.FAILED,
                time_ms=(time.time() - start_time) * 1000,
                iterations=len(latencies),
                error=str(e)
            )

            self.warmup_results['tts'] = result
            structured_logger.error("❌ Profile-aware TTS warmup failed", exception=e)
            raise

    async def warmup_all(self, stt_module=None, llm_module=None, tts_module=None) -> Dict[str, WarmupResult]:
        """
        Execute profile-aware warmup for all modules

        Supports both parallel and sequential warmup based on profile configuration
        """
        structured_logger.info(
            f"🔥 STARTING PROFILE-AWARE WARMUP (profile: {self.profile_id})"
        )
        structured_logger.info(
            f"   Mode: {'PARALLEL' if self.warmup_config.parallel else 'SEQUENTIAL'}"
        )

        total_start = time.time()
        results = {}

        # Build task list
        tasks = []
        if stt_module and self.warmup_config.stt and self.warmup_config.stt.iterations > 0:
            tasks.append(('stt', self.warmup_stt(stt_module)))

        if llm_module and self.warmup_config.llm and self.warmup_config.llm.iterations > 0:
            tasks.append(('llm', self.warmup_llm(llm_module)))

        if tts_module and self.warmup_config.tts and self.warmup_config.tts.iterations > 0:
            tasks.append(('tts', self.warmup_tts(tts_module)))

        if not tasks:
            structured_logger.warning("⚠️ No modules configured for warmup in this profile")
            return {}

        # Execute warmup (parallel or sequential based on profile)
        if self.warmup_config.parallel:
            # Parallel warmup (hybrid/minimal_gpu profiles - different devices)
            structured_logger.info("🚀 Running warmup tasks in parallel...")

            with metrics.measure_time("warmup", "profile_all_parallel"):
                for module_name, task in tasks:
                    try:
                        result = await asyncio.wait_for(task, timeout=self.warmup_config.timeout_seconds)
                        results[module_name] = result
                    except asyncio.TimeoutError:
                        structured_logger.error(f"⏱️ Timeout in {module_name} warmup")
                        results[module_name] = WarmupResult(
                            module=module_name,
                            status=WarmupStatus.FAILED,
                            time_ms=self.warmup_config.timeout_seconds * 1000,
                            iterations=0,
                            error="Timeout"
                        )
                    except Exception as e:
                        structured_logger.error(f"❌ Error in {module_name} warmup: {e}")
                        results[module_name] = WarmupResult(
                            module=module_name,
                            status=WarmupStatus.FAILED,
                            time_ms=0,
                            iterations=0,
                            error=str(e)
                        )
        else:
            # Sequential warmup (full_gpu profile - shared GPU memory)
            structured_logger.info("📋 Running warmup tasks sequentially...")

            with metrics.measure_time("warmup", "profile_all_sequential"):
                for module_name, task in tasks:
                    try:
                        result = await asyncio.wait_for(task, timeout=self.warmup_config.timeout_seconds)
                        results[module_name] = result
                    except asyncio.TimeoutError:
                        structured_logger.error(f"⏱️ Timeout in {module_name} warmup")
                        results[module_name] = WarmupResult(
                            module=module_name,
                            status=WarmupStatus.FAILED,
                            time_ms=self.warmup_config.timeout_seconds * 1000,
                            iterations=0,
                            error="Timeout"
                        )
                    except Exception as e:
                        structured_logger.error(f"❌ Error in {module_name} warmup: {e}")
                        results[module_name] = WarmupResult(
                            module=module_name,
                            status=WarmupStatus.FAILED,
                            time_ms=0,
                            iterations=0,
                            error=str(e)
                        )

        total_time = (time.time() - total_start) * 1000

        # Warmup report
        structured_logger.info("\n" + "="*70)
        structured_logger.info(f"📊 PROFILE-AWARE WARMUP REPORT (Profile: {self.profile_id})")
        structured_logger.info("="*70)

        all_success = True
        for module, result in results.items():
            icon = "✅" if result.status == WarmupStatus.COMPLETED else "❌"
            structured_logger.info(
                f"{icon} {module.upper()}: {result.status.value} "
                f"({result.time_ms:.0f}ms, {result.iterations} iterations)"
            )

            if result.metrics:
                if 'avg_latency_ms' in result.metrics:
                    structured_logger.info(
                        f"    Avg latency: {result.metrics['avg_latency_ms']:.0f}ms"
                    )
                if 'p95_latency_ms' in result.metrics:
                    structured_logger.info(
                        f"    P95 latency: {result.metrics['p95_latency_ms']:.0f}ms"
                    )

            if result.status != WarmupStatus.COMPLETED:
                all_success = False

        structured_logger.info(f"\n⏱️ Total warmup time: {total_time:.0f}ms")
        structured_logger.info(f"🎯 Profile: {self.profile_id}")
        structured_logger.info(f"🔄 Mode: {'Parallel' if self.warmup_config.parallel else 'Sequential'}")

        if all_success:
            structured_logger.info("✅ SYSTEM READY FOR USE!")
        else:
            structured_logger.error("❌ WARMUP INCOMPLETE - SYSTEM NOT READY!")

        return results

    def get_profile_warmup_report(self) -> Dict[str, Any]:
        """Get detailed profile-aware warmup report"""
        base_report = self.get_warmup_report()

        return {
            **base_report,
            'profile_id': self.profile_id,
            'parallel_mode': self.warmup_config.parallel,
            'timeout_seconds': self.warmup_config.timeout_seconds,
            'profile_metrics': {
                module: {
                    'profile_id': m.profile_id,
                    'status': m.status.value,
                    'total_time_ms': m.total_time_ms,
                    'iterations_completed': m.iterations_completed,
                    'iterations_planned': m.iterations_planned,
                    'avg_latency_ms': m.avg_latency_ms,
                    'min_latency_ms': m.min_latency_ms,
                    'max_latency_ms': m.max_latency_ms,
                    'p50_latency_ms': m.p50_latency_ms,
                    'p95_latency_ms': m.p95_latency_ms,
                    'per_iteration_metrics': m.per_iteration_metrics,
                    'error': m.error
                }
                for module, m in self.profile_metrics.items()
            }
        }


# Singleton for profile-aware warmup
_profile_warmup_manager: Optional[ProfileWarmupManager] = None


def get_profile_warmup_manager(profile_id: Optional[str] = None, reload: bool = False) -> ProfileWarmupManager:
    """
    Get profile-aware warmup manager singleton

    Args:
        profile_id: GPU profile to use (defaults to active profile)
        reload: Force reload with new profile

    Returns:
        ProfileWarmupManager instance
    """
    global _profile_warmup_manager

    if _profile_warmup_manager is None or reload:
        _profile_warmup_manager = ProfileWarmupManager(profile_id=profile_id)

    return _profile_warmup_manager
