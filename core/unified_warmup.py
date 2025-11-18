"""
Sistema Unificado de Warmup - Módulo Central
Gerencia o aquecimento de todos os módulos do sistema
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WarmupStatus(Enum):
    """Status do warmup"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WarmupResult:
    """Resultado do warmup"""
    module: str
    status: WarmupStatus
    duration_ms: float
    iterations: int
    avg_latency_ms: float = 0.0
    error: Optional[str] = None


class UnifiedWarmupManager:
    """
    Gerenciador Unificado de Warmup

    Centraliza todo o aquecimento dos módulos:
    - Groq STT
    - Ultravox LLM
    - Kokoro TTS
    """

    def __init__(self):
        self.status: Dict[str, WarmupStatus] = {}
        self.results: Dict[str, WarmupResult] = {}
        self._ready_modules: set = set()

        logger.info("🔥 UnifiedWarmupManager inicializado")

    def is_ready(self, module: str) -> bool:
        """Verifica se módulo está pronto"""
        return module in self._ready_modules

    def all_ready(self) -> bool:
        """Verifica se todos os módulos estão prontos"""
        expected_modules = {"groq_stt", "ultravox_llm", "kokoro_tts"}
        return expected_modules.issubset(self._ready_modules)

    async def warmup_groq_stt(self, groq_module, iterations: int = 2) -> WarmupResult:
        """Warmup do Groq STT"""
        module_name = "groq_stt"
        logger.info(f"🎤 Iniciando warmup {module_name}...")

        self.status[module_name] = WarmupStatus.IN_PROGRESS
        start_time = time.time()
        latencies = []

        try:
            for i in range(iterations):
                iter_start = time.time()

                # Áudio sintético para warmup
                sample_rate = 16000
                duration = 1.0
                audio = np.random.randn(int(sample_rate * duration)) * 0.01
                audio = audio.astype(np.float32)

                # Transcever usando método real do Groq
                if hasattr(groq_module, 'transcribe_audio_async'):
                    await groq_module.transcribe_audio_async(audio, sample_rate)
                elif hasattr(groq_module, 'transcribe'):
                    await asyncio.to_thread(groq_module.transcribe, audio)

                iter_time = (time.time() - iter_start) * 1000
                latencies.append(iter_time)

                logger.info(f"   ✓ Warmup {i+1}/{iterations}: {iter_time:.0f}ms")

                if i < iterations - 1:
                    await asyncio.sleep(0.1)

            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)

            self.status[module_name] = WarmupStatus.COMPLETED
            self._ready_modules.add(module_name)

            result = WarmupResult(
                module=module_name,
                status=WarmupStatus.COMPLETED,
                duration_ms=total_time,
                iterations=iterations,
                avg_latency_ms=avg_latency
            )

            self.results[module_name] = result
            logger.info(f"✅ {module_name} pronto! Média: {avg_latency:.0f}ms")
            return result

        except Exception as e:
            self.status[module_name] = WarmupStatus.FAILED
            result = WarmupResult(
                module=module_name,
                status=WarmupStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                iterations=len(latencies),
                error=str(e)
            )
            self.results[module_name] = result
            logger.error(f"❌ Falha no warmup {module_name}: {e}")
            raise

    async def warmup_ultravox_llm(self, ultravox_module, iterations: int = 3) -> WarmupResult:
        """Warmup do Ultravox LLM"""
        module_name = "ultravox_llm"
        logger.info(f"🤖 Iniciando warmup {module_name}...")

        self.status[module_name] = WarmupStatus.IN_PROGRESS
        start_time = time.time()
        latencies = []

        try:
            for i in range(iterations):
                iter_start = time.time()

                # Áudio sintético para warmup
                sample_rate = 16000
                duration = 0.5
                audio = np.random.randn(int(sample_rate * duration)) * 0.01
                audio = audio.astype(np.float32)

                # Processar com Ultravox usando método correto
                if hasattr(ultravox_module, 'process_audio'):
                    await ultravox_module.process_audio(
                        audio=audio,
                        sample_rate=sample_rate,
                        context="Warmup test",
                        session_id=f"warmup_{i}"
                    )
                elif hasattr(ultravox_module, 'generate_response'):
                    await asyncio.to_thread(
                        ultravox_module.generate_response,
                        prompt="Warmup test",
                        max_tokens=5
                    )

                iter_time = (time.time() - iter_start) * 1000
                latencies.append(iter_time)

                logger.info(f"   ✓ Warmup {i+1}/{iterations}: {iter_time:.0f}ms")

                # Pausa maior na primeira iteração (compilação CUDA)
                if i == 0:
                    await asyncio.sleep(1.0)
                elif i < iterations - 1:
                    await asyncio.sleep(0.2)

            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)

            self.status[module_name] = WarmupStatus.COMPLETED
            self._ready_modules.add(module_name)

            result = WarmupResult(
                module=module_name,
                status=WarmupStatus.COMPLETED,
                duration_ms=total_time,
                iterations=iterations,
                avg_latency_ms=avg_latency
            )

            self.results[module_name] = result
            logger.info(f"✅ {module_name} pronto! Média: {avg_latency:.0f}ms (1ª: {latencies[0]:.0f}ms)")
            return result

        except Exception as e:
            self.status[module_name] = WarmupStatus.FAILED
            result = WarmupResult(
                module=module_name,
                status=WarmupStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                iterations=len(latencies),
                error=str(e)
            )
            self.results[module_name] = result
            logger.error(f"❌ Falha no warmup {module_name}: {e}")
            raise

    async def warmup_kokoro_tts(self, kokoro_module, iterations: int = 2) -> WarmupResult:
        """Warmup do Kokoro TTS"""
        module_name = "kokoro_tts"
        logger.info(f"🔊 Iniciando warmup {module_name}...")

        self.status[module_name] = WarmupStatus.IN_PROGRESS
        start_time = time.time()
        latencies = []

        try:
            warmup_texts = ["Olá", "Teste de áudio", "Sistema inicializado"]

            for i in range(iterations):
                iter_start = time.time()

                text = warmup_texts[min(i, len(warmup_texts)-1)]

                # Sintetizar com Kokoro usando método correto
                if hasattr(kokoro_module, 'generate_speech'):
                    await asyncio.to_thread(
                        kokoro_module.generate_speech,
                        text=text,
                        voice="pf_dora"
                    )
                elif hasattr(kokoro_module, 'synthesize'):
                    await asyncio.to_thread(
                        kokoro_module.synthesize,
                        text
                    )

                iter_time = (time.time() - iter_start) * 1000
                latencies.append(iter_time)

                logger.info(f"   ✓ Warmup {i+1}/{iterations}: {iter_time:.0f}ms")

                if i < iterations - 1:
                    await asyncio.sleep(0.1)

            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)

            self.status[module_name] = WarmupStatus.COMPLETED
            self._ready_modules.add(module_name)

            result = WarmupResult(
                module=module_name,
                status=WarmupStatus.COMPLETED,
                duration_ms=total_time,
                iterations=iterations,
                avg_latency_ms=avg_latency
            )

            self.results[module_name] = result
            logger.info(f"✅ {module_name} pronto! Média: {avg_latency:.0f}ms")
            return result

        except Exception as e:
            self.status[module_name] = WarmupStatus.FAILED
            result = WarmupResult(
                module=module_name,
                status=WarmupStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                iterations=len(latencies),
                error=str(e)
            )
            self.results[module_name] = result
            logger.error(f"❌ Falha no warmup {module_name}: {e}")
            raise

    async def warmup_all(self,
                        groq_module=None,
                        ultravox_module=None,
                        kokoro_module=None,
                        timeout_seconds: int = 120) -> Dict[str, WarmupResult]:
        """
        Executa warmup de todos os módulos

        Args:
            groq_module: Instância do Groq STT
            ultravox_module: Instância do Ultravox LLM
            kokoro_module: Instância do Kokoro TTS
            timeout_seconds: Timeout para cada módulo

        Returns:
            Dict com resultados do warmup
        """
        logger.info("🔥 INICIANDO WARMUP UNIFICADO DO SISTEMA")
        logger.info("="*60)

        total_start = time.time()
        tasks = []

        # Criar tarefas de warmup
        if groq_module:
            tasks.append(("groq_stt", self.warmup_groq_stt(groq_module)))

        if ultravox_module:
            tasks.append(("ultravox_llm", self.warmup_ultravox_llm(ultravox_module)))

        if kokoro_module:
            tasks.append(("kokoro_tts", self.warmup_kokoro_tts(kokoro_module)))

        if not tasks:
            logger.warning("⚠️ Nenhum módulo fornecido para warmup!")
            return {}

        # Executar warmups sequencialmente para evitar concorrência na GPU
        results = {}
        for module_name, task in tasks:
            try:
                logger.info(f"🚀 Executando warmup: {module_name}")
                result = await asyncio.wait_for(task, timeout=timeout_seconds)
                results[module_name] = result
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout no warmup do {module_name}")
                results[module_name] = WarmupResult(
                    module=module_name,
                    status=WarmupStatus.FAILED,
                    duration_ms=timeout_seconds * 1000,
                    iterations=0,
                    error="Timeout"
                )
            except Exception as e:
                logger.error(f"❌ Erro no warmup do {module_name}: {e}")
                results[module_name] = WarmupResult(
                    module=module_name,
                    status=WarmupStatus.FAILED,
                    duration_ms=0,
                    iterations=0,
                    error=str(e)
                )

        total_time = (time.time() - total_start) * 1000

        # Relatório final
        logger.info("\n" + "="*60)
        logger.info("📊 RELATÓRIO DE WARMUP UNIFICADO")
        logger.info("="*60)

        all_success = True
        for module, result in results.items():
            icon = "✅" if result.status == WarmupStatus.COMPLETED else "❌"
            logger.info(
                f"{icon} {module.upper()}: {result.status.value} "
                f"({result.duration_ms:.0f}ms, {result.iterations} iterações)"
            )

            if result.status == WarmupStatus.COMPLETED:
                logger.info(f"    Latência média: {result.avg_latency_ms:.0f}ms")
            else:
                all_success = False
                if result.error:
                    logger.info(f"    Erro: {result.error}")

        logger.info(f"\n⏱️ Tempo total de warmup: {total_time:.0f}ms")

        if all_success:
            logger.info("✅ SISTEMA TOTALMENTE PRONTO PARA USO!")
        else:
            logger.error("❌ WARMUP INCOMPLETO - ALGUNS MÓDULOS FALHARAM!")

        logger.info("="*60)

        return results

    def get_status_report(self) -> Dict[str, Any]:
        """Obtém relatório de status detalhado"""
        return {
            "ready_modules": list(self._ready_modules),
            "all_ready": self.all_ready(),
            "status": {k: v.value for k, v in self.status.items()},
            "results": {
                k: {
                    "status": v.status.value,
                    "duration_ms": v.duration_ms,
                    "iterations": v.iterations,
                    "avg_latency_ms": v.avg_latency_ms,
                    "error": v.error
                }
                for k, v in self.results.items()
            }
        }

    def enforce_ready(self, module: str) -> None:
        """Garante que módulo fez warmup"""
        if not self.is_ready(module):
            status = self.status.get(module, WarmupStatus.NOT_STARTED)
            raise RuntimeError(
                f"❌ MÓDULO {module.upper()} NÃO FEZ WARMUP! "
                f"Status: {status.value}. Execute warmup primeiro!"
            )


# Singleton global
_warmup_manager: Optional[UnifiedWarmupManager] = None


def get_unified_warmup_manager() -> UnifiedWarmupManager:
    """Obtém instância singleton do UnifiedWarmupManager"""
    global _warmup_manager
    if _warmup_manager is None:
        _warmup_manager = UnifiedWarmupManager()
    return _warmup_manager