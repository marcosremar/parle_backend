"""
Gerenciador de Warm-up - Sistema obrigatório de aquecimento dos módulos
Garante que STT, LLM e TTS estejam prontos antes do primeiro uso
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Import metrics and logging
from .metrics import get_metrics_collector
from .structured_logger import get_logger

logger = logging.getLogger(__name__)
structured_logger = get_logger("WarmupManager")
metrics = get_metrics_collector()


class WarmupStatus(Enum):
    """Status do warm-up de cada módulo"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WarmupResult:
    """Resultado do warm-up de um módulo"""
    module: str
    status: WarmupStatus
    time_ms: float
    iterations: int
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class WarmupManager:
    """
    Gerenciador de Warm-up OBRIGATÓRIO
    
    Garante que todos os módulos façam warm-up antes do primeiro uso real.
    Sem warm-up = Sem funcionamento!
    """
    
    def __init__(self,
                 stt_iterations: int = 3,
                 llm_iterations: int = 2, 
                 tts_iterations: int = 3,
                 timeout_seconds: int = 60):
        """
        Initialize warmup manager
        
        Args:
            stt_iterations: Número de iterações de warm-up para STT
            llm_iterations: Número de iterações de warm-up para LLM
            tts_iterations: Número de iterações de warm-up para TTS
            timeout_seconds: Timeout máximo para cada módulo
        """
        self.stt_iterations = stt_iterations
        self.llm_iterations = llm_iterations
        self.tts_iterations = tts_iterations
        self.timeout_seconds = timeout_seconds
        
        # Status de warm-up
        self.warmup_status: Dict[str, WarmupStatus] = {
            'stt': WarmupStatus.NOT_STARTED,
            'llm': WarmupStatus.NOT_STARTED,
            'tts': WarmupStatus.NOT_STARTED
        }
        
        # Resultados de warm-up
        self.warmup_results: Dict[str, WarmupResult] = {}
        
        # Flags de módulos prontos
        self.modules_ready = {
            'stt': False,
            'llm': False,
            'tts': False
        }
        
        structured_logger.info("🔥 WarmupManager inicializado",
                             metadata={
                                 'stt_iterations': stt_iterations,
                                 'llm_iterations': llm_iterations,
                                 'tts_iterations': tts_iterations
                             })
    
    def is_module_ready(self, module: str) -> bool:
        """
        Verifica se módulo está pronto (warm-up completo)
        
        OBRIGATÓRIO: Deve ser chamado antes de usar qualquer módulo!
        """
        return self.modules_ready.get(module, False)
    
    def check_all_ready(self) -> bool:
        """Verifica se TODOS os módulos estão prontos"""
        return all(self.modules_ready.values())
    
    async def warmup_stt(self, stt_module) -> WarmupResult:
        """
        Warm-up OBRIGATÓRIO do módulo STT
        
        Args:
            stt_module: Instância do módulo STT
            
        Returns:
            WarmupResult com status e métricas
        """
        structured_logger.info("🎤 Iniciando warm-up do STT...")
        self.warmup_status['stt'] = WarmupStatus.IN_PROGRESS
        
        start_time = time.time()
        latencies = []
        
        try:
            with metrics.measure_time("warmup", "stt_total"):
                for i in range(self.stt_iterations):
                    iter_start = time.time()
                    
                    # Criar áudio sintético para warm-up
                    sample_rate = 16000
                    duration = 1.0  # 1 segundo
                    audio = np.random.randn(int(sample_rate * duration)) * 0.01
                    audio = audio.astype(np.float32)
                    
                    # Importar classes necessárias
                    from src.services.external_stt.transcription.base import STTRequest
                    
                    # Criar request de warm-up
                    request = STTRequest(
                        audio=audio,
                        sample_rate=sample_rate,
                        language="pt",
                        metadata={'warmup': True, 'iteration': i+1}
                    )
                    
                    # Executar transcrição
                    with metrics.measure_time("warmup", f"stt_iteration_{i+1}"):
                        response = await stt_module.transcribe_batch(request)
                    
                    iter_time = (time.time() - iter_start) * 1000
                    latencies.append(iter_time)
                    
                    structured_logger.info(f"  ✓ STT warm-up {i+1}/{self.stt_iterations}: {iter_time:.0f}ms")
                    
                    # Pequena pausa entre iterações
                    if i < self.stt_iterations - 1:
                        await asyncio.sleep(0.1)
            
            # Calcular métricas
            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)
            
            # Marcar como pronto
            self.warmup_status['stt'] = WarmupStatus.COMPLETED
            self.modules_ready['stt'] = True
            
            result = WarmupResult(
                module='stt',
                status=WarmupStatus.COMPLETED,
                time_ms=total_time,
                iterations=self.stt_iterations,
                metrics={
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'latencies': latencies
                }
            )
            
            self.warmup_results['stt'] = result
            
            structured_logger.pipeline_stage("stt_warmup", total_time, success=True,
                                            avg_latency=avg_latency,
                                            iterations=self.stt_iterations)
            
            structured_logger.info(f"✅ STT warm-up completo! Média: {avg_latency:.0f}ms")
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
            structured_logger.error("❌ Falha no warm-up do STT", exception=e)
            raise
    
    async def warmup_llm(self, llm_module) -> WarmupResult:
        """
        Warm-up OBRIGATÓRIO do módulo LLM
        
        Args:
            llm_module: Instância do módulo LLM
            
        Returns:
            WarmupResult com status e métricas
        """
        structured_logger.info("🤖 Iniciando warm-up do LLM...")
        self.warmup_status['llm'] = WarmupStatus.IN_PROGRESS
        
        start_time = time.time()
        latencies = []
        
        try:
            with metrics.measure_time("warmup", "llm_total"):
                for i in range(self.llm_iterations):
                    iter_start = time.time()
                    
                    # Criar áudio sintético para warm-up
                    sample_rate = 16000
                    duration = 0.5  # 0.5 segundos
                    audio = np.random.randn(int(sample_rate * duration)) * 0.01
                    audio = audio.astype(np.float32)
                    
                    # Importar classes necessárias
                    from src.services.llm.evaluators.base import LLMRequest
                    
                    # Prompts de warm-up progressivos
                    warmup_prompts = [
                        "<|audio|>\nOlá",  # Muito curto
                        "<|audio|>\nResponda: teste",  # Curto
                        "<|audio|>\nQual é a capital do Brasil? Responda: Brasília"  # Normal
                    ]
                    
                    prompt = warmup_prompts[min(i, len(warmup_prompts)-1)]
                    
                    # Criar request de warm-up
                    request = LLMRequest(
                        prompt=prompt,
                        max_tokens=10,  # Poucos tokens para warm-up
                        temperature=0.1,
                        metadata={
                            'warmup': True,
                            'iteration': i+1,
                            'audio_data': {
                                'audio_tuple': (audio, sample_rate),
                                'audio_shape': audio.shape,
                                'sample_rate': sample_rate
                            }
                        }
                    )
                    
                    # Executar geração
                    with metrics.measure_time("warmup", f"llm_iteration_{i+1}"):
                        response = await llm_module.generate(request)
                    
                    iter_time = (time.time() - iter_start) * 1000
                    latencies.append(iter_time)
                    
                    structured_logger.info(f"  ✓ LLM warm-up {i+1}/{self.llm_iterations}: {iter_time:.0f}ms")
                    
                    # Pausa maior na primeira iteração (compilação CUDA)
                    if i == 0:
                        await asyncio.sleep(1.0)
                    elif i < self.llm_iterations - 1:
                        await asyncio.sleep(0.2)
            
            # Calcular métricas
            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)
            
            # Marcar como pronto
            self.warmup_status['llm'] = WarmupStatus.COMPLETED
            self.modules_ready['llm'] = True
            
            result = WarmupResult(
                module='llm',
                status=WarmupStatus.COMPLETED,
                time_ms=total_time,
                iterations=self.llm_iterations,
                metrics={
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'first_iter_ms': latencies[0],  # Importante: primeira é mais lenta
                    'latencies': latencies
                }
            )
            
            self.warmup_results['llm'] = result
            
            structured_logger.pipeline_stage("llm_warmup", total_time, success=True,
                                            avg_latency=avg_latency,
                                            first_iter=latencies[0],
                                            iterations=self.llm_iterations)
            
            structured_logger.info(f"✅ LLM warm-up completo! Média: {avg_latency:.0f}ms (1ª: {latencies[0]:.0f}ms)")
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
            structured_logger.error("❌ Falha no warm-up do LLM", exception=e)
            raise
    
    async def warmup_tts(self, tts_module) -> WarmupResult:
        """
        Warm-up OBRIGATÓRIO do módulo TTS
        
        Args:
            tts_module: Instância do módulo TTS
            
        Returns:
            WarmupResult com status e métricas
        """
        structured_logger.info("🔊 Iniciando warm-up do TTS...")
        self.warmup_status['tts'] = WarmupStatus.IN_PROGRESS
        
        start_time = time.time()
        latencies = []
        
        try:
            with metrics.measure_time("warmup", "tts_total"):
                for i in range(self.tts_iterations):
                    iter_start = time.time()
                    
                    # Textos de warm-up progressivos
                    warmup_texts = [
                        "Oi",  # Muito curto
                        "Teste de áudio",  # Curto
                        "Sistema de síntese de voz inicializado"  # Normal
                    ]
                    
                    text = warmup_texts[min(i, len(warmup_texts)-1)]
                    
                    # Importar classes necessárias
                    from src.services.tts.kokoro.base import TTSRequest
                    
                    # Criar request de warm-up
                    request = TTSRequest(
                        text=text,
                        voice_id="pt_BR_female_1",
                        speed=1.0,
                        metadata={'warmup': True, 'iteration': i+1}
                    )
                    
                    # Executar síntese
                    with metrics.measure_time("warmup", f"tts_iteration_{i+1}"):
                        response = await tts_module.synthesize_advanced(request)
                    
                    iter_time = (time.time() - iter_start) * 1000
                    latencies.append(iter_time)
                    
                    structured_logger.info(f"  ✓ TTS warm-up {i+1}/{self.tts_iterations}: {iter_time:.0f}ms")
                    
                    # Pequena pausa entre iterações
                    if i < self.tts_iterations - 1:
                        await asyncio.sleep(0.1)
            
            # Calcular métricas
            total_time = (time.time() - start_time) * 1000
            avg_latency = sum(latencies) / len(latencies)
            
            # Marcar como pronto
            self.warmup_status['tts'] = WarmupStatus.COMPLETED
            self.modules_ready['tts'] = True
            
            result = WarmupResult(
                module='tts',
                status=WarmupStatus.COMPLETED,
                time_ms=total_time,
                iterations=self.tts_iterations,
                metrics={
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'latencies': latencies
                }
            )
            
            self.warmup_results['tts'] = result
            
            structured_logger.pipeline_stage("tts_warmup", total_time, success=True,
                                            avg_latency=avg_latency,
                                            iterations=self.tts_iterations)
            
            structured_logger.info(f"✅ TTS warm-up completo! Média: {avg_latency:.0f}ms")
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
            structured_logger.error("❌ Falha no warm-up do TTS", exception=e)
            raise
    
    async def warmup_all(self, stt_module=None, llm_module=None, tts_module=None) -> Dict[str, WarmupResult]:
        """
        Executa warm-up de TODOS os módulos disponíveis
        
        OBRIGATÓRIO: Deve ser chamado na inicialização do sistema!
        
        Args:
            stt_module: Módulo STT (opcional)
            llm_module: Módulo LLM (opcional)
            tts_module: Módulo TTS (opcional)
            
        Returns:
            Dict com resultados de cada módulo
        """
        structured_logger.info("🔥 INICIANDO WARM-UP OBRIGATÓRIO DO SISTEMA")
        structured_logger.info("⚠️  SEM WARM-UP = SEM FUNCIONAMENTO!")
        
        total_start = time.time()
        tasks = []
        
        # Criar tarefas de warm-up paralelas
        if stt_module:
            tasks.append(('stt', self.warmup_stt(stt_module)))
        
        if llm_module:
            tasks.append(('llm', self.warmup_llm(llm_module)))
        
        if tts_module:
            tasks.append(('tts', self.warmup_tts(tts_module)))
        
        if not tasks:
            structured_logger.warning("⚠️ Nenhum módulo fornecido para warm-up!")
            return {}
        
        # Executar warm-ups em paralelo
        results = {}
        with metrics.measure_time("warmup", "all_modules"):
            for module_name, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=self.timeout_seconds)
                    results[module_name] = result
                except asyncio.TimeoutError:
                    structured_logger.error(f"⏱️ Timeout no warm-up do {module_name}")
                    results[module_name] = WarmupResult(
                        module=module_name,
                        status=WarmupStatus.FAILED,
                        time_ms=self.timeout_seconds * 1000,
                        iterations=0,
                        error="Timeout"
                    )
                except Exception as e:
                    structured_logger.error(f"❌ Erro no warm-up do {module_name}: {e}")
                    results[module_name] = WarmupResult(
                        module=module_name,
                        status=WarmupStatus.FAILED,
                        time_ms=0,
                        iterations=0,
                        error=str(e)
                    )
        
        total_time = (time.time() - total_start) * 1000
        
        # Relatório de warm-up
        structured_logger.info("\n" + "="*60)
        structured_logger.info("📊 RELATÓRIO DE WARM-UP")
        structured_logger.info("="*60)
        
        all_success = True
        for module, result in results.items():
            icon = "✅" if result.status == WarmupStatus.COMPLETED else "❌"
            structured_logger.info(
                f"{icon} {module.upper()}: {result.status.value} "
                f"({result.time_ms:.0f}ms, {result.iterations} iterações)"
            )
            
            if result.metrics and 'avg_latency_ms' in result.metrics:
                structured_logger.info(
                    f"    Latência média: {result.metrics['avg_latency_ms']:.0f}ms"
                )
            
            if result.status != WarmupStatus.COMPLETED:
                all_success = False
        
        structured_logger.info(f"\n⏱️ Tempo total de warm-up: {total_time:.0f}ms")
        
        if all_success:
            structured_logger.info("✅ SISTEMA PRONTO PARA USO!")
        else:
            structured_logger.error("❌ WARM-UP INCOMPLETO - SISTEMA NÃO ESTÁ PRONTO!")
        
        return results
    
    def get_warmup_report(self) -> Dict[str, Any]:
        """Obtém relatório detalhado do warm-up"""
        return {
            'status': {
                module: status.value 
                for module, status in self.warmup_status.items()
            },
            'ready': self.modules_ready,
            'all_ready': self.check_all_ready(),
            'results': {
                module: {
                    'status': result.status.value,
                    'time_ms': result.time_ms,
                    'iterations': result.iterations,
                    'metrics': result.metrics,
                    'error': result.error
                }
                for module, result in self.warmup_results.items()
            }
        }
    
    def enforce_warmup(self, module: str) -> None:
        """
        ENFORCE: Garante que módulo fez warm-up
        
        Raises:
            RuntimeError: Se módulo não fez warm-up
        """
        if not self.is_module_ready(module):
            raise RuntimeError(
                f"❌ MÓDULO {module.upper()} NÃO FEZ WARM-UP! "
                f"Status: {self.warmup_status.get(module, 'unknown')}. "
                f"Execute warmup_{module}() antes de usar!"
            )


# Singleton global
_warmup_manager: Optional[WarmupManager] = None


def get_warmup_manager() -> WarmupManager:
    """Obtém instância singleton do WarmupManager"""
    global _warmup_manager
    if _warmup_manager is None:
        _warmup_manager = WarmupManager()
    return _warmup_manager