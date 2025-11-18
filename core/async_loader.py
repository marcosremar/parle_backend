#!/usr/bin/env python3
"""
Carregador Assíncrono Simples
Carrega modelos pesados em threads sem travar a inicialização
"""

import logging
import threading
import time
from typing import Any, Callable, Optional, Dict
import asyncio

logger = logging.getLogger(__name__)

class AsyncModelLoader:
    """
    Carregador assíncrono simples para modelos pesados
    Abordagem mais direta que lazy loading complexo
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.loading_status: Dict[str, Dict] = {}
        self.loading_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start_loading(self, model_name: str, factory_func: Callable, *args, **kwargs) -> None:
        """
        Inicia carregamento assíncrono de um modelo

        Args:
            model_name: Nome do modelo
            factory_func: Função que cria o modelo
            *args, **kwargs: Argumentos para factory_func
        """
        with self._lock:
            if model_name in self.loading_status:
                logger.info(f"🔄 {model_name} já está sendo carregado")
                return

            # Inicializa status
            self.loading_status[model_name] = {
                "is_loading": True,
                "is_ready": False,
                "error": None,
                "start_time": time.time()
            }

            # Cria e inicia thread
            thread = threading.Thread(
                target=self._load_model_async,
                args=(model_name, factory_func, args, kwargs),
                daemon=True,
                name=f"AsyncLoader-{model_name}"
            )

            self.loading_threads[model_name] = thread
            thread.start()

            logger.info(f"🚀 Iniciando carregamento assíncrono de {model_name}...")

    def _load_model_async(self, model_name: str, factory_func: Callable, args: tuple, kwargs: dict) -> None:
        """Carrega modelo em thread separada"""
        try:
            logger.info(f"📥 Carregando {model_name}...")
            start_time = time.time()

            # Chama factory function
            model = factory_func(*args, **kwargs)

            # Inicializa se necessário
            if hasattr(model, 'initialize') and callable(model.initialize):
                if asyncio.iscoroutinefunction(model.initialize):
                    # Async initialize
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(model.initialize())
                    finally:
                        loop.close()
                else:
                    # Sync initialize
                    model.initialize()

            # Warmup se disponível
            if hasattr(model, 'warmup') and callable(model.warmup):
                logger.info(f"🔥 Fazendo warmup de {model_name}...")
                if asyncio.iscoroutinefunction(model.warmup):
                    # Async warmup
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(model.warmup())
                    finally:
                        loop.close()
                else:
                    # Sync warmup
                    model.warmup()

            # Modelo carregado com sucesso
            with self._lock:
                self.models[model_name] = model
                self.loading_status[model_name].update({
                    "is_loading": False,
                    "is_ready": True,
                    "error": None,
                    "elapsed_time": time.time() - start_time
                })

            elapsed = time.time() - start_time
            logger.info(f"✅ {model_name} carregado com sucesso em {elapsed:.1f}s!")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Erro ao carregar {model_name}: {error_msg}")

            with self._lock:
                self.loading_status[model_name].update({
                    "is_loading": False,
                    "is_ready": False,
                    "error": error_msg
                })

    def get_model(self, model_name: str, wait_timeout: float = 30.0) -> Optional[Any]:
        """
        Obtém modelo (espera carregar se necessário)

        Args:
            model_name: Nome do modelo
            wait_timeout: Tempo limite para esperar (segundos)

        Returns:
            Modelo carregado ou None se falhou
        """
        # Se já está carregado, retorna imediatamente
        if model_name in self.models:
            return self.models[model_name]

        # Se não está sendo carregado, retorna None
        if model_name not in self.loading_status:
            logger.warning(f"⚠️ Modelo {model_name} não foi iniciado para carregamento")
            return None

        # Espera carregamento terminar
        start_wait = time.time()
        while time.time() - start_wait < wait_timeout:
            status = self.loading_status[model_name]

            if status["is_ready"]:
                return self.models[model_name]

            if status["error"]:
                raise RuntimeError(f"Modelo {model_name} falhou ao carregar: {status['error']}")

            time.sleep(0.5)

        # Timeout
        raise TimeoutError(f"Timeout esperando {model_name} carregar ({wait_timeout}s)")

    def is_ready(self, model_name: str) -> bool:
        """Verifica se modelo está pronto"""
        return model_name in self.models

    def is_loading(self, model_name: str) -> bool:
        """Verifica se modelo está carregando"""
        status = self.loading_status.get(model_name, {})
        return status.get("is_loading", False)

    def get_status(self, model_name: str) -> Dict:
        """Obtém status detalhado do modelo"""
        if model_name not in self.loading_status:
            return {"is_loading": False, "is_ready": False, "error": "Not started"}

        status = self.loading_status[model_name].copy()
        status["is_ready"] = model_name in self.models
        return status

    def get_all_status(self) -> Dict[str, Dict]:
        """Obtém status de todos os modelos"""
        return {name: self.get_status(name) for name in self.loading_status}

    def cleanup(self, model_name: str) -> None:
        """Remove modelo da memória"""
        with self._lock:
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                del self.models[model_name]
                logger.info(f"🗑️ Modelo {model_name} removido")

            if model_name in self.loading_status:
                del self.loading_status[model_name]

            if model_name in self.loading_threads:
                del self.loading_threads[model_name]


# Instância global
async_loader = AsyncModelLoader()


def start_ultravox_loading():
    """Helper para iniciar carregamento do Ultravox"""
    def create_ultravox():
        from src.services.llm.ultravox.ultravox_vllm import UltravoxVLLM
        config = {
            "system_prompt": """Você é um assistente acadêmico especializado em responder perguntas em português.
Responda de forma clara, objetiva e educativa. Mantenha suas respostas concisas mas informativas,
adequadas para um contexto acadêmico. Use linguagem formal mas acessível.""",
            "temperature": 0.8,
            "max_tokens": 80
        }
        return UltravoxVLLM(config)

    async_loader.start_loading("ultravox", create_ultravox)


def start_kokoro_loading():
    """Helper para iniciar carregamento do Kokoro TTS"""
    def create_kokoro():
        from src.services.tts.kokoro.wrapper import KokoroTTS
        tts = KokoroTTS(device='cuda')
        return tts

    async_loader.start_loading("kokoro_tts", create_kokoro)


def get_ultravox(wait_timeout: float = 30.0):
    """Helper para obter Ultravox (com espera se necessário)"""
    return async_loader.get_model("ultravox", wait_timeout)


def get_kokoro_tts(wait_timeout: float = 30.0):
    """Helper para obter Kokoro TTS (com espera se necessário)"""
    return async_loader.get_model("kokoro_tts", wait_timeout)