"""
Wrapper de Inicialização Unificado
Substitui todos os sistemas de inicialização espalhados
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any

from .unified_warmup import get_unified_warmup_manager, WarmupStatus
from .singleton_manager import SingletonManager

logger = logging.getLogger(__name__)


class InitializationWrapper:
    """
    Wrapper que unifica toda a inicialização do sistema

    Substitui:
    - O antigo initialization_manager
    - Os warmups espalhados nos módulos
    - As inicializações duplicadas
    """

    def __init__(self):
        self.warmup_manager = get_unified_warmup_manager()
        self.singleton_manager = SingletonManager()
        self.modules = {}
        self.is_initialized = False

        logger.info("🚀 InitializationWrapper criado")

    async def initialize_full_system(self) -> Dict[str, Any]:
        """
        Inicializa o sistema completo com warmup

        Returns:
            Relatório de inicialização
        """
        logger.info("🚀 INICIANDO SISTEMA COMPLETO")
        logger.info("="*60)

        total_start = time.time()

        try:
            # 1. Criar módulos via singleton
            logger.info("📦 Criando módulos...")
            await self._create_modules()

            # 2. Executar warmup unificado
            logger.info("🔥 Executando warmup unificado...")
            warmup_results = await self.warmup_manager.warmup_all(
                groq_module=self.modules.get('groq_stt'),
                ultravox_module=self.modules.get('ultravox_llm'),
                kokoro_module=self.modules.get('kokoro_tts')
            )

            # 3. Verificar se todos estão prontos
            if self.warmup_manager.all_ready():
                self.is_initialized = True
                logger.info("✅ SISTEMA TOTALMENTE INICIALIZADO!")
            else:
                logger.error("❌ FALHA NA INICIALIZAÇÃO!")

            total_time = (time.time() - total_start) * 1000

            return {
                "success": self.is_initialized,
                "total_time_ms": total_time,
                "modules_ready": list(self.warmup_manager._ready_modules),
                "warmup_results": warmup_results,
                "status_report": self.warmup_manager.get_status_report()
            }

        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_time_ms": (time.time() - total_start) * 1000
            }

    async def _create_modules(self):
        """Cria todos os módulos via singleton"""
        try:
            # Groq STT
            logger.info("   📡 Criando Groq STT...")
            self.modules['groq_stt'] = self.singleton_manager.get_or_create('groq_stt')

            # Ultravox LLM
            logger.info("   🤖 Criando Ultravox LLM...")
            self.modules['ultravox_llm'] = self.singleton_manager.get_or_create('ultravox')

            # Kokoro TTS
            logger.info("   🔊 Criando Kokoro TTS...")
            self.modules['kokoro_tts'] = self.singleton_manager.get_or_create('kokoro_tts')

            logger.info("✅ Todos os módulos criados com sucesso!")

        except Exception as e:
            logger.error(f"❌ Erro ao criar módulos: {e}")
            raise

    def get_module(self, module_name: str) -> Optional[Any]:
        """
        Obtém módulo já inicializado

        Args:
            module_name: Nome do módulo (groq_stt, ultravox_llm, kokoro_tts)

        Returns:
            Instância do módulo ou None
        """
        # Enforçar que o módulo fez warmup
        self.warmup_manager.enforce_ready(module_name)
        return self.modules.get(module_name)

    def is_module_ready(self, module_name: str) -> bool:
        """Verifica se módulo está pronto"""
        return self.warmup_manager.is_ready(module_name)

    def is_system_ready(self) -> bool:
        """Verifica se sistema completo está pronto"""
        return self.is_initialized and self.warmup_manager.all_ready()

    def get_status_report(self) -> Dict[str, Any]:
        """Obtém relatório completo do sistema"""
        return {
            "system_initialized": self.is_initialized,
            "warmup_report": self.warmup_manager.get_status_report(),
            "available_modules": list(self.modules.keys())
        }


# Singleton global para o wrapper
_initialization_wrapper: Optional[InitializationWrapper] = None


def get_initialization_wrapper() -> InitializationWrapper:
    """Obtém instância singleton do InitializationWrapper"""
    global _initialization_wrapper
    if _initialization_wrapper is None:
        _initialization_wrapper = InitializationWrapper()
    return _initialization_wrapper


async def initialize_system_unified() -> Dict[str, Any]:
    """
    Função de conveniência para inicialização completa

    Use esta função para substituir todas as inicializações existentes!
    """
    wrapper = get_initialization_wrapper()
    return await wrapper.initialize_full_system()


def get_ready_module(module_name: str) -> Any:
    """
    Obtém módulo já pronto (com warmup)

    Args:
        module_name: groq_stt, ultravox_llm, ou kokoro_tts

    Returns:
        Instância do módulo

    Raises:
        RuntimeError: Se módulo não fez warmup
    """
    wrapper = get_initialization_wrapper()
    return wrapper.get_module(module_name)