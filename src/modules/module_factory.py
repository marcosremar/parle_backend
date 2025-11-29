"""
Module Factory - Cria instâncias de módulos para chamadas diretas
"""

import os
from typing import Dict, Any, Optional
from loguru import logger

# Cache de instâncias (singletons)
_module_instances: Dict[str, Any] = {}


def create(module_name: str) -> Any:
    """
    Cria ou retorna instância de um módulo
    
    Args:
        module_name: Nome do módulo (ex: 'stt', 'llm', 'tts', 'user', etc.)
    
    Returns:
        Instância do módulo
    """
    global _module_instances
    
    # Retornar se já existe
    if module_name in _module_instances:
        return _module_instances[module_name]
    
    # Criar nova instância
    try:
        module = _create_module(module_name)
        _module_instances[module_name] = module
        logger.debug(f"✅ Module '{module_name}' created")
        return module
    except Exception as e:
        logger.error(f"❌ Failed to create module '{module_name}': {e}")
        raise


def _create_module(module_name: str) -> Any:
    """Cria instância específica de um módulo"""
    
    # Tutoring modules (desativados por padrão - future flag)
    TUTORING_MODULES = {
        "student_model",
        "diagnostic_module",
        "pedagogical_policy",
        "learning_path"
    }
    
    # Verificar se tutoring está desativado
    enable_tutoring = os.getenv("ENABLE_TUTORING_MODULES", "false").lower() == "true"
    if module_name in TUTORING_MODULES and not enable_tutoring:
        logger.warning(f"⚠️  Tutoring module '{module_name}' is disabled. Set ENABLE_TUTORING_MODULES=true to enable.")
        return _create_disabled_tutoring_wrapper(module_name)
    
    # Mapeamento de nomes de módulos para classes (nova estrutura com subdiretórios)
    module_map = {
        # Speech modules
        "stt": "src.modules.speech.stt.module.STTModule",
        "tts": "src.modules.speech.tts.module.TTSModule",
        "neural_codec": "src.modules.speech.neural_codec.module.NeuralCodecModule",
        
        # LLM module
        "llm": "src.modules.llm.llm_module.LLMModule",
        
        # Conversation modules
        "orchestrator": "src.modules.conversation.orchestrator.module.OrchestratorModule",
        "session": "src.modules.conversation.session.module.SessionModule",
        "scenarios": "src.modules.conversation.scenarios.module.ScenariosModule",
        
        # Storage modules
        "conversation_store": "src.modules.storage.conversation_store.module.ConversationStoreModule",
        "conversation_history": "src.modules.storage.conversation_history.module.ConversationHistoryModule",
        "file_storage": "src.modules.storage.file_storage.module.FileStorageModule",
        "database": "src.modules.storage.database.module.DatabaseModule",
        
        # Auth module
        "user": "src.modules.auth.module.UserModule",
        
        # Tutoring modules (só criados se ENABLE_TUTORING_MODULES=true)
        "student_model": "src.modules.tutoring.student_model.module.StudentModelModule",
        "diagnostic_module": "src.modules.tutoring.diagnostic.module.DiagnosticModule",
        "pedagogical_policy": "src.modules.tutoring.pedagogical_policy.module.PedagogicalPolicyModule",
        "learning_path": "src.modules.tutoring.learning_path.module.LearningPathModule",
        
        # Realtime module
        "rest_polling": "src.modules.realtime.rest_polling.module.RestPollingModule",
    }
    
    if module_name not in module_map:
        raise ValueError(f"Unknown module: {module_name}")
    
    # Importar e instanciar
    module_path = module_map[module_name]
    parts = module_path.split(".")
    class_name = parts[-1]
    module_path = ".".join(parts[:-1])
    
    try:
        module = __import__(module_path, fromlist=[class_name])
        module_class = getattr(module, class_name)
        instance = module_class()
        return instance
    except ImportError as e:
        # All modules should be available - raise error if not found
        logger.error(f"❌ Module {module_name} not found at {module_path}: {e}")
        raise ValueError(f"Module '{module_name}' not found. Please ensure all modules are properly installed.")


def _create_disabled_tutoring_wrapper(module_name: str) -> Any:
    """Cria wrapper para módulos de tutoring desativados"""
    from src.modules.base_module import BaseModule
    
    class DisabledTutoringWrapper(BaseModule):
        """Wrapper para módulos de tutoring desativados"""
        
        def __init__(self):
            super().__init__(module_name)
            self.disabled = True
        
        async def _initialize(self) -> bool:
            """Inicialização - retorna True mas módulo está desativado"""
            self.logger.warning(f"⚠️  {module_name} is disabled (ENABLE_TUTORING_MODULES=false)")
            return True
        
        def __getattr__(self, name):
            """Retorna função que levanta erro informando que módulo está desativado"""
            if name.startswith('_'):
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
            async def disabled_method(*args, **kwargs):
                raise RuntimeError(
                    f"Module '{module_name}' is disabled. "
                    f"Set ENABLE_TUTORING_MODULES=true environment variable to enable tutoring modules."
                )
            
            return disabled_method
    
    return DisabledTutoringWrapper()


# BasicWrapper removed - all modules are now real implementations
# If a module is not found, we raise an error instead of creating a wrapper


def get_all_modules() -> Dict[str, Any]:
    """Retorna todos os módulos criados"""
    return _module_instances.copy()


def clear_cache():
    """Limpa cache de módulos (útil para testes)"""
    global _module_instances
    _module_instances.clear()
