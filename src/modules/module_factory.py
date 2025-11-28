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
    
    # Mapeamento de nomes de módulos para classes
    module_map = {
        # Speech modules
        "stt": "src.modules.speech.stt_module.STTModule",
        "tts": "src.modules.speech.tts_module.TTSModule",
        # "neural_codec": "src.modules.speech.neural_codec_module.NeuralCodecModule",  # TODO
        
        # LLM module
        "llm": "src.modules.llm.llm_module.LLMModule",
        
        # Conversation modules
        "orchestrator": "src.modules.conversation.orchestrator_module.OrchestratorModule",
        "session": "src.modules.conversation.session_module.SessionModule",
        "scenarios": "src.modules.conversation.scenarios_module.ScenariosModule",
        
        # Storage modules
        "conversation_store": "src.modules.storage.conversation_store_module.ConversationStoreModule",
        "conversation_history": "src.modules.storage.conversation_history_module.ConversationHistoryModule",
        "file_storage": "src.modules.storage.file_storage_module.FileStorageModule",
        "database": "src.modules.storage.database_module.DatabaseModule",
        
        # Auth module
        "user": "src.modules.auth.user_module.UserModule",
        
        # Tutoring modules
        "student_model": "src.modules.tutoring.student_model_module.StudentModelModule",
        "diagnostic_module": "src.modules.tutoring.diagnostic_module.DiagnosticModule",
        "pedagogical_policy": "src.modules.tutoring.pedagogical_policy_module.PedagogicalPolicyModule",
        "learning_path": "src.modules.tutoring.learning_path_module.LearningPathModule",
        
        # Realtime module
        "rest_polling": "src.modules.realtime.rest_polling_module.RestPollingModule",
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
    except ImportError:
        # Fallback: criar wrapper básico usando o serviço existente
        logger.warning(f"Module {module_name} not found, creating basic wrapper")
        return _create_basic_wrapper(module_name)


def _create_basic_wrapper(module_name: str) -> Any:
    """Cria wrapper básico que usa o serviço existente diretamente"""
    from src.modules.base_module import BaseModule
    
    class BasicWrapper(BaseModule):
        def __init__(self):
            super().__init__(module_name)
            # Importar serviço correspondente
            self._import_service()
        
        async def _initialize(self) -> bool:
            """Inicialização básica - retorna True sempre"""
            return True
        
        def _import_service(self):
            """Importa o serviço correspondente"""
            service_map = {
                "stt": ("src.services.stt", "STTService"),
                "tts": ("src.services.tts", "TTSService"),
                "llm": ("src.services.llm", "LLMService"),
                "user": ("src.services.user", "UserService"),
                "orchestrator": ("src.services.orchestrator", "OrchestratorService"),
                "session": ("src.services.session", "SessionService"),
                "scenarios": ("src.services.scenarios", "ScenariosService"),
                "conversation_store": ("src.services.conversation_store", "ConversationStoreService"),
                "conversation_history": ("src.services.conversation_history", "ConversationHistoryService"),
                "file_storage": ("src.services.file_storage", "FileStorageService"),
                "database": ("src.services.database", "DatabaseService"),
                "student_model": ("src.services.student_model", "StudentModelService"),
                "diagnostic_module": ("src.services.diagnostic_module", "DiagnosticModuleService"),
                "pedagogical_policy": ("src.services.pedagogical_policy", "PedagogicalPolicyService"),
                "learning_path": ("src.services.learning_path", "LearningPathService"),
                "rest_polling": ("src.services.rest_polling", "RestPollingService"),
            }
            
            if module_name in service_map:
                module_path, class_name = service_map[module_name]
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    self.service_class = getattr(module, class_name, None)
                    self.service_instance = None
                except ImportError:
                    self.service_class = None
                    self.service_instance = None
    
    return BasicWrapper()


def get_all_modules() -> Dict[str, Any]:
    """Retorna todos os módulos criados"""
    return _module_instances.copy()


def clear_cache():
    """Limpa cache de módulos (útil para testes)"""
    global _module_instances
    _module_instances.clear()
