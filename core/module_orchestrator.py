"""
Module Orchestrator - Orquestrador central para todos os módulos
Gerencia ciclo de vida, status e coordenação entre módulos
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from .status_manager import StatusManager, ModuleStatus, SystemStatus
from .stt.factory import STTFactory
from .llm.factory import LLMFactory
from .tts.factory import TTSFactory
from .ultravox_separated import UltravoxSeparatedProcessor
from src.core.interfaces import IMemoryStore
from src.core.config import Config

logger = logging.getLogger(__name__)


class ModuleType(Enum):
    """Tipos de módulos disponíveis"""
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    MEMORY = "memory"
    ULTRAVOX = "ultravox"


class ModuleOrchestrator:
    """
    Orquestrador central para gerenciar todos os módulos
    Fornece interface unificada com gestão independente
    """
    
    def __init__(self, config: Config):
        """
        Initialize orchestrator
        
        Args:
            config: System configuration
        """
        self.config = config
        self.status_manager = StatusManager()
        
        # Module instances
        self.modules: Dict[str, Any] = {}
        
        # Module metadata
        self.module_info: Dict[str, Dict[str, Any]] = {}
        
        # Initialization state
        self.is_initialized = False
        
        logger.info("🎭 Module Orchestrator initialized")
    
    async def initialize_module(self, 
                               module_type: ModuleType,
                               module_name: str,
                               **kwargs) -> None:
        """
        Initialize a specific module
        
        Args:
            module_type: Type of module
            module_name: Unique name for the module
            **kwargs: Module-specific configuration
        """
        logger.info(f"🚀 Initializing module: {module_name} (type: {module_type.value})")
        
        # Update status
        self.status_manager.update_module_status(module_name, ModuleStatus.INITIALIZING)
        
        try:
            # Create module based on type
            if module_type == ModuleType.STT:
                engine = kwargs.get('engine', 'ultravox_stt')
                if engine == 'ultravox_stt':
                    from .stt.ultravox_stt import UltravoxSTTModule
                    module = UltravoxSTTModule(**kwargs)
                else:
                    module = STTFactory.create(engine, **kwargs)
                    
            elif module_type == ModuleType.LLM:
                engine = kwargs.get('engine', 'ultravox_llm')
                if engine == 'ultravox_llm':
                    from .llm.ultravox_llm import UltravoxLLMModule
                    module = UltravoxLLMModule(**kwargs)
                else:
                    module = LLMFactory.create(engine, **kwargs)
                    
            elif module_type == ModuleType.TTS:
                engine = kwargs.get('engine', 'kokoro')
                module = TTSFactory.create(engine, **kwargs)
                
            elif module_type == ModuleType.ULTRAVOX:
                module = UltravoxSeparatedProcessor(
                    config=kwargs.get('config', self.config.ultravox),
                    memory_store=kwargs.get('memory_store')
                )
                
            elif module_type == ModuleType.MEMORY:
                from core.memory import InMemoryStore
                module = InMemoryStore(**kwargs)
                
            else:
                raise ValueError(f"Unknown module type: {module_type}")
            
            # Initialize the module
            if hasattr(module, 'initialize'):
                await module.initialize()
            
            # Store module
            self.modules[module_name] = module
            
            # Store metadata
            self.module_info[module_name] = {
                'type': module_type.value,
                'engine': kwargs.get('engine', 'default'),
                'config': kwargs,
                'initialized_at': asyncio.get_event_loop().time()
            }
            
            # Register with status manager
            self.status_manager.register_module(module_name, module)
            self.status_manager.update_module_status(module_name, ModuleStatus.READY)
            
            logger.info(f"✅ Module {module_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize module {module_name}: {e}")
            self.status_manager.update_module_status(module_name, ModuleStatus.ERROR)
            raise
    
    async def initialize_all(self) -> None:
        """Initialize all modules based on configuration"""
        logger.info("🚀 Initializing all modules...")
        
        # Initialize Ultravox modules (STT + LLM)
        await self.initialize_module(
            ModuleType.STT,
            "ultravox_stt",
            engine="ultravox_stt",
            model_path=self.config.ultravox.model_path
        )
        
        await self.initialize_module(
            ModuleType.LLM,
            "ultravox_llm",
            engine="ultravox_llm",
            model_path=self.config.ultravox.model_path,
            gpu_memory_utilization=self.config.ultravox.gpu_memory_utilization
        )
        
        # Initialize TTS
        await self.initialize_module(
            ModuleType.TTS,
            "tts_primary",
            engine=self.config.tts.engine,
            language=self.config.tts.language
        )
        
        # Initialize Memory
        await self.initialize_module(
            ModuleType.MEMORY,
            "memory_store",
            max_sessions=100,
            max_messages_per_session=50
        )
        
        # Start health monitoring
        await self.status_manager.start_health_monitoring(interval=30)
        
        self.is_initialized = True
        logger.info("✅ All modules initialized")
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """
        Get a module instance
        
        Args:
            module_name: Module name
            
        Returns:
            Module instance or None
        """
        return self.modules.get(module_name)
    
    async def get_module_status(self, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific module
        
        Args:
            module_name: Module name
            
        Returns:
            Module status dictionary
        """
        health = await self.status_manager.get_module_status(module_name)
        if not health:
            return None
        
        return {
            'name': health.name,
            'status': health.status.value,
            'healthy': health.is_healthy,
            'response_time_ms': health.response_time_ms,
            'error': health.error_message,
            'metrics': health.metrics,
            'info': self.module_info.get(module_name, {})
        }
    
    async def get_all_status(self) -> Dict[str, Any]:
        """
        Get status of all modules
        
        Returns:
            Complete system status
        """
        system_status = await self.status_manager.get_system_status()
        
        return {
            'system': {
                'healthy': system_status.system_health,
                'uptime_seconds': system_status.uptime_seconds,
                'total_requests': system_status.total_requests,
                'active_sessions': system_status.active_sessions,
                'warnings': system_status.warnings
            },
            'modules': {
                name: {
                    'type': self.module_info.get(name, {}).get('type', 'unknown'),
                    'status': health.status.value,
                    'healthy': health.is_healthy,
                    'response_time_ms': health.response_time_ms,
                    'metrics': health.metrics
                }
                for name, health in system_status.modules.items()
            }
        }
    
    async def restart_module(self, module_name: str) -> bool:
        """
        Restart a specific module
        
        Args:
            module_name: Module name
            
        Returns:
            Success status
        """
        logger.info(f"🔄 Restarting module: {module_name}")
        
        if module_name not in self.modules:
            logger.error(f"Module not found: {module_name}")
            return False
        
        try:
            # Get module info
            module_info = self.module_info.get(module_name, {})
            
            # Cleanup existing module
            module = self.modules[module_name]
            if hasattr(module, 'cleanup'):
                await module.cleanup()
            
            # Remove from tracking
            del self.modules[module_name]
            self.status_manager.unregister_module(module_name)
            
            # Re-initialize
            await self.initialize_module(
                ModuleType(module_info['type']),
                module_name,
                **module_info['config']
            )
            
            logger.info(f"✅ Module {module_name} restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restart module {module_name}: {e}")
            return False
    
    async def cleanup_module(self, module_name: str) -> None:
        """
        Cleanup a specific module
        
        Args:
            module_name: Module name
        """
        if module_name not in self.modules:
            return
        
        logger.info(f"🧹 Cleaning up module: {module_name}")
        
        self.status_manager.update_module_status(module_name, ModuleStatus.SHUTTING_DOWN)
        
        try:
            module = self.modules[module_name]
            if hasattr(module, 'cleanup'):
                await module.cleanup()
            
            del self.modules[module_name]
            del self.module_info[module_name]
            self.status_manager.unregister_module(module_name)
            
            logger.info(f"✅ Module {module_name} cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up module {module_name}: {e}")
    
    async def cleanup_all(self) -> None:
        """Cleanup all modules"""
        logger.info("🧹 Cleaning up all modules...")
        
        # Stop health monitoring
        await self.status_manager.stop_health_monitoring()
        
        # Cleanup modules in parallel
        cleanup_tasks = []
        for module_name in list(self.modules.keys()):
            cleanup_tasks.append(self.cleanup_module(module_name))
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.is_initialized = False
        logger.info("✅ All modules cleaned up")
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """
        List all registered modules
        
        Returns:
            List of module information
        """
        return [
            {
                'name': name,
                'type': info.get('type', 'unknown'),
                'engine': info.get('engine', 'default'),
                'status': self.status_manager._module_status.get(
                    name, ModuleStatus.NOT_INITIALIZED
                ).value
            }
            for name, info in self.module_info.items()
        ]
    
    def is_healthy(self) -> bool:
        """Check if all modules are healthy"""
        return all(
            self.status_manager.is_module_healthy(name)
            for name in self.modules.keys()
        )