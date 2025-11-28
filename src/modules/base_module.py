"""
Base Module - Classe base para todos os módulos internos
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger


class BaseModule(ABC):
    """Classe base para módulos internos (chamadas diretas Python)"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.initialized = False
        self.logger = logger.bind(module=module_name)
    
    async def initialize(self) -> bool:
        """Inicializa o módulo"""
        if self.initialized:
            return True
        
        try:
            result = await self._initialize()
            self.initialized = result
            if result:
                self.logger.info(f"✅ Module {self.module_name} initialized")
            return result
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {self.module_name}: {e}")
            return False
    
    @abstractmethod
    async def _initialize(self) -> bool:
        """Implementação específica de inicialização"""
        pass
    
    async def cleanup(self):
        """Limpa recursos do módulo"""
        if hasattr(self, '_cleanup'):
            await self._cleanup()
        self.initialized = False
    
    def __repr__(self):
        return f"<{self.__class__.__name__}({self.module_name})>"
