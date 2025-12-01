"""
Módulos Internos - Chamadas Diretas Python
Wrappers dos serviços para uso interno sem overhead HTTP
"""

from .module_factory import clear_cache, create, get_all_modules

__all__ = ["clear_cache", "create", "get_all_modules"]
