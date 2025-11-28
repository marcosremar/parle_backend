"""
Módulos Internos - Chamadas Diretas Python
Wrappers dos serviços para uso interno sem overhead HTTP
"""

from .module_factory import create, get_all_modules, clear_cache

__all__ = ["create", "get_all_modules", "clear_cache"]
