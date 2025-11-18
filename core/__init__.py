"""
Módulo Core
Funcionalidades compartilhadas entre todos os módulos
"""

from .metrics import get_metrics_collector
from .structured_logger import get_logger

__all__ = [
    'get_metrics_collector',
    'get_logger'
]