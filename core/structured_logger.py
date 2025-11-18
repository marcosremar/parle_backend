"""
⚠️ DEPRECATED (v5.3): This module has been consolidated into src.core.core_logging

Use instead:
    from src.core.core_logging import get_logger  # NEW

Old (DEPRECATED):
    from src.core.structured_logger import get_logger

This file is kept for backward compatibility only and will be removed in v6.0.

Sistema de logging estruturado
"""
import warnings
import logging
import sys
from typing import Any, Dict

# Emit deprecation warning
warnings.warn(
    "structured_logger.py is deprecated. Use 'from src.core.core_logging import get_logger' instead.",
    DeprecationWarning,
    stacklevel=2
)

class StructuredLogger:
    """Logger estruturado com formatação específica"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            # Configurar handler apenas se não existir
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, **kwargs):
        """Log info com contexto"""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.info(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug com contexto"""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.debug(message)
    
    def error(self, message: str, **kwargs):
        """Log error com contexto"""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning com contexto"""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.warning(message)
    
    def success(self, message: str, **kwargs):
        """Log success (info colorido)"""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.info(f"✅ {message}")

def get_logger(name: str) -> StructuredLogger:
    """Obtém um logger estruturado"""
    return StructuredLogger(name)