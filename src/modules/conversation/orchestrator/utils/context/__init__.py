#!/usr/bin/env python3
"""
Context and Dependency Injection system (v4.0)

UNIFIED ARCHITECTURE (v4.0):
- ServiceContext: Single unified context (replaces 3-layer system)

DEPRECATED (v3.x - kept for backward compatibility):
- GlobalContext: DEPRECATED - Use ServiceContext instead
- ProcessContext: DEPRECATED - Use ServiceContext instead
- ServiceContext (old): DEPRECATED - Use unified ServiceContext from unified_context.py
"""

import warnings

# v4.0: Unified context system
from ..unified_context import ServiceContext as UnifiedServiceContext, ResourceLimits as UnifiedResourceLimits

# v3.x: Old context system (DEPRECATED - kept for backward compatibility)
from .shared_state import SharedGPUState, GPUMemoryError
from .global_context import GlobalContext as OldGlobalContext
from .process_context import ProcessContext as OldProcessContext, ResourceLimits as OldResourceLimits
from .service_context import ServiceContext as OldServiceContext, LoggerFactory

# Export unified context as primary
ServiceContext = UnifiedServiceContext
ResourceLimits = UnifiedResourceLimits

# Deprecated exports (for backward compatibility)
GlobalContext = OldGlobalContext
ProcessContext = OldProcessContext

__all__ = [
    # v4.0: Unified context (PRIMARY)
    'ServiceContext',
    'ResourceLimits',

    # GPU shared state (still used)
    'SharedGPUState',
    'GPUMemoryError',

    # DEPRECATED (v3.x - backward compatibility only)
    'GlobalContext',  # DEPRECATED
    'ProcessContext',  # DEPRECATED
    'LoggerFactory',  # DEPRECATED
]

# Show deprecation warning when old context classes are imported
def _warn_deprecated():
    warnings.warn(
        "Importing GlobalContext/ProcessContext from src.core.context is deprecated. "
        "Use ServiceContext from src.core.unified_context instead.",
        DeprecationWarning,
        stacklevel=3
    )
