"""
Core managers for various system aspects

This module provides centralized access to all system managers.
Managers handle cross-cutting concerns like communication, sessions,
profiles, GPU allocation, and service warmup.

Usage:
    from src.core.managers import (
        get_communication_manager,
        get_profile_manager,
        session_manager,
    )

    # Get communication manager (singleton)
    comm = get_communication_manager()
    await comm.call_service("orchestrator", "/process", "POST", data)

    # Check if service enabled in current profile
    pm = get_profile_manager()
    if pm.is_service_enabled("llm"):
        # Start LLM service
        pass
"""

# ═══════════════════════════════════════════════════════════════════
# COMMUNICATION MANAGER
# ═══════════════════════════════════════════════════════════════════
from .communication_manager import (
    ServiceCommunicationManager,
    get_communication_manager,
)

# ═══════════════════════════════════════════════════════════════════
# SESSION MANAGER
# ═══════════════════════════════════════════════════════════════════
from .session_manager import (
    SessionManager,
    Session,
    Message,
    session_manager,  # Singleton instance
    create_session,
    get_session,
    get_or_create_session,
    add_interaction,
    get_context,
)

# ═══════════════════════════════════════════════════════════════════
# PROFILE MANAGERS (Environment + GPU)
# ═══════════════════════════════════════════════════════════════════
from .profile_manager import (
    ProfileManager,
    get_profile_manager,
    Profile,
)

from .gpu_profile_manager import (
    GPUProfileManager,
    get_gpu_profile_manager,
    GPUProfile,
    ServiceConfig,
)

# ═══════════════════════════════════════════════════════════════════
# WARMUP MANAGERS (Base + Profile-Aware)
# ═══════════════════════════════════════════════════════════════════
from .warmup_manager import (
    WarmupManager,
    get_warmup_manager,
    WarmupStatus,
    WarmupResult,
)

from .profile_warmup_manager import (
    ProfileWarmupManager,
    ProfileWarmupMetrics,
)

# ═══════════════════════════════════════════════════════════════════
# OPTIMIZATION & BENCHMARKING
# ═══════════════════════════════════════════════════════════════════
from .benchmark_manager import (
    BenchmarkManager,
    get_benchmark_manager,
)

# ═══════════════════════════════════════════════════════════════════
# RESOURCE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
from .gpu_memory_manager import GPUManager

from .port_pool import PortPool

from .isolated_service_manager import IsolatedServiceManager

# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════
__all__ = [
    # ─── Communication ───
    "ServiceCommunicationManager",
    "get_communication_manager",

    # ─── Session Management ───
    "SessionManager",
    "Session",
    "Message",
    "session_manager",
    "create_session",
    "get_session",
    "get_or_create_session",
    "add_interaction",
    "get_context",

    # ─── Profile Management ───
    "ProfileManager",
    "get_profile_manager",
    "Profile",
    "GPUProfileManager",
    "get_gpu_profile_manager",
    "GPUProfile",
    "ServiceConfig",

    # ─── Warmup Management ───
    "WarmupManager",
    "get_warmup_manager",
    "WarmupStatus",
    "WarmupResult",
    "ProfileWarmupManager",
    "ProfileWarmupMetrics",

    # ─── Optimization & Benchmarking ───
    "BenchmarkManager",
    "get_benchmark_manager",

    # ─── Resource Management ───
    "GPUManager",
    "PortPool",
    "IsolatedServiceManager",
]
