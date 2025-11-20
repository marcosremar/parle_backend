#!/usr/bin/env python3
"""
ProcessContext - One per process

Manages resources specific to a process:
- Communication Strategy (in-process, gRPC, HTTP)
- Resource Limits (CPU, RAM, GPU)
- Reference to GlobalContext
"""

import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ResourceLimits:
    """Resource limits for a process"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize resource limits

        Args:
            config: Limits configuration from profile
        """
        if config is None:
            config = {}

        self.max_cpu_percent = config.get('max_cpu_percent', 80)
        self.max_ram_mb = config.get('max_ram_mb', 4096)
        self.max_gpu_mb = config.get('max_gpu_mb', None)  # None = no limit

        logger.debug(
            f"Resource limits: "
            f"CPU={self.max_cpu_percent}%, "
            f"RAM={self.max_ram_mb}MB, "
            f"GPU={self.max_gpu_mb or 'unlimited'}MB"
        )

    def to_dict(self) -> dict:
        """Export limits as dict"""
        return {
            'max_cpu_percent': self.max_cpu_percent,
            'max_ram_mb': self.max_ram_mb,
            'max_gpu_mb': self.max_gpu_mb
        }


class ProcessContext:
    """
    One per process
    Gerencia recursos específicos do processo

    Usage:
        # In ServiceManager:
        global_ctx = GlobalContext.get_instance()
        process_ctx = ProcessContext(
            process_id="service_manager",
            execution_mode="module",
            global_context=global_ctx
        )

        # In services:
        communication = process_ctx.communication
        gpu = process_ctx.global_context.gpu_manager
    """

    def __init__(
        self,
        process_id: str,
        execution_mode: str,
        global_context,
        config: Optional[Dict] = None
    ):
        """
        Initialize ProcessContext

        Args:
            process_id: Process identifier (e.g., "service_manager", "llm_worker")
            execution_mode: "module", "internal", or "external"
            global_context: GlobalContext instance
            config: Process-specific configuration
        """
        self.process_id = process_id
        self.execution_mode = execution_mode
        self.global_context = global_context
        self.config = config or {}

        # Communication Strategy (decided at runtime based on execution_mode)
        self.communication = None

        # Resource Limits
        self.limits = ResourceLimits(self.config.get('limits', {}))

        logger.info(
            f"🔧 ProcessContext created: {process_id} "
            f"(mode: {execution_mode}, PID: {os.getpid()})"
        )

    async def initialize(self):
        """
        Initialize ProcessContext

        Creates communication strategy based on execution_mode
        """
        logger.info(f"🚀 Initializing ProcessContext: {self.process_id}")

        # Create communication strategy
        self.communication = await self._create_communication_strategy()

        logger.info(
            f"✅ ProcessContext initialized: {self.process_id}\n"
            f"   Mode: {self.execution_mode}\n"
            f"   Communication: {type(self.communication).__name__ if self.communication else 'None'}"
        )

    async def _create_communication_strategy(self):
        """
        Create ServiceCommunicationManager with automatic protocol selection

        Returns:
            ServiceCommunicationManager instance
        """
        try:
            from src.core.managers.communication_manager import ServiceCommunicationManager

            # Create ServiceCommunicationManager (unified communication system)
            # It automatically selects protocol based on service type:
            # - module (in-process) → Direct call (zero overhead)
            # - service (separate process) → HTTP
            # - external (remote) → HTTP with resilience
            comm_manager = ServiceCommunicationManager()
            await comm_manager.initialize()

            logger.info(
                f"   📞 Communication: ServiceCommunicationManager\n"
                f"      Auto-selects: Direct → ZeroMQ → gRPC → HTTP → JSON\n"
                f"      Execution mode: {self.execution_mode}"
            )

            return comm_manager

        except ImportError as e:
            logger.warning(
                f"⚠️ ServiceCommunicationManager not available: {e}\n"
                f"   Using stub (no-op) communication"
            )
            # Return a stub/no-op communication
            return StubCommunication(self)

    async def shutdown(self):
        """Shutdown ProcessContext"""
        logger.info(f"🛑 Shutting down ProcessContext: {self.process_id}")

        # Shutdown communication
        if self.communication and hasattr(self.communication, 'shutdown'):
            try:
                await self.communication.shutdown()
                logger.info("   Communication shutdown complete")
            except Exception as e:
                logger.error(f"   Communication shutdown error: {e}")

        logger.info(f"✅ ProcessContext shutdown complete: {self.process_id}")

    def get_status(self) -> dict:
        """Get ProcessContext status"""
        return {
            "process_id": self.process_id,
            "execution_mode": self.execution_mode,
            "pid": os.getpid(),
            "communication": type(self.communication).__name__ if self.communication else None,
            "limits": self.limits.to_dict()
        }


class StubCommunication:
    """
    Stub communication for when real strategies are not available
    Used during development/testing
    """

    def __init__(self, process_context):
        self.process_context = process_context
        logger.warning("⚠️ Using StubCommunication (no-op)")

    async def call(self, service_name: str, method: str, params: dict = None):
        """Stub call - logs and returns None"""
        logger.warning(
            f"StubCommunication.call({service_name}.{method}) - "
            f"No real communication available"
        )
        return None

    async def shutdown(self):
        """Stub shutdown"""
        pass
