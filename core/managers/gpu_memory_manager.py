#!/usr/bin/env python3
"""
GPU Memory Manager V2 - with SharedGPUState and Universal Ultravox support

New features:
- multiprocessing.Manager for cross-process state (100K ops/s)
- Universal backend selection (vLLM RTX + Vulkan others)
- Simplified API aligned with Context architecture
"""

import os
import logging
import torch
from typing import Dict, Optional
from pathlib import Path

# Import SharedGPUState from context
try:
    from .context.shared_state import SharedGPUState, GPUMemoryError
    SHARED_STATE_AVAILABLE = True
except ImportError:
    SHARED_STATE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SharedGPUState not available, using legacy mode")

# Import pynvml for GPU detection
try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, Exception) as e:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUManager:
    """
    GPU Manager V2 with shared state and universal backend support

    Features:
    - Shared state via multiprocessing.Manager (cross-process coordination)
    - Universal Ultravox support (vLLM RTX + Vulkan others)
    - Automatic backend selection based on GPU model
    - Simple, clean API for ServiceContext integration
    """

    def __init__(self, shared_state: Optional[SharedGPUState] = None):
        """
        Initialize GPU Manager

        Args:
            shared_state: Shared GPU state (created by GlobalContext)
                         If None, creates new one (legacy mode)
        """
        self.shared_state = shared_state or (SharedGPUState() if SHARED_STATE_AVAILABLE else None)
        self.gpu_name = None
        self.compute_cap = None
        self.total_mb = 0
        self.device_id = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(',')[0])

        logger.info(f"✅ GPU Manager V2 initialized (device {self.device_id})")

    async def initialize(self):
        """
        Eager initialization (called by GlobalContext)

        Detects GPU and updates shared state
        """
        if not torch.cuda.is_available():
            logger.warning("❌ CUDA not available")
            return

        try:
            # Get GPU info
            if PYNVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)

                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                self.gpu_name = name
                self.total_mb = info.total / (1024**2)
            else:
                # Fallback to PyTorch
                props = torch.cuda.get_device_properties(self.device_id)
                self.gpu_name = props.name
                self.total_mb = props.total_memory / (1024**2)

            self.compute_cap = torch.cuda.get_device_capability(self.device_id)

            # Update shared state
            if self.shared_state:
                self.shared_state.update_gpu_info({
                    "name": self.gpu_name,
                    "compute_cap": self.compute_cap,
                    "total_mb": self.total_mb,
                    "device_id": self.device_id
                })

            logger.info(
                f"🎯 GPU detected:\n"
                f"   Name: {self.gpu_name}\n"
                f"   Compute: sm_{self.compute_cap[0]}{self.compute_cap[1]}\n"
                f"   Memory: {self.total_mb:.0f}MB"
            )

        except Exception as e:
            logger.error(f"❌ GPU initialization failed: {e}")

    def get_recommendations(
        self,
        memory_gb: float,
        service_name: str
    ) -> dict:
        """
        Get backend recommendations for Ultravox

        Args:
            memory_gb: Required memory in GB
            service_name: Service identifier

        Returns:
            {
                "backend": "vllm" | "vulkan" | "external_api",
                "strategy": "gpu" | "offload" | "external",
                "model": "model_path",
                "memory_mb": float
            }
        """
        memory_mb = memory_gb * 1024

        # Get GPU info
        gpu_name = self.gpu_name.lower() if self.gpu_name else ""
        sm_version = self.compute_cap[0] * 10 + self.compute_cap[1] if self.compute_cap else 0

        # RTX 3090/4090/A100 → vLLM + original model
        optimized_gpus = ['3090', '4090', 'a100', 'h100', 'a6000', 'rtx 6000']
        for gpu in optimized_gpus:
            if gpu in gpu_name:
                try:
                    if self.shared_state:
                        self.shared_state.reserve(service_name, memory_mb, "vllm")

                    logger.info(f"✅ {service_name}: vLLM backend (optimized GPU)")

                    return {
                        "backend": "vllm",
                        "strategy": "gpu",
                        "model": "fixie-ai/ultravox-v0_6-llama-3_1-8b",
                        "memory_mb": memory_mb,
                        "gpu_utilization": 0.75
                    }
                except GPUMemoryError as e:
                    logger.warning(f"⚠️ GPU full: {e}")
                    return {
                        "backend": "external_api",
                        "strategy": "external",
                        "reason": "GPU memory full",
                        "alternative": "Groq API"
                    }

        # Outras GPUs → Vulkan + GGUF
        if torch.cuda.is_available() and sm_version > 0:
            # Modelo quantizado usa menos memória (5GB)
            vulkan_memory = 5000  # 5GB

            try:
                if self.shared_state:
                    self.shared_state.reserve(service_name, vulkan_memory, "vulkan")

                logger.info(
                    f"✅ {service_name}: Vulkan backend (universal compatibility)\n"
                    f"   GPU: {self.gpu_name} (sm_{sm_version})"
                )

                return {
                    "backend": "vulkan",
                    "strategy": "gpu",
                    "model": "ultravox-v0_5-llama-3_2-1b-Q4_K_M.gguf",
                    "model_repo": "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
                    "memory_mb": vulkan_memory,
                    "quantization": "Q4_K_M"
                }
            except GPUMemoryError as e:
                logger.warning(f"⚠️ GPU full: {e}")
                return {
                    "backend": "external_api",
                    "strategy": "external",
                    "reason": "GPU memory full",
                    "alternative": "Groq API"
                }

        # Sem GPU → External API
        logger.info(f"ℹ️ {service_name}: External API (no compatible GPU)")
        return {
            "backend": "external_api",
            "strategy": "external",
            "reason": "No GPU available",
            "alternative": "Groq API"
        }

    def reserve(self, service_name: str, memory_mb: float, backend: str = "vllm") -> dict:
        """
        Reserve GPU memory

        Args:
            service_name: Service identifier
            memory_mb: Memory in MB
            backend: Backend type

        Returns:
            Allocation status

        Raises:
            GPUMemoryError: If not enough memory
        """
        if self.shared_state:
            return self.shared_state.reserve(service_name, memory_mb, backend)
        else:
            # Legacy mode: no reservation
            logger.warning(f"⚠️ Shared state not available, skipping reservation for {service_name}")
            return {"allocated": memory_mb, "free": 0, "total": 0}

    def release(self, service_name: str):
        """Release GPU memory"""
        if self.shared_state:
            self.shared_state.release(service_name)
        else:
            logger.warning(f"⚠️ Shared state not available, skipping release for {service_name}")

    def get_status(self) -> dict:
        """Get GPU status"""
        if self.shared_state:
            return self.shared_state.get_status()
        else:
            return {
                "allocations": {},
                "total_allocated_mb": 0,
                "free_mb": 0,
                "total_mb": 0,
                "services_count": 0
            }

    def get_gpu_info(self) -> dict:
        """Get GPU information"""
        if self.shared_state:
            return self.shared_state.get_gpu_info()
        else:
            return {
                "name": self.gpu_name,
                "compute_cap": self.compute_cap,
                "total_mb": self.total_mb,
                "device_id": self.device_id
            }


# Singleton instance (will be replaced by GlobalContext)
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """
    Get or create global GPU manager instance

    NOTE: This is legacy API. New code should use:
        context.gpu  (from ServiceContext)
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
