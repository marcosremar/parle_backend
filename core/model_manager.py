#!/usr/bin/env python3
"""
Model Manager - Singleton Pattern for Shared Models
Ensures Ultravox and Kokoro TTS are loaded only once and shared
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from threading import Lock
import time

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for shared model instances
    Ensures models are loaded once and shared between WebRTC and REST API
    """

    _instance: Optional['ModelManager'] = None
    _lock = Lock()
    _initialized = False

    def __new__(cls) -> 'ModelManager':
        """Ensure only one instance exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize manager (called only once)"""
        if not ModelManager._initialized:
            self.pipeline = None
            self.initialization_lock = asyncio.Lock()
            self.initialization_time = None
            self.initialization_status = "not_started"  # not_started, initializing, completed, failed
            self.gpu_memory_usage = None
            ModelManager._initialized = True
            logger.info("🏗️ ModelManager singleton created")

    async def get_pipeline(self):
        """
        Get the shared pipeline instance, initializing if needed

        Returns:
            ConversationPipeline: Shared pipeline instance
        """
        if self.pipeline is not None:
            logger.debug("♻️ Returning existing pipeline instance")
            return self.pipeline

        # Use async lock to prevent concurrent initialization
        async with self.initialization_lock:
            # Double-check pattern - another coroutine might have initialized while waiting
            if self.pipeline is not None:
                logger.debug("♻️ Pipeline was initialized by another coroutine")
                return self.pipeline

            # Initialize pipeline for the first time
            await self._initialize_pipeline()
            return self.pipeline

    async def _initialize_pipeline(self):
        """Initialize the shared pipeline instance"""
        logger.info("🚀 Initializing shared ConversationPipeline...")
        self.initialization_status = "initializing"
        start_time = time.time()

        try:
            # Import here to avoid circular imports
            from pipeline.conversation import LocalConversationPipeline

            # Create HTTP-based pipeline (não carrega modelos)
            self.pipeline = LocalConversationPipeline(
                ultravox_url="http://localhost:8100",
                kokoro_url="http://localhost:8101",
                language="Portuguese",
                voice="pf_dora"
            )
            await self.pipeline.initialize()

            # Track initialization metrics
            self.initialization_time = time.time() - start_time
            self.initialization_status = "completed"

            # Try to get GPU memory usage
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
            except (ImportError, RuntimeError):
                # PyTorch not available or CUDA not accessible
                pass

            logger.info(f"✅ Shared ConversationPipeline initialized in {self.initialization_time:.1f}s")
            if self.gpu_memory_usage:
                logger.info(f"💾 GPU Memory Usage: {self.gpu_memory_usage:.2f}GB")

        except Exception as e:
            self.initialization_status = "failed"
            logger.error(f"❌ Failed to initialize shared pipeline: {e}")
            raise

    def is_initialized(self) -> bool:
        """Check if pipeline is initialized"""
        return self.pipeline is not None and self.initialization_status == "completed"

    def get_status(self) -> Dict[str, Any]:
        """Get manager status and metrics"""
        return {
            "initialized": self.is_initialized(),
            "status": self.initialization_status,
            "initialization_time_seconds": self.initialization_time,
            "gpu_memory_gb": self.gpu_memory_usage,
            "pipeline_available": self.pipeline is not None
        }

    async def clear_pipeline(self):
        """Clear the pipeline (for testing/debugging)"""
        async with self.initialization_lock:
            if self.pipeline:
                logger.info("🗑️ Clearing shared pipeline")
                # Try to free GPU memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    # PyTorch not available or CUDA cleanup not possible
                    pass

                self.pipeline = None
                self.initialization_status = "not_started"
                self.initialization_time = None
                self.gpu_memory_usage = None
                logger.info("✅ Shared pipeline cleared")


# Global singleton instance
model_manager = ModelManager()


async def get_shared_pipeline():
    """
    Convenience function to get the shared pipeline

    Returns:
        ConversationPipeline: Shared pipeline instance
    """
    return await model_manager.get_pipeline()


def get_model_status() -> Dict[str, Any]:
    """
    Get current model manager status

    Returns:
        Dict with status information
    """
    return model_manager.get_status()