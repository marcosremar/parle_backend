"""
Queue Controller with Worker Pool
Manages request processing with configurable concurrency
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
from enum import Enum
from .base_controller import BaseController
from .conversation_controller import ConversationController

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Request priority levels"""
    HIGH = 1    # Audio requests (real-time)
    NORMAL = 2  # Text requests
    LOW = 3     # Background tasks


@dataclass
class QueuedRequest:
    """Request in queue"""
    data: Dict[str, Any]
    priority: Priority
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)

    def __lt__(self, other):
        """For priority queue sorting"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


class QueueController(BaseController):
    """
    Controller with queue and worker pool
    Limits concurrent processing to optimize latency
    """

    def __init__(self, max_workers: int = None):
        """
        Initialize queue controller

        Args:
            max_workers: Maximum concurrent workers (if None, uses settings.yaml)
        """
        super().__init__()

        # Load from settings if not provided
        if max_workers is None:
            from core.settings import get_worker_settings
            worker_settings = get_worker_settings()
            self.max_workers = worker_settings.max_workers
            logger.info(f"📋 Loading max_workers from settings: {self.max_workers}")
        else:
            self.max_workers = max_workers
            logger.info(f"📋 Using provided max_workers: {self.max_workers}")
        self.request_queue = asyncio.PriorityQueue()
        self.active_workers = 0
        self.workers = []
        self.conversation_controller = None
        self.stats = {
            'total_requests': 0,
            'queued_requests': 0,
            'processed_requests': 0,
            'avg_queue_time_ms': 0,
            'avg_processing_time_ms': 0
        }
        self.queue_times = []
        self.processing_times = []

    async def initialize(self):
        """Initialize the queue controller"""
        logger.info(f"🚀 Initializing QueueController with {self.max_workers} workers")

        # Initialize conversation controller
        self.conversation_controller = ConversationController()
        await self.conversation_controller.initialize()

        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

        logger.info(f"✅ QueueController initialized with {self.max_workers} workers")

    async def _worker(self, worker_id: int):
        """
        Worker coroutine that processes requests from queue

        Args:
            worker_id: Worker identifier
        """
        logger.info(f"👷 Worker {worker_id} started")

        while True:
            try:
                # Get request from queue
                _, request = await self.request_queue.get()

                queue_time = (time.time() - request.timestamp) * 1000
                self.queue_times.append(queue_time)

                logger.info(f"👷 Worker {worker_id} processing request after {queue_time:.0f}ms in queue")

                # Process request
                start_time = time.time()
                self.active_workers += 1

                try:
                    result = await self.conversation_controller.handle_request(request.data)

                    processing_time = (time.time() - start_time) * 1000
                    self.processing_times.append(processing_time)

                    # Add queue metrics to result
                    result['metrics']['queue_time_ms'] = queue_time
                    result['metrics']['worker_id'] = worker_id
                    result['metrics']['active_workers'] = self.active_workers

                    request.future.set_result(result)
                    self.stats['processed_requests'] += 1

                    logger.info(f"✅ Worker {worker_id} completed in {processing_time:.0f}ms")

                except Exception as e:
                    logger.error(f"❌ Worker {worker_id} error: {e}")
                    request.future.set_exception(e)
                finally:
                    self.active_workers -= 1

            except asyncio.CancelledError:
                logger.info(f"👷 Worker {worker_id} shutting down")
                break
            except Exception as e:
                logger.error(f"❌ Worker {worker_id} unexpected error: {e}")

    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Queue request for processing

        Args:
            request_data: Request to process

        Returns:
            Response dictionary
        """
        self.stats['total_requests'] += 1

        # Determine priority
        request_type = request_data.get('type', 'text')
        if request_type == 'audio':
            priority = Priority.HIGH
        elif request_type == 'control':
            priority = Priority.NORMAL
        else:
            priority = Priority.NORMAL

        # Create queued request
        queued_request = QueuedRequest(
            data=request_data,
            priority=priority
        )

        # Add to queue
        queue_size = self.request_queue.qsize()
        await self.request_queue.put((priority.value, queued_request))
        self.stats['queued_requests'] = max(self.stats['queued_requests'], queue_size + 1)

        logger.info(f"📥 Request queued (priority={priority.name}, queue_size={queue_size + 1})")

        # Wait for result
        try:
            result = await asyncio.wait_for(queued_request.future, timeout=30.0)

            # Update statistics
            if self.queue_times:
                self.stats['avg_queue_time_ms'] = sum(self.queue_times) / len(self.queue_times)
            if self.processing_times:
                self.stats['avg_processing_time_ms'] = sum(self.processing_times) / len(self.processing_times)

            return result

        except asyncio.TimeoutError:
            logger.error("⏰ Request timeout after 30 seconds")
            return {
                'success': False,
                'error': 'Request timeout',
                'metrics': {'timeout': True}
            }

    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request - delegates to conversation controller"""
        if self.conversation_controller:
            return await self.conversation_controller.validate_request(request_data)
        return {'valid': True}

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request - not used, workers handle processing"""
        pass  # Workers handle the actual processing

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            'current_queue_size': self.request_queue.qsize(),
            'active_workers': self.active_workers,
            'max_workers': self.max_workers
        }

    async def shutdown(self):
        """Shutdown queue controller"""
        logger.info("🛑 Shutting down QueueController")

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        logger.info("✅ QueueController shutdown complete")