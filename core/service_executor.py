"""
Service Executor
Wrapper for executing requests against real services during optimization
"""

import os
import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .orchestrator_client import OrchestratorClient
from .performance_test_runner import TestRequest
from .logging import get_logger

logger = get_logger("ServiceExecutor")


@dataclass
class ServiceExecutorConfig:
    """Configuration for service executor"""
    orchestrator_url: str = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")
    use_internal_services: bool = True
    timeout_seconds: int = 30


class ServiceExecutor:
    """
    Service executor for optimization testing

    Executes test requests against real STT/LLM/TTS services
    via the orchestrator or direct service calls
    """

    def __init__(self, config: Optional[ServiceExecutorConfig] = None):
        self.config = config or ServiceExecutorConfig()
        self.orchestrator_client = OrchestratorClient(orchestrator_url=self.config.orchestrator_url)

        logger.info(
            "🔧 ServiceExecutor initialized",
            metadata={
                'orchestrator_url': self.config.orchestrator_url,
                'use_internal': self.config.use_internal_services
            }
        )

    async def execute_stt_request(self, request: TestRequest) -> Dict[str, Any]:
        """
        Execute STT request

        Args:
            request: Test request with audio payload

        Returns:
            STT response with transcript
        """
        try:
            audio = request.payload.get('audio')
            sample_rate = request.payload.get('sample_rate', 16000)
            language = request.payload.get('language', 'pt')

            # Call STT service via orchestrator
            response = await self.orchestrator_client.transcribe(
                audio=audio,
                sample_rate=sample_rate,
                language=language
            )

            return {
                'transcript': response.get('transcript', ''),
                'latency_ms': response.get('latency_ms', 0),
                'success': True
            }

        except Exception as e:
            logger.error("❌ STT request failed", exception=e)
            return {
                'transcript': '',
                'latency_ms': 0,
                'success': False,
                'error': str(e)
            }

    async def execute_llm_request(self, request: TestRequest) -> Dict[str, Any]:
        """
        Execute LLM request

        Args:
            request: Test request with prompt payload

        Returns:
            LLM response with generated text
        """
        try:
            prompt = request.payload.get('prompt')
            max_tokens = request.payload.get('max_tokens', 50)
            temperature = request.payload.get('temperature', 0.7)

            # Call LLM service via orchestrator
            response = await self.orchestrator_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            return {
                'text': response.get('text', ''),
                'latency_ms': response.get('latency_ms', 0),
                'tokens_generated': response.get('tokens_generated', 0),
                'success': True
            }

        except Exception as e:
            logger.error("❌ LLM request failed", exception=e)
            return {
                'text': '',
                'latency_ms': 0,
                'tokens_generated': 0,
                'success': False,
                'error': str(e)
            }

    async def execute_tts_request(self, request: TestRequest) -> Dict[str, Any]:
        """
        Execute TTS request

        Args:
            request: Test request with text payload

        Returns:
            TTS response with audio
        """
        try:
            text = request.payload.get('text')
            voice_id = request.payload.get('voice_id', 'af_sky')
            speed = request.payload.get('speed', 1.0)

            # Call TTS service via orchestrator
            response = await self.orchestrator_client.synthesize(
                text=text,
                voice_id=voice_id,
                speed=speed
            )

            return {
                'audio': response.get('audio'),
                'sample_rate': response.get('sample_rate', 24000),
                'latency_ms': response.get('latency_ms', 0),
                'success': True
            }

        except Exception as e:
            logger.error("❌ TTS request failed", exception=e)
            return {
                'audio': None,
                'sample_rate': 24000,
                'latency_ms': 0,
                'success': False,
                'error': str(e)
            }

    async def execute_pipeline_request(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = 'pt'
    ) -> Dict[str, Any]:
        """
        Execute full STT→LLM→TTS pipeline

        Args:
            audio: Input audio
            sample_rate: Audio sample rate
            language: Language code

        Returns:
            Pipeline response with all stages
        """
        try:
            # Stage 1: STT
            stt_request = TestRequest(
                request_id="pipeline_stt",
                module="stt",
                payload={
                    'audio': audio,
                    'sample_rate': sample_rate,
                    'language': language
                }
            )
            stt_response = await self.execute_stt_request(stt_request)

            if not stt_response['success']:
                return {
                    'success': False,
                    'error': 'STT failed',
                    'stage': 'stt'
                }

            # Stage 2: LLM
            transcript = stt_response['transcript']
            llm_request = TestRequest(
                request_id="pipeline_llm",
                module="llm",
                payload={
                    'prompt': f"<|audio|>\n{transcript}",
                    'max_tokens': 100,
                    'temperature': 0.7
                }
            )
            llm_response = await self.execute_llm_request(llm_request)

            if not llm_response['success']:
                return {
                    'success': False,
                    'error': 'LLM failed',
                    'stage': 'llm',
                    'transcript': transcript
                }

            # Stage 3: TTS
            response_text = llm_response['text']
            tts_request = TestRequest(
                request_id="pipeline_tts",
                module="tts",
                payload={
                    'text': response_text,
                    'voice_id': 'af_sky',
                    'speed': 1.0
                }
            )
            tts_response = await self.execute_tts_request(tts_request)

            if not tts_response['success']:
                return {
                    'success': False,
                    'error': 'TTS failed',
                    'stage': 'tts',
                    'transcript': transcript,
                    'response_text': response_text
                }

            # Success - return all stages
            total_latency = (
                stt_response['latency_ms'] +
                llm_response['latency_ms'] +
                tts_response['latency_ms']
            )

            return {
                'success': True,
                'transcript': transcript,
                'response_text': response_text,
                'response_audio': tts_response['audio'],
                'stt_latency_ms': stt_response['latency_ms'],
                'llm_latency_ms': llm_response['latency_ms'],
                'tts_latency_ms': tts_response['latency_ms'],
                'total_latency_ms': total_latency,
                'tokens_generated': llm_response['tokens_generated']
            }

        except Exception as e:
            logger.error("❌ Pipeline request failed", exception=e)
            return {
                'success': False,
                'error': str(e),
                'stage': 'unknown'
            }

    async def __call__(self, request: TestRequest) -> Dict[str, Any]:
        """
        Execute request based on module type

        This is the main entry point used by PerformanceTestRunner

        Args:
            request: Test request

        Returns:
            Service response
        """
        module = request.module.lower()

        if module == 'stt':
            return await self.execute_stt_request(request)
        elif module == 'llm':
            return await self.execute_llm_request(request)
        elif module == 'tts':
            return await self.execute_tts_request(request)
        elif module == 'pipeline':
            # Full pipeline test
            audio = request.payload.get('audio')
            sample_rate = request.payload.get('sample_rate', 16000)
            language = request.payload.get('language', 'pt')
            return await self.execute_pipeline_request(audio, sample_rate, language)
        else:
            raise ValueError(f"Unknown module: {module}")


# Singleton instance
_service_executor: Optional[ServiceExecutor] = None


def get_service_executor(config: Optional[ServiceExecutorConfig] = None) -> ServiceExecutor:
    """Get global service executor instance"""
    global _service_executor

    if _service_executor is None:
        _service_executor = ServiceExecutor(config)

    return _service_executor
