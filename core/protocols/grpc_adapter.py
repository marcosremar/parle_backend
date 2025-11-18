#!/usr/bin/env python3
"""
gRPC Protocol Adapter
Ultra-low latency service communication using gRPC
5-10x faster than HTTP POST
"""

import logging
from typing import Dict, Any, Tuple, Optional
from concurrent import futures
import asyncio

# Import grpc BEFORE any local imports to avoid circular import
import grpc as grpc_lib
import grpc.aio as grpc_aio

from src.core.protocols.base_adapter import BaseProtocolAdapter

logger = logging.getLogger(__name__)

# Proto imports
from protos.generated import ultravox_services_pb2, ultravox_services_pb2_grpc


class GrpcProtocolAdapter(BaseProtocolAdapter):
    """gRPC protocol adapter for ultra-low latency communication"""

    def __init__(self):
        self.name_value = "grpc"
        self.channels = {}  # Service name -> gRPC channel (connection pool)
        self.stubs = {}     # Service name -> gRPC stub

        # gRPC configuration
        self.grpc_options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
        ]

        logger.info("🚀 gRPC Protocol Adapter initialized")

    @property
    def name(self) -> str:
        return self.name_value

    @property
    def content_type(self) -> str:
        return "application/grpc"

    def _get_channel(self, service_name: str, host: str, port: int) -> grpc_aio.Channel:
        """Get or create gRPC channel for service (connection pooling)"""
        channel_key = f"{service_name}:{host}:{port}"

        if channel_key not in self.channels:
            target = f"{host}:{port}"
            self.channels[channel_key] = grpc_aio.insecure_channel(
                target,
                options=self.grpc_options
            )
            logger.info(f"✅ Created gRPC channel for {service_name} at {target}")

        return self.channels[channel_key]

    async def close_all_channels(self):
        """Close all gRPC channels (cleanup)"""
        for channel in self.channels.values():
            await channel.close()
        self.channels.clear()
        self.stubs.clear()
        logger.info("🔒 All gRPC channels closed")

    def encode_audio(self, audio_data: bytes, sample_rate: int,
                    metadata: Dict[str, Any] = None) -> Tuple[bytes, Dict[str, str]]:
        """
        Encode audio for gRPC transmission

        Note: For gRPC, we don't actually encode to bytes here.
        This method exists for interface compatibility.
        The actual transmission uses proto messages.
        """
        # Return audio_data as-is for gRPC - will be wrapped in proto message
        headers = {
            'Content-Type': self.content_type,
            'sample_rate': str(sample_rate)
        }

        return audio_data, headers

    def decode_audio_response(self, response_data: bytes,
                             content_type: str) -> Dict[str, Any]:
        """Decode gRPC audio response"""
        # For gRPC, response_data is already a proto message
        # This is a placeholder for interface compatibility
        return {
            'audio_data': response_data,
            'metadata': {}
        }

    def encode_text(self, text: str, metadata: Dict[str, Any] = None) -> Tuple[bytes, Dict[str, str]]:
        """Encode text for gRPC transmission"""
        # For gRPC, text is wrapped in proto message, not encoded to bytes
        headers = {
            'Content-Type': self.content_type
        }

        return text.encode('utf-8'), headers

    def decode_text_response(self, response_data: bytes,
                            content_type: str) -> Dict[str, Any]:
        """Decode gRPC text response"""
        return {
            'text': response_data.decode('utf-8'),
            'metadata': {}
        }

    def estimate_size(self, audio_data: bytes = None, text: str = None) -> int:
        """Estimate gRPC payload size (protobuf is very efficient)"""
        if audio_data:
            # Proto overhead is minimal (~10-20 bytes)
            return len(audio_data) + 20
        elif text:
            return len(text.encode('utf-8')) + 20
        return 0

    async def call_stt_service(self, audio_data: bytes, sample_rate: int,
                               language: str, service_host: str, service_port: int,
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call STT service via gRPC

        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            language: Language code (e.g., 'pt-BR')
            service_host: STT service hostname
            service_port: STT service port (gRPC port, e.g., 50099)
            metadata: Additional metadata

        Returns:
            Transcription result
        """
        try:
            # Get channel
            channel = self._get_channel("stt", service_host, service_port)

            # Create gRPC stub
            stub = ultravox_services_pb2_grpc.STTServiceStub(channel)

            # Build metadata
            proto_metadata = ultravox_services_pb2.Metadata()
            if metadata:
                for key, value in metadata.items():
                    proto_metadata.fields[key] = str(value)

            # Create request
            request = ultravox_services_pb2.TranscribeRequest(
                audio_data=audio_data,
                sample_rate=sample_rate,
                language=language,
                metadata=proto_metadata
            )

            # Call service
            response = await stub.Transcribe(request, timeout=30.0)

            # Convert response to dict
            result = {
                'text': response.text,
                'confidence': response.confidence,
                'metadata': dict(response.metadata.fields) if response.metadata else {}
            }

            logger.debug(f"✅ gRPC STT call succeeded: {len(result['text'])} chars")
            return result

        except grpc_lib.RpcError as e:
            logger.error(f"gRPC error calling STT: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Error calling STT via gRPC: {e}")
            raise

    async def call_llm_service(self, prompt: str, system_prompt: str,
                              temperature: float, max_tokens: int,
                              service_host: str, service_port: int,
                              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call LLM service via gRPC

        Returns:
            Generated text response
        """
        try:
            channel = self._get_channel("llm", service_host, service_port)

            # Create gRPC stub
            stub = ultravox_services_pb2_grpc.LLMServiceStub(channel)

            # Build metadata
            proto_metadata = ultravox_services_pb2.Metadata()
            if metadata:
                for key, value in metadata.items():
                    proto_metadata.fields[key] = str(value)

            # Create request
            request = ultravox_services_pb2.GenerateRequest(
                prompt=prompt,
                system_prompt=system_prompt or "",
                temperature=temperature,
                max_tokens=max_tokens,
                metadata=proto_metadata
            )

            # Call service
            response = await stub.Generate(request, timeout=60.0)

            # Convert response to dict
            result = {
                'text': response.text,
                'tokens_used': response.tokens_used,
                'metadata': dict(response.metadata.fields) if response.metadata else {}
            }

            logger.debug(f"✅ gRPC LLM call succeeded: {response.tokens_used} tokens")
            return result

        except grpc_lib.RpcError as e:
            logger.error(f"gRPC error calling LLM: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Error calling LLM via gRPC: {e}")
            raise

    async def call_tts_service(self, text: str, voice_id: str, speed: float,
                              service_host: str, service_port: int,
                              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call TTS service via gRPC

        Returns:
            Synthesized audio data
        """
        try:
            channel = self._get_channel("tts", service_host, service_port)

            # Create gRPC stub
            stub = ultravox_services_pb2_grpc.TTSServiceStub(channel)

            # Build metadata
            proto_metadata = ultravox_services_pb2.Metadata()
            if metadata:
                for key, value in metadata.items():
                    proto_metadata.fields[key] = str(value)

            # Create request
            request = ultravox_services_pb2.SynthesizeRequest(
                text=text,
                voice_id=voice_id or "",
                speed=speed,
                metadata=proto_metadata
            )

            # Call service
            response = await stub.Synthesize(request, timeout=30.0)

            # Convert response to dict
            result = {
                'audio_data': response.audio_data,
                'sample_rate': response.sample_rate,
                'audio_length_ms': response.audio_length_ms,
                'metadata': dict(response.metadata.fields) if response.metadata else {}
            }

            logger.debug(f"✅ gRPC TTS call succeeded: {response.audio_length_ms}ms audio")
            return result

        except grpc_lib.RpcError as e:
            logger.error(f"gRPC error calling TTS: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Error calling TTS via gRPC: {e}")
            raise

    async def call_generic_service(self, service_name: str, endpoint: str,
                                  payload: bytes, service_host: str, service_port: int,
                                  metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generic service call via gRPC (for services not yet with specific stubs)

        Uses the generic UltravoxService.Call RPC
        """
        try:
            channel = self._get_channel(service_name, service_host, service_port)

            # Create gRPC stub
            stub = ultravox_services_pb2_grpc.UltravoxServiceStub(channel)

            # Build metadata
            proto_metadata = ultravox_services_pb2.Metadata()
            if metadata:
                for key, value in metadata.items():
                    proto_metadata.fields[key] = str(value)

            # Create request
            request = ultravox_services_pb2.ServiceRequest(
                service_name=service_name,
                endpoint=endpoint,
                payload=payload,
                metadata=proto_metadata
            )

            # Call service
            response = await stub.Call(request, timeout=30.0)

            # Convert response to dict
            result = {
                'status_code': response.status_code,
                'payload': response.payload,
                'metadata': dict(response.metadata.fields) if response.metadata else {}
            }

            logger.debug(f"✅ gRPC generic call to {service_name}.{endpoint} succeeded")
            return result

        except grpc_lib.RpcError as e:
            logger.error(f"gRPC error calling {service_name}: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Error calling {service_name} via gRPC: {e}")
            raise
