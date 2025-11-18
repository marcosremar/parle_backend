#!/usr/bin/env python3
"""
Generic gRPC Servicer
Routes gRPC calls to FastAPI endpoints automatically
Works with ANY service without specific implementation!
"""

# Import grpc library (not local package) - use absolute import
import grpc as grpc_lib
import json
import logging
from typing import Any, Dict
from fastapi import Request
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)

# Note: Import proto generated files after compilation
# from protos.generated import ultravox_services_pb2, ultravox_services_pb2_grpc


class GenericGrpcServicer:
    """
    Generic gRPC servicer that routes to FastAPI endpoints

    Features:
    - Works with ANY BaseService
    - No service-specific code needed
    - Automatically adapts to endpoint changes
    - Reuses FastAPI validation and logic

    How it works:
    1. Client calls: UltravoxService.Call(ServiceRequest)
    2. Extract endpoint path and payload from request
    3. Find matching route in service.router
    4. Call the FastAPI handler
    5. Return ServiceResponse with result
    """

    def __init__(self, service, service_name: str):
        """
        Initialize generic servicer

        Args:
            service: BaseService instance (has .router with FastAPI routes)
            service_name: Service name for logging
        """
        self.service = service
        self.service_name = service_name
        self.router = service.get_router()

        # Build endpoint map: {"/path": handler_function}
        self.endpoint_map = self._build_endpoint_map()

        logger.info(f"🚀 Generic gRPC Servicer initialized for {service_name}")
        logger.info(f"   Available endpoints: {list(self.endpoint_map.keys())}")

    def _build_endpoint_map(self) -> Dict[str, Any]:
        """Build map of endpoints to handler functions"""
        endpoint_map = {}

        for route in self.router.routes:
            if isinstance(route, APIRoute):
                # FastAPI routes have .path and .endpoint
                path = route.path
                handler = route.endpoint
                endpoint_map[path] = handler

                # Also add without leading slash
                if path.startswith('/'):
                    endpoint_map[path[1:]] = handler

        return endpoint_map

    async def Call(self, request, context):
        """
        Handle generic gRPC call

        This is the UltravoxService.Call RPC implementation
        Routes to appropriate FastAPI endpoint based on ServiceRequest.endpoint

        Args:
            request: ServiceRequest proto message
            context: gRPC context

        Returns:
            ServiceResponse proto message
        """
        try:
            from protos.generated import ultravox_services_pb2

            # 1. Parse request
            endpoint = request.endpoint
            payload_json = request.payload.decode('utf-8')
            payload = json.loads(payload_json) if payload_json else {}

            logger.debug(f"📥 gRPC Call: {endpoint} with payload: {payload}")

            # 2. Find handler
            handler = self.endpoint_map.get(endpoint)
            if not handler:
                context.set_code(grpc_lib.StatusCode.NOT_FOUND)
                context.set_details(f"Endpoint {endpoint} not found")
                raise Exception(f"Endpoint {endpoint} not found")

            # 3. Call handler
            # Note: FastAPI handlers expect Request object
            # We simulate it with the payload
            result = await handler(**payload)

            # 4. Build response
            response_json = json.dumps(result)
            response_bytes = response_json.encode('utf-8')

            return ultravox_services_pb2.ServiceResponse(
                status_code=200,
                payload=response_bytes,
                metadata={}
            )

        except grpc_lib.RpcError:
            # Re-raise gRPC errors
            raise

        except Exception as e:
            logger.error(f"❌ Error in generic gRPC call: {e}")
            context.set_code(grpc_lib.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise


# Specific servicers (optional - for type-specific optimization)
# These inherit from the specific proto servicers after compilation

class STTGrpcServicer:
    """
    STT-specific gRPC servicer (optional optimization)
    Falls back to generic if not needed
    """

    def __init__(self, service):
        self.service = service

    async def Transcribe(self, request, context):
        """Handle STT transcription via gRPC"""
        # TODO: After proto compilation
        logger.warning("⚠️ STT gRPC servicer not yet compiled")
        raise NotImplementedError("Proto stubs need to be compiled")


class LLMGrpcServicer:
    """LLM-specific gRPC servicer (optional optimization)"""

    def __init__(self, service):
        self.service = service

    async def Generate(self, request, context):
        """Handle LLM generation via gRPC"""
        logger.warning("⚠️ LLM gRPC servicer not yet compiled")
        raise NotImplementedError("Proto stubs need to be compiled")


class TTSGrpcServicer:
    """TTS-specific gRPC servicer (optional optimization)"""

    def __init__(self, service):
        self.service = service

    async def Synthesize(self, request, context):
        """Handle TTS synthesis via gRPC"""
        logger.warning("⚠️ TTS gRPC servicer not yet compiled")
        raise NotImplementedError("Proto stubs need to be compiled")
