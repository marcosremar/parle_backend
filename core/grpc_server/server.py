#!/usr/bin/env python3
"""
Generic gRPC Server
Automatically starts gRPC server for any BaseService
"""

# Import grpc library (not local package) - use absolute import
import grpc as grpc_lib
from grpc import aio as grpc_aio
import asyncio
import logging
from concurrent import futures
from typing import Optional

from .generic_servicer import GenericGrpcServicer

logger = logging.getLogger(__name__)

# Global gRPC server instance per service
_grpc_servers = {}


async def start_generic_grpc_server(
    service,
    port: int,
    service_name: str = None,
    max_workers: int = 10
) -> Optional[grpc_aio.Server]:
    """
    Start generic gRPC server for a BaseService

    This function:
    1. Creates a GenericGrpcServicer for the service
    2. Starts a gRPC server on the specified port
    3. Registers the UltravoxService.Call RPC
    4. Returns the server instance

    Args:
        service: BaseService instance
        port: gRPC port to listen on
        service_name: Service name for logging (optional)
        max_workers: Max concurrent RPC workers

    Returns:
        gRPC server instance or None if failed

    Example:
        # In BaseService._start_grpc_server():
        grpc_port = self._get_grpc_port()  # e.g., 50099
        await start_generic_grpc_server(self, grpc_port, "stt")
    """

    if service_name is None:
        service_name = service.__class__.__name__

    try:
        logger.info(f"🚀 Starting generic gRPC server for {service_name} on port {port}")

        # Create generic servicer
        servicer = GenericGrpcServicer(service, service_name)

        # Create gRPC server
        server = grpc_aio.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
            ]
        )

        # Register servicer with gRPC server
        from protos.generated import ultravox_services_pb2_grpc
        ultravox_services_pb2_grpc.add_UltravoxServiceServicer_to_server(
            servicer, server
        )

        # Add insecure port
        server.add_insecure_port(f'[::]:{port}')

        # Start server
        await server.start()

        logger.info(f"✅ gRPC server for {service_name} listening on port {port}")
        logger.info(f"   Endpoints: {list(servicer.endpoint_map.keys())}")

        # Store server instance for cleanup
        _grpc_servers[service_name] = server

        # Don't await termination - let it run in background
        # The server will be stopped when stop_generic_grpc_server() is called

        return server

    except Exception as e:
        logger.error(f"❌ Failed to start gRPC server for {service_name}: {e}")
        return None


async def stop_generic_grpc_server(service_name: str, grace_period: float = 5.0):
    """
    Stop gRPC server for a service

    Args:
        service_name: Name of the service
        grace_period: Grace period in seconds for graceful shutdown
    """

    server = _grpc_servers.get(service_name)

    if server:
        try:
            logger.info(f"🛑 Stopping gRPC server for {service_name}")
            await server.stop(grace_period)
            del _grpc_servers[service_name]
            logger.info(f"✅ gRPC server for {service_name} stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping gRPC server for {service_name}: {e}")
    else:
        logger.warning(f"⚠️ No gRPC server found for {service_name}")


def get_grpc_port_from_http_port(http_port: int) -> int:
    """
    Calculate gRPC port from HTTP port

    Formula: grpc_port = 50000 + (http_port % 1000)

    Examples:
        8099 (STT) -> 50099
        8100 (LLM) -> 50100
        8101 (TTS) -> 50101
        8020 (API Gateway) -> 50020
        8900 (Orchestrator) -> 50900

    Args:
        http_port: HTTP port number

    Returns:
        gRPC port number
    """

    # Extract last 3 digits of HTTP port
    last_digits = http_port % 1000

    # gRPC ports are 50xxx
    grpc_port = 50000 + last_digits

    return grpc_port


def get_all_grpc_servers():
    """Get all running gRPC servers"""
    return _grpc_servers.copy()


async def stop_all_grpc_servers(grace_period: float = 5.0):
    """Stop all gRPC servers"""
    logger.info("🛑 Stopping all gRPC servers...")

    tasks = []
    for service_name in list(_grpc_servers.keys()):
        tasks.append(stop_generic_grpc_server(service_name, grace_period))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("✅ All gRPC servers stopped")
