"""
Generic gRPC Server Infrastructure
Automatically creates gRPC servers for any BaseService
"""

from .generic_servicer import GenericGrpcServicer
from .server import start_generic_grpc_server

__all__ = ['GenericGrpcServicer', 'start_generic_grpc_server']
