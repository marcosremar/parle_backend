"""
Protocol Adapters for Communication Service

Priority (with automatic fallback):
1. ZeroMQ - Ultra-fast IPC (~0.01ms, 410k msg/s)
2. gRPC - Fast binary protocol (~7ms)
3. HTTP Binary - msgpack encoding
4. HTTP JSON - Last resort fallback
"""

from .base_adapter import BaseProtocolAdapter
from .binary_adapter import BinaryProtocolAdapter
from .json_adapter import JsonProtocolAdapter
from .grpc_adapter import GrpcProtocolAdapter
from .direct_call_adapter import DirectCallProtocolAdapter

# Aliases for backwards compatibility
GRPCProtocolAdapter = GrpcProtocolAdapter
DirectCallAdapter = DirectCallProtocolAdapter

# ZeroMQ adapter (optional - requires pyzmq)
try:
    from .zeromq_adapter import ZeroMQProtocolAdapter
    ZEROMQ_AVAILABLE = True
except ImportError:
    ZEROMQ_AVAILABLE = False
    ZeroMQProtocolAdapter = None

__all__ = [
    'BaseProtocolAdapter',
    'ZeroMQProtocolAdapter',
    'GRPCProtocolAdapter',
    'BinaryProtocolAdapter',
    'JsonProtocolAdapter',
    'DirectCallAdapter',
    'ZEROMQ_AVAILABLE'
]
