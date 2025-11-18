#!/usr/bin/env python3
"""
ZeroMQ Coordination Server
Async server using ZeroMQ ROUTER socket with inproc:// transport

⚠️  SECURITY FIX: Migrated from pickle.loads() to json.loads() to prevent RCE
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
import zmq
import zmq.asyncio
import numpy as np

logger = logging.getLogger(__name__)


class ZeroMQCoordinationServer:
    """
    ZeroMQ server for handling service requests

    Architecture:
    - ROUTER socket (receives from multiple DEALER clients)
    - inproc:// transport (in-memory, ultra-fast)
    - Async request handling

    Performance: ~0.01ms latency per request
    """

    def __init__(self, service_name: str, service_handler,
                 context: Optional[zmq.asyncio.Context] = None):
        self.service_name = service_name
        self.service_handler = service_handler
        self.context = context or zmq.asyncio.Context()
        self.endpoint = f"inproc://{service_name}"
        self.socket = None
        self.running = False
        self.handler_task = None

    async def start(self):
        """Start ZeroMQ server"""
        try:
            # Create ROUTER socket
            self.socket = self.context.socket(zmq.ROUTER)

            # Set socket options
            self.socket.setsockopt(zmq.LINGER, 0)

            # Bind to inproc endpoint
            self.socket.bind(self.endpoint)

            self.running = True
            logger.info(f"✅ ZeroMQ server listening on {self.endpoint}")

            # Start async handler
            self.handler_task = asyncio.create_task(self._handle_requests())

        except Exception as e:
            logger.error(f"❌ Failed to start ZeroMQ server: {e}")
            raise

    async def _handle_requests(self):
        """Handle incoming requests (async loop)"""
        while self.running:
            try:
                # Receive request (ROUTER receives [identity, message])
                identity = await self.socket.recv()
                message = await self.socket.recv()

                # Process in background (don't block)
                asyncio.create_task(self._process_request(identity, message))

            except zmq.ZMQError as e:
                if not self.running:
                    break
                logger.error(f"❌ ZeroMQ error: {e}")
            except Exception as e:
                logger.error(f"❌ Request handling error: {e}")

    async def _process_request(self, identity: bytes, message: bytes):
        """Process single request"""
        try:
            # Deserialize request from JSON (safe from RCE)
            request = json.loads(message.decode('utf-8'))

            endpoint = request['endpoint']
            metadata = request.get('metadata', {})
            payload = request['payload']

            logger.debug(f"📥 Received request: {endpoint}")

            # Call service handler
            result = await self._call_handler(endpoint, payload, metadata)

            # Serialize response to JSON (safe from RCE)
            try:
                response_bytes = json.dumps(result).encode('utf-8')
            except TypeError as e:
                # If result contains non-JSON-serializable objects (like NumPy arrays),
                # convert them to lists/dicts
                logger.warning(f"Result contains non-JSON objects: {e}. Attempting conversion...")
                def convert_to_json_safe(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: convert_to_json_safe(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_json_safe(item) for item in obj]
                    return obj

                converted_result = convert_to_json_safe(result)
                response_bytes = json.dumps(converted_result).encode('utf-8')

            # Send response (ROUTER sends [identity, message])
            await self.socket.send(identity, zmq.SNDMORE)
            await self.socket.send(response_bytes)

            logger.debug(f"📤 Sent response: {endpoint}")

        except Exception as e:
            logger.error(f"❌ Request processing failed: {e}")

            # Send error response
            try:
                error_response = {'error': str(e)}
                response_bytes = pickle.dumps(error_response)

                await self.socket.send(identity, zmq.SNDMORE)
                await self.socket.send(response_bytes)
            except (OSError, zmq.ZMQError) as send_err:
                logger.warning(f"Failed to send error response: {send_err}")
            except Exception as send_err:
                logger.error(f"Unexpected error sending error response: {send_err}")

    async def _call_handler(self, endpoint: str, payload: Any, metadata: Dict[str, Any]):
        """Call service handler with payload"""
        # Find handler function
        handler = self._find_handler(endpoint)

        if not handler:
            raise Exception(f"Handler not found: {endpoint}")

        # Prepare arguments for handler
        if isinstance(payload, dict) and 'audio_data' in payload:
            audio_data = payload['audio_data']

            # Convert NumPy to bytes if needed
            if isinstance(audio_data, np.ndarray):
                audio_bytes = audio_data.tobytes()
            else:
                audio_bytes = audio_data

            handler_args = {
                'audio_data': audio_bytes,
                'sample_rate': metadata.get('sample_rate', 16000)
            }
            handler_args.update(metadata)
        else:
            handler_args = payload if isinstance(payload, dict) else {'data': payload}

        # Call handler
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**handler_args)
            else:
                result = handler(**handler_args)

            # Ensure result is dict
            if not isinstance(result, dict):
                result = {'result': result}

            return result

        except Exception as e:
            logger.error(f"❌ Handler call failed: {e}")
            raise

    def _find_handler(self, endpoint: str):
        """Find handler function for endpoint"""
        try:
            router = self.service_handler.get_router()

            # Normalize endpoint
            endpoint_variants = [endpoint]
            if endpoint.startswith('/'):
                endpoint_variants.append(endpoint[1:])
            else:
                endpoint_variants.append(f"/{endpoint}")

            for route in router.routes:
                if hasattr(route, 'path') and route.path in endpoint_variants:
                    return route.endpoint

        except Exception as e:
            logger.error(f"❌ Failed to find handler: {e}")

        return None

    async def stop(self):
        """Stop ZeroMQ server"""
        self.running = False

        if self.handler_task:
            self.handler_task.cancel()
            try:
                await self.handler_task
            except asyncio.CancelledError:
                pass

        if self.socket:
            try:
                self.socket.close()
                logger.info("🛑 ZeroMQ server stopped")
            except zmq.ZMQError as e:
                logger.warning(f"ZMQ error closing socket: {e}")
            except Exception as e:
                logger.error(f"Unexpected error closing ZeroMQ socket: {e}")
