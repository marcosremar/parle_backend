"""
SimplePeer WebRTC Signal Handler
Handles signaling for SimplePeer connections
"""

import asyncio
import json
import logging
from typing import Dict, Optional
from datetime import datetime
import uuid

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
from aiortc.contrib.signaling import object_to_string, object_from_string

logger = logging.getLogger(__name__)


class SimplePeerHandler:
    """
    Handler for SimplePeer WebRTC connections
    Manages signaling between SimplePeer frontend and aiortc backend
    """

    def __init__(self, ice_servers=None):
        """Initialize SimplePeer handler"""
        if ice_servers is None:
            self.ice_servers = [
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"])
            ]
        else:
            self.ice_servers = [RTCIceServer(urls=server["urls"]) for server in ice_servers]

        # Active peer connections
        self.peers: Dict[str, RTCPeerConnection] = {}
        self.peer_websockets: Dict[str, any] = {}  # Map peer_id to websocket
        self.websocket_peers: Dict[any, str] = {}  # Map websocket to peer_id

        # Audio processor and TTS
        self.audio_processor = None
        self.tts_module = None

        logger.info("✅ SimplePeer handler initialized")

    def set_audio_processor(self, processor):
        """Set audio processor (Ultravox)"""
        self.audio_processor = processor

    def set_tts_module(self, tts):
        """Set TTS module"""
        self.tts_module = tts

    async def handle_simplepeer_signal(self, ws, data):
        """
        Handle SimplePeer signaling messages
        """
        try:
            signal_type = data.get('type')

            if signal_type == 'request_offer':
                # Client wants server to initiate connection
                await self.create_offer_for_client(ws)

            elif signal_type == 'webrtc_signal':
                # Handle SimplePeer signal
                signal = data.get('signal')
                if signal:
                    await self.process_simplepeer_signal(ws, signal)

            elif signal_type == 'offer':
                # SimplePeer offer received
                offer_data = data.get('offer')
                if offer_data:
                    await self.handle_offer(ws, offer_data)

            elif signal_type == 'answer':
                # SimplePeer answer received
                answer_data = data.get('answer')
                if answer_data:
                    await self.handle_answer(ws, answer_data)

            elif signal_type == 'ice_candidate':
                # ICE candidate received
                candidate = data.get('candidate')
                if candidate:
                    await self.handle_ice_candidate(ws, candidate)

        except Exception as e:
            logger.error(f"❌ Error handling SimplePeer signal: {e}")
            await ws.send_json({
                'type': 'error',
                'message': f'Signal error: {str(e)}'
            })

    async def create_offer_for_client(self, ws):
        """
        Server creates offer to initiate connection
        """
        try:
            # Generate unique peer ID
            peer_id = f"peer_{uuid.uuid4().hex[:8]}"

            # Create peer connection
            config = RTCConfiguration(iceServers=self.ice_servers)
            pc = RTCPeerConnection(configuration=config)

            # Store references
            self.peers[peer_id] = pc
            self.peer_websockets[peer_id] = ws
            self.websocket_peers[ws] = peer_id

            logger.info(f"🔌 Creating offer for peer: {peer_id}")

            # Create data channel
            channel = pc.createDataChannel("data", ordered=False, maxRetransmits=0)

            @channel.on("open")
            def on_open():
                logger.info(f"✅ DataChannel opened for {peer_id}")

            @channel.on("message")
            async def on_message(message):
                await self.handle_datachannel_message(peer_id, message, channel)

            # Set up connection state monitoring
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"📡 Connection state {peer_id}: {pc.connectionState}")
                if pc.connectionState in ["failed", "closed"]:
                    await self.cleanup_peer(peer_id)

            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            # Send offer to client
            await ws.send_json({
                'type': 'webrtc_signal',
                'signal': {
                    'type': 'offer',
                    'sdp': pc.localDescription.sdp
                }
            })

            logger.info(f"📤 Offer sent to {peer_id}")

        except Exception as e:
            logger.error(f"❌ Error creating offer: {e}")

    async def process_simplepeer_signal(self, ws, signal):
        """
        Process SimplePeer signal data
        """
        try:
            signal_type = signal.get('type')
            peer_id = self.websocket_peers.get(ws)

            if not peer_id:
                # Create new peer if doesn't exist
                peer_id = f"peer_{uuid.uuid4().hex[:8]}"
                config = RTCConfiguration(iceServers=self.ice_servers)
                pc = RTCPeerConnection(configuration=config)

                self.peers[peer_id] = pc
                self.peer_websockets[peer_id] = ws
                self.websocket_peers[ws] = peer_id

                # Set up data channel handler
                @pc.on("datachannel")
                def on_datachannel(channel: RTCDataChannel):
                    logger.info(f"✅ DataChannel received: {channel.label}")

                    @channel.on("message")
                    async def on_message(message):
                        await self.handle_datachannel_message(peer_id, message, channel)

                # Monitor connection state
                @pc.on("connectionstatechange")
                async def on_connectionstatechange():
                    logger.info(f"📡 Connection state {peer_id}: {pc.connectionState}")
                    if pc.connectionState in ["failed", "closed"]:
                        await self.cleanup_peer(peer_id)

            pc = self.peers[peer_id]

            if signal_type == 'offer':
                # Handle offer
                offer = RTCSessionDescription(sdp=signal['sdp'], type='offer')
                await pc.setRemoteDescription(offer)

                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                # Send answer back
                await ws.send_json({
                    'type': 'webrtc_signal',
                    'signal': {
                        'type': 'answer',
                        'sdp': pc.localDescription.sdp
                    }
                })

                logger.info(f"📤 Answer sent to {peer_id}")

            elif signal_type == 'answer':
                # Handle answer
                answer = RTCSessionDescription(sdp=signal['sdp'], type='answer')
                await pc.setRemoteDescription(answer)
                logger.info(f"✅ Answer processed for {peer_id}")

            elif signal_type == 'candidate':
                # Handle ICE candidate
                if signal.get('candidate'):
                    # SimplePeer sends full signal object
                    await pc.addIceCandidate(signal['candidate'])
                    logger.info(f"🧊 ICE candidate added for {peer_id}")

        except Exception as e:
            logger.error(f"❌ Error processing SimplePeer signal: {e}")

    async def handle_datachannel_message(self, peer_id, message, channel):
        """
        Handle messages from DataChannel
        """
        try:
            # Parse message
            if isinstance(message, str):
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'ping':
                    # Respond to ping
                    pong = json.dumps({
                        'type': 'pong',
                        'timestamp': data.get('timestamp')
                    })
                    channel.send(pong)

                elif msg_type == 'audio_chunk':
                    # Handle audio data
                    audio_data = data.get('audio', [])
                    await self.process_audio(peer_id, audio_data, channel)

                elif msg_type == 'recording_start':
                    logger.info(f"🎤 Recording started for {peer_id}")

                elif msg_type == 'recording_stop':
                    logger.info(f"⏹️ Recording stopped for {peer_id}")

            elif isinstance(message, bytes):
                # Binary audio data
                await self.process_audio_binary(peer_id, message, channel)

        except Exception as e:
            logger.error(f"❌ Error handling datachannel message: {e}")

    async def process_audio(self, peer_id, audio_data, channel):
        """
        Process audio data and send response
        """
        try:
            if self.audio_processor:
                # Process with Ultravox
                response_text = await self.audio_processor.process_audio(
                    audio_data,
                    peer_id
                )

                # Send transcription
                channel.send(json.dumps({
                    'type': 'transcription',
                    'text': response_text
                }))

                # Generate TTS if available
                if self.tts_module and response_text:
                    audio_response = await self.tts_module.synthesize(response_text)

                    # Send audio response
                    channel.send(json.dumps({
                        'type': 'response',
                        'text': response_text,
                        'audio': audio_response.hex() if isinstance(audio_response, bytes) else audio_response
                    }))

                logger.info(f"✅ Audio processed for {peer_id}")

        except Exception as e:
            logger.error(f"❌ Error processing audio: {e}")

    async def cleanup_peer(self, peer_id):
        """
        Clean up peer connection
        """
        try:
            if peer_id in self.peers:
                pc = self.peers[peer_id]
                await pc.close()
                del self.peers[peer_id]

            if peer_id in self.peer_websockets:
                ws = self.peer_websockets[peer_id]
                del self.peer_websockets[peer_id]
                if ws in self.websocket_peers:
                    del self.websocket_peers[ws]

            logger.info(f"🧹 Cleaned up peer: {peer_id}")

        except Exception as e:
            logger.error(f"❌ Error cleaning up peer: {e}")

    async def cleanup_websocket(self, ws):
        """
        Clean up when websocket disconnects
        """
        if ws in self.websocket_peers:
            peer_id = self.websocket_peers[ws]
            await self.cleanup_peer(peer_id)