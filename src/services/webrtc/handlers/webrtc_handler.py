"""
WebRTC Handler Module
Manages WebRTC connections and data channels for real-time audio streaming
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Callable
import numpy as np

from src.core.interfaces import IWebRTCHandler, IAudioProcessor, ITextToSpeech
from src.core.config import WebRTCConfig

logger = logging.getLogger(__name__)

# Try to import aiortc
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCIceCandidate
    from aiortc.contrib.media import MediaPlayer, MediaRecorder
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logger.warning("aiortc not available. WebRTC functionality will be limited.")


class WebRTCConnection:
    """Represents a single WebRTC connection"""
    
    def __init__(self, session_id: str, pc: 'RTCPeerConnection'):
        self.session_id = session_id
        self.pc = pc
        self.data_channel = None
        self.created_at = time.time()
        self.last_activity = time.time()
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'audio_chunks_processed': 0
        }
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        
    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if connection is expired"""
        return (time.time() - self.last_activity) > timeout_seconds


class WebRTCHandler(IWebRTCHandler):
    """
    WebRTC handler implementation using aiortc
    Manages peer connections and audio streaming
    """
    
    def __init__(self,
                 config: WebRTCConfig,
                 audio_processor: IAudioProcessor,
                 tts: ITextToSpeech):
        """
        Initialize WebRTC handler
        
        Args:
            config: WebRTC configuration
            audio_processor: Audio processor (Ultravox)
            tts: Text-to-speech module
        """
        self.config = config
        self.audio_processor = audio_processor
        self.tts = tts
        
        # Active connections
        self.connections: Dict[str, WebRTCConnection] = {}
        self._lock = asyncio.Lock()
        
        # Stats
        self.total_connections = 0
        self.total_messages = 0
        
        # Cleanup task
        self._cleanup_task = None
        
    async def initialize(self) -> None:
        """Initialize WebRTC handler"""
        logger.info("🌐 Initializing WebRTC handler...")
        
        if not AIORTC_AVAILABLE:
            logger.error("❌ aiortc not available! Install with: pip install aiortc")
            raise RuntimeError("aiortc is required for WebRTC support")
        
        logger.info(f"   Max connections: {self.config.max_connections}")
        logger.info(f"   Audio codec: {self.config.audio_codec}")
        logger.info(f"   STUN servers: {len(self.config.stun_servers)}")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("✅ WebRTC handler initialized")
        
    async def handle_offer(self,
                          offer: Dict[str, Any],
                          session_id: str) -> Dict[str, Any]:
        """
        Handle WebRTC offer and return answer
        
        Args:
            offer: WebRTC offer from client
            session_id: Unique session identifier
            
        Returns:
            WebRTC answer to send back to client
        """
        async with self._lock:
            # Check connection limit
            if len(self.connections) >= self.config.max_connections:
                # Remove oldest connection
                oldest_id = min(
                    self.connections.keys(),
                    key=lambda k: self.connections[k].created_at
                )
                await self._close_connection(oldest_id)
                logger.info(f"Evicted oldest connection {oldest_id[:8]} to make room")
            
            # Create peer connection
            pc = RTCPeerConnection(configuration={
                "iceServers": [
                    {"urls": self.config.stun_servers}
                ]
            })
            
            # Store connection
            connection = WebRTCConnection(session_id, pc)
            self.connections[session_id] = connection
            self.total_connections += 1
            
            logger.info(f"🔌 Creating WebRTC connection for session {session_id[:8]}")
            
            # Set up data channel handler
            @pc.on("datachannel")
            def on_datachannel(channel: RTCDataChannel):
                connection.data_channel = channel
                logger.info(f"📡 Data channel opened for session {session_id[:8]}")
                
                @channel.on("message")
                async def on_message(message):
                    await self._handle_data_channel_message(
                        message, session_id, channel
                    )
            
            # Set up connection state handlers
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"Connection state for {session_id[:8]}: {pc.connectionState}")
                
                if pc.connectionState == "failed":
                    await self._close_connection(session_id)
                elif pc.connectionState == "closed":
                    await self._close_connection(session_id)
            
            # Set remote description (offer)
            offer_sdp = RTCSessionDescription(
                sdp=offer["sdp"],
                type=offer["type"]
            )
            await pc.setRemoteDescription(offer_sdp)
            
            # Create and set local description (answer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            logger.info(f"✅ WebRTC answer created for session {session_id[:8]}")
            
            return {
                "type": answer.type,
                "sdp": answer.sdp
            }
    
    async def _handle_data_channel_message(self,
                                          message: Any,
                                          session_id: str,
                                          channel: RTCDataChannel) -> None:
        """
        Handle message from data channel
        
        Args:
            message: Message from client
            session_id: Session identifier
            channel: Data channel to send response
        """
        try:
            connection = self.connections.get(session_id)
            if not connection:
                logger.warning(f"No connection found for session {session_id[:8]}")
                return
            
            connection.update_activity()
            connection.stats['messages_received'] += 1
            
            # Parse message
            if isinstance(message, str):
                data = json.loads(message)
            else:
                # Binary data (audio)
                data = {"type": "audio", "data": message}
            
            # Handle different message types
            if data.get("type") == "audio":
                await self._handle_audio_message(data, session_id, channel)
            elif data.get("type") == "ping":
                await self._send_message(channel, {"type": "pong"})
            elif data.get("type") == "config":
                await self._handle_config_message(data, session_id)
            else:
                logger.warning(f"Unknown message type: {data.get('type')}")
                
        except Exception as e:
            logger.error(f"Error handling message for {session_id[:8]}: {e}")
            await self._send_error(channel, str(e))
    
    async def _handle_audio_message(self,
                                   data: Dict[str, Any],
                                   session_id: str,
                                   channel: RTCDataChannel) -> None:
        """
        Handle audio message from client
        
        Args:
            data: Audio data message
            session_id: Session identifier
            channel: Data channel for response
        """
        try:
            connection = self.connections[session_id]
            connection.stats['audio_chunks_processed'] += 1
            
            # Extract audio data
            if isinstance(data.get("data"), bytes):
                audio_bytes = data["data"]
            elif isinstance(data.get("audio"), list):
                audio_bytes = bytes(data["audio"])
            else:
                raise ValueError("Invalid audio data format")
            
            # Get sample rate
            sample_rate = data.get("sampleRate", self.config.audio_sample_rate)
            
            # Convert to numpy array
            if self.config.audio_codec == "opus":
                # Decode Opus if needed
                audio_array = await self._decode_opus(audio_bytes, sample_rate)
            else:
                # PCM float32
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            logger.info(f"🎤 Processing {len(audio_array)} audio samples for {session_id[:8]}")
            
            # Process with Ultravox
            start_time = time.time()
            
            text_response = await self.audio_processor.process(
                audio=audio_array,
                sample_rate=sample_rate,
                session_id=session_id,
                context=data.get("context", {})
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"💬 Response: '{text_response[:50]}...' ({processing_time:.0f}ms)")
            
            # Synthesize TTS response
            audio_response = await self.tts.synthesize(text_response)
            
            # Send response
            response = {
                "type": "response",
                "text": text_response,
                "audio": list(audio_response) if isinstance(audio_response, bytes) else audio_response.tolist(),
                "processingTime": processing_time,
                "timestamp": time.time()
            }
            
            await self._send_message(channel, response)
            
            connection.stats['bytes_sent'] += len(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Error processing audio for {session_id[:8]}: {e}")
            await self._send_error(channel, f"Audio processing failed: {str(e)}")
    
    async def _decode_opus(self, 
                          opus_data: bytes, 
                          sample_rate: int) -> np.ndarray:
        """
        Decode Opus audio to PCM
        
        Args:
            opus_data: Opus encoded audio
            sample_rate: Sample rate
            
        Returns:
            PCM audio as numpy array
        """
        # This would require an Opus decoder
        # For now, assume PCM input
        logger.warning("Opus decoding not implemented, treating as PCM")
        return np.frombuffer(opus_data, dtype=np.float32)
    
    async def _handle_config_message(self,
                                    data: Dict[str, Any],
                                    session_id: str) -> None:
        """
        Handle configuration message from client
        
        Args:
            data: Configuration data
            session_id: Session identifier
        """
        logger.info(f"Configuration update for {session_id[:8]}: {data}")
        # Could update per-session configuration here
    
    async def _send_message(self, 
                           channel: RTCDataChannel,
                           message: Dict[str, Any]) -> None:
        """
        Send message through data channel
        
        Args:
            channel: Data channel
            message: Message to send
        """
        try:
            if channel and channel.readyState == "open":
                channel.send(json.dumps(message))
                self.total_messages += 1
            else:
                logger.warning("Data channel not open, cannot send message")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def _send_error(self,
                         channel: RTCDataChannel,
                         error_message: str) -> None:
        """
        Send error message to client
        
        Args:
            channel: Data channel
            error_message: Error description
        """
        await self._send_message(channel, {
            "type": "error",
            "error": error_message,
            "timestamp": time.time()
        })
    
    async def handle_ice_candidate(self,
                                  candidate: Dict[str, Any],
                                  session_id: str) -> None:
        """
        Handle ICE candidate from client
        
        Args:
            candidate: ICE candidate information
            session_id: Session identifier
        """
        async with self._lock:
            connection = self.connections.get(session_id)
            if not connection:
                logger.warning(f"No connection for ICE candidate: {session_id[:8]}")
                return
            
            try:
                ice_candidate = RTCIceCandidate(
                    candidate=candidate.get("candidate"),
                    sdpMLineIndex=candidate.get("sdpMLineIndex"),
                    sdpMid=candidate.get("sdpMid")
                )
                
                await connection.pc.addIceCandidate(ice_candidate)
                logger.debug(f"Added ICE candidate for {session_id[:8]}")
                
            except Exception as e:
                logger.error(f"Error adding ICE candidate for {session_id[:8]}: {e}")
    
    async def close_connection(self, session_id: str) -> None:
        """
        Close WebRTC connection for a session
        
        Args:
            session_id: Session identifier
        """
        async with self._lock:
            await self._close_connection(session_id)
    
    async def _close_connection(self, session_id: str) -> None:
        """
        Internal method to close connection
        
        Args:
            session_id: Session identifier
        """
        connection = self.connections.get(session_id)
        if not connection:
            return
        
        try:
            # Close peer connection
            await connection.pc.close()
            
            # Remove from connections
            del self.connections[session_id]
            
            logger.info(f"🔌 Closed connection for session {session_id[:8]}")
            logger.info(f"   Stats: {connection.stats}")
            
        except Exception as e:
            logger.error(f"Error closing connection for {session_id[:8]}: {e}")
    
    async def get_connection_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get connection statistics
        
        Args:
            session_id: Session identifier
            
        Returns:
            Connection statistics or None if not connected
        """
        async with self._lock:
            connection = self.connections.get(session_id)
            if not connection:
                return None
            
            # Get WebRTC stats
            stats = await connection.pc.getStats()
            
            return {
                "session_id": session_id,
                "created_at": connection.created_at,
                "last_activity": connection.last_activity,
                "connection_state": connection.pc.connectionState,
                "ice_connection_state": connection.pc.iceConnectionState,
                "custom_stats": connection.stats,
                "webrtc_stats": stats  # Raw WebRTC stats
            }
    
    async def _cleanup_loop(self) -> None:
        """Cleanup expired connections periodically"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                async with self._lock:
                    expired = [
                        sid for sid, conn in self.connections.items()
                        if conn.is_expired(self.config.connection_timeout_seconds)
                    ]
                    
                    for sid in expired:
                        await self._close_connection(sid)
                        logger.info(f"Cleaned up expired connection: {sid[:8]}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup all connections"""
        logger.info("🧹 Cleaning up WebRTC handler...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._lock:
            for session_id in list(self.connections.keys()):
                await self._close_connection(session_id)
        
        logger.info(f"📊 WebRTC Stats:")
        logger.info(f"   Total connections: {self.total_connections}")
        logger.info(f"   Total messages: {self.total_messages}")
        
        logger.info("✅ WebRTC handler cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            "active_connections": len(self.connections),
            "total_connections": self.total_connections,
            "total_messages": self.total_messages,
            "max_connections": self.config.max_connections,
            "audio_codec": self.config.audio_codec
        }