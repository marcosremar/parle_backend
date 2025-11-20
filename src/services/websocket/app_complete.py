"""
WebSocket Service Standalone - Consolidated for Nomad deployment
Socket.IO server for real-time communication with barge-in support
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json
import asyncio
import socketio
import logging
from loguru import logger

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import local utils (fallback implementations if not available)
try:
    from .utils.route_helpers import add_standard_endpoints
    from .utils.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_standard_endpoints(router, service_instance=None, service_name=None):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "websocket",
        "port": 8022,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "websocket": {
        "ping_timeout": 60,
        "ping_interval": 25,
        "max_connections": 1000,
        "cors_origins": ["*"],
        "heartbeat_interval": 30,
        "connection_timeout": 3600  # 1 hour
    }
}

def get_config():
    """Get websocket service configuration"""
    config = DEFAULT_CONFIG.copy()
    return config

# ============================================================================
# Pydantic Models (Standalone)
# ============================================================================

class ConnectionStats(BaseModel):
    """Connection statistics"""
    total_connections: int
    active_connections: int
    total_messages: int
    uptime_seconds: int
    timestamp: datetime

class RoomInfo(BaseModel):
    """Room information"""
    room_id: str
    participants: List[str]
    created_at: datetime
    message_count: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    websocket_enabled: bool
    connections: Dict[str, int]
    timestamp: datetime

class MessageData(BaseModel):
    """Message data model"""
    type: str
    content: Any
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BargeInRequest(BaseModel):
    """Barge-in request model"""
    conversation_id: str
    user_message: str
    priority: Optional[int] = 1

# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections and real-time communication
    Standalone version with Socket.IO support
    """

    def __init__(self, max_connections: int = 1000, heartbeat_interval: int = 30):
        """Initialize connection manager"""
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval

        # Connection tracking
        self.active_connections: Dict[str, Dict[str, Any]] = {}  # sid -> connection_info
        self.rooms: Dict[str, set] = {}  # room_id -> set of sids
        self.user_sessions: Dict[str, str] = {}  # user_id -> sid

        # Statistics
        self.total_connections = 0
        self.total_messages = 0
        self.start_time = datetime.now()

        # Message queues for barge-in support
        self.message_queues: Dict[str, List[Dict]] = {}

        print(f"✅ Connection Manager initialized (max: {max_connections})")

    def add_connection(self, sid: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add a new connection"""
        if len(self.active_connections) >= self.max_connections:
            raise HTTPException(status_code=503, detail="Maximum connections reached")

        self.active_connections[sid] = {
            "user_id": user_id,
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "metadata": metadata or {},
            "rooms": set()
        }

        if user_id:
            self.user_sessions[user_id] = sid

        self.total_connections += 1
        print(f"✅ Connection added: {sid} (user: {user_id})")

    def remove_connection(self, sid: str):
        """Remove a connection"""
        if sid in self.active_connections:
            connection_info = self.active_connections[sid]
            user_id = connection_info["user_id"]

            # Remove from rooms
            for room_id in connection_info["rooms"]:
                if room_id in self.rooms:
                    self.rooms[room_id].discard(sid)
                    if not self.rooms[room_id]:
                        del self.rooms[room_id]

            # Remove user session
            if user_id and self.user_sessions.get(user_id) == sid:
                del self.user_sessions[user_id]

            del self.active_connections[sid]
            print(f"✅ Connection removed: {sid}")

    def update_activity(self, sid: str):
        """Update last activity timestamp"""
        if sid in self.active_connections:
            self.active_connections[sid]["last_activity"] = datetime.now()

    def join_room(self, sid: str, room_id: str):
        """Join a room"""
        if sid not in self.active_connections:
            return False

        if room_id not in self.rooms:
            self.rooms[room_id] = set()

        self.rooms[room_id].add(sid)
        self.active_connections[sid]["rooms"].add(room_id)
        print(f"✅ {sid} joined room: {room_id}")
        return True

    def leave_room(self, sid: str, room_id: str):
        """Leave a room"""
        if sid in self.active_connections and room_id in self.rooms:
            self.rooms[room_id].discard(sid)
            self.active_connections[sid]["rooms"].discard(room_id)

            if not self.rooms[room_id]:
                del self.rooms[room_id]

            print(f"✅ {sid} left room: {room_id}")
            return True
        return False

    def get_room_participants(self, room_id: str) -> List[str]:
        """Get participants in a room"""
        return list(self.rooms.get(room_id, set()))

    def broadcast_to_room(self, room_id: str, event: str, data: Any, exclude_sid: Optional[str] = None):
        """Broadcast message to room (placeholder - actual broadcast done by Socket.IO)"""
        participants = self.get_room_participants(room_id)
        if exclude_sid:
            participants = [sid for sid in participants if sid != exclude_sid]

        self.total_messages += len(participants)
        print(f"📤 Broadcasting '{event}' to {len(participants)} participants in room {room_id}")

    def send_to_user(self, user_id: str, event: str, data: Any):
        """Send message to specific user (placeholder)"""
        if user_id in self.user_sessions:
            sid = self.user_sessions[user_id]
            self.total_messages += 1
            print(f"📤 Sending '{event}' to user {user_id} (sid: {sid})")

    def add_to_message_queue(self, conversation_id: str, message: Dict):
        """Add message to queue for barge-in processing"""
        if conversation_id not in self.message_queues:
            self.message_queues[conversation_id] = []

        self.message_queues[conversation_id].append({
            **message,
            "queued_at": datetime.now(),
            "status": "queued"
        })

        print(f"📋 Message queued for conversation {conversation_id}")

    def get_message_queue(self, conversation_id: str) -> List[Dict]:
        """Get message queue for conversation"""
        return self.message_queues.get(conversation_id, [])

    def clear_message_queue(self, conversation_id: str):
        """Clear message queue for conversation"""
        if conversation_id in self.message_queues:
            del self.message_queues[conversation_id]
            print(f"🗑️ Cleared message queue for conversation {conversation_id}")

    def get_stats(self) -> ConnectionStats:
        """Get connection statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return ConnectionStats(
            total_connections=self.total_connections,
            active_connections=len(self.active_connections),
            total_messages=self.total_messages,
            uptime_seconds=int(uptime),
            timestamp=datetime.now()
        )

    def cleanup_inactive_connections(self, max_idle_seconds: int = 3600):
        """Clean up inactive connections"""
        now = datetime.now()
        to_remove = []

        for sid, info in self.active_connections.items():
            idle_time = (now - info["last_activity"]).total_seconds()
            if idle_time > max_idle_seconds:
                to_remove.append(sid)

        for sid in to_remove:
            print(f"🧹 Removing inactive connection: {sid}")
            self.remove_connection(sid)

        return len(to_remove)

# ============================================================================
# Global Connection Manager Instance
# ============================================================================

try:
    config = get_config()
    connection_manager = ConnectionManager(
        max_connections=config["websocket"]["max_connections"],
        heartbeat_interval=config["websocket"]["heartbeat_interval"]
    )
    print("✅ WebSocket Connection Manager initialized")
except Exception as e:
    print(f"⚠️  Connection Manager failed: {e}")
    connection_manager = None

# ============================================================================
# Socket.IO Server Setup
# ============================================================================

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    ping_timeout=config["websocket"]["ping_timeout"],
    ping_interval=config["websocket"]["ping_interval"],
    logger=False,
    engineio_logger=False
)

# ============================================================================
# Socket.IO Event Handlers
# ============================================================================

@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    print(f"🔌 Client connected: {sid}")

    user_id = auth.get("user_id") if auth else None
    metadata = auth or {}

    try:
        if connection_manager:
            connection_manager.add_connection(sid, user_id, metadata)

        # Send welcome message
        await sio.emit('connected', {
            'sid': sid,
            'timestamp': datetime.now().isoformat(),
            'message': 'Successfully connected to WebSocket service'
        }, to=sid)

    except Exception as e:
        print(f"❌ Connection error for {sid}: {e}")
        await sio.disconnect(sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"🔌 Client disconnected: {sid}")

    if connection_manager:
        connection_manager.remove_connection(sid)

@sio.event
async def join_room(sid, data):
    """Handle room joining"""
    room_id = data.get("room_id")
    if not room_id:
        await sio.emit('error', {'message': 'room_id required'}, to=sid)
        return

    print(f"🏠 {sid} joining room: {room_id}")

    if connection_manager and connection_manager.join_room(sid, room_id):
        await sio.emit('room_joined', {
            'room_id': room_id,
            'timestamp': datetime.now().isoformat()
        }, to=sid)

        # Notify others in room
        await sio.emit('user_joined', {
            'user_sid': sid,
            'room_id': room_id,
            'timestamp': datetime.now().isoformat()
        }, room=room_id, skip_sid=sid)
    else:
        await sio.emit('error', {'message': 'Failed to join room'}, to=sid)

@sio.event
async def leave_room(sid, data):
    """Handle room leaving"""
    room_id = data.get("room_id")
    if not room_id:
        await sio.emit('error', {'message': 'room_id required'}, to=sid)
        return

    print(f"🏠 {sid} leaving room: {room_id}")

    if connection_manager and connection_manager.leave_room(sid, room_id):
        await sio.emit('room_left', {
            'room_id': room_id,
            'timestamp': datetime.now().isoformat()
        }, to=sid)

        # Notify others in room
        await sio.emit('user_left', {
            'user_sid': sid,
            'room_id': room_id,
            'timestamp': datetime.now().isoformat()
        }, room=room_id, skip_sid=sid)

@sio.event
async def message(sid, data):
    """Handle incoming messages"""
    if connection_manager:
        connection_manager.update_activity(sid)
        connection_manager.total_messages += 1

    message_data = MessageData(**data) if isinstance(data, dict) else MessageData(type="unknown", content=data)
    message_data.timestamp = datetime.now()

    print(f"💬 Message from {sid}: {message_data.type}")

    # Echo message back (simple implementation)
    await sio.emit('message_received', {
        'original_message': data,
        'timestamp': message_data.timestamp.isoformat(),
        'processed': True
    }, to=sid)

@sio.event
async def barge_in(sid, data):
    """Handle barge-in requests (interrupt current conversation)"""
    if not isinstance(data, dict):
        await sio.emit('error', {'message': 'Invalid barge-in data'}, to=sid)
        return

    barge_in_request = BargeInRequest(**data)

    print(f"🚨 Barge-in from {sid} for conversation {barge_in_request.conversation_id}")

    if connection_manager:
        # Add to message queue
        connection_manager.add_to_message_queue(
            barge_in_request.conversation_id,
            {
                "sid": sid,
                "user_message": barge_in_request.user_message,
                "priority": barge_in_request.priority,
                "timestamp": datetime.now().isoformat()
            }
        )

        # Acknowledge barge-in
        await sio.emit('barge_in_acknowledged', {
            'conversation_id': barge_in_request.conversation_id,
            'priority': barge_in_request.priority,
            'queued': True,
            'timestamp': datetime.now().isoformat()
        }, to=sid)

        # Broadcast barge-in to room (if in a conversation room)
        room_id = f"conversation_{barge_in_request.conversation_id}"
        if room_id in (connection_manager.rooms if connection_manager else {}):
            await sio.emit('barge_in_notification', {
                'conversation_id': barge_in_request.conversation_id,
                'user_sid': sid,
                'priority': barge_in_request.priority,
                'timestamp': datetime.now().isoformat()
            }, room=room_id, skip_sid=sid)

@sio.event
async def heartbeat(sid, data):
    """Handle heartbeat messages"""
    if connection_manager:
        connection_manager.update_activity(sid)

    await sio.emit('heartbeat_response', {
        'timestamp': datetime.now().isoformat(),
        'status': 'alive'
    }, to=sid)

@sio.event
async def speech_to_speech(sid, data):
    """Handle speech-to-speech request via WebSocket - connects to orchestrator"""
    import aiohttp
    import base64
    
    if not isinstance(data, dict):
        await sio.emit('error', {'message': 'Invalid speech-to-speech data'}, to=sid)
        return
    
    try:
        print(f"🎤 Speech-to-speech request from {sid}")
        
        # Extract parameters
        audio_base64 = data.get('audio_base64')
        file_data = data.get('file')  # Binary data if sent directly
        language = data.get('language', 'en')
        # Get voice_id - let TTS service choose default if not specified
        voice_id_raw = data.get('voice_id')  # None will use TTS service default
        # Normalize voice_id - treat None, empty, or "None" string as None
        voice_id = None if (not voice_id_raw or voice_id_raw == "None" or voice_id_raw == "") else voice_id_raw
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.7)
        voice_speed = data.get('voice_speed', 1.0)
        
        if not audio_base64 and not file_data:
            await sio.emit('error', {
                'message': 'Either audio_base64 or file must be provided'
            }, to=sid)
            return
        
        # Get orchestrator URL from environment or use default
        orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8500")
        
        # Prepare form data for orchestrator
        form = aiohttp.FormData()
        
        if audio_base64:
            form.add_field('audio_base64', audio_base64)
        elif file_data:
            # If file_data is base64 string, decode it
            if isinstance(file_data, str):
                audio_bytes = base64.b64decode(file_data)
            else:
                audio_bytes = file_data
            
            form.add_field('file', audio_bytes, filename='audio.wav', content_type='audio/wav')
        
        form.add_field('language', language)
        # Only add voice_id if explicitly specified
        # The orchestrator will use default voice for the provider if not specified
        if voice_id:
            form.add_field('voice_id', voice_id)
        form.add_field('max_tokens', str(max_tokens))
        form.add_field('temperature', str(temperature))
        form.add_field('voice_speed', str(voice_speed))
        
        # Send progress update
        await sio.emit('speech_to_speech_progress', {
            'stage': 'processing',
            'message': 'Sending to orchestrator...',
            'timestamp': datetime.now().isoformat()
        }, to=sid)
        
        # Call orchestrator
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{orchestrator_url}/api/process",
                data=form,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    
                    if result.get('success'):
                        # Send success response
                        await sio.emit('speech_response', {
                            'success': True,
                            'transcription': result.get('transcription', ''),
                            'response_text': result.get('response_text', ''),
                            'audio_base64': result.get('audio_base64', ''),
                            'processing_times': result.get('processing_times', {}),
                            'metadata': result.get('metadata', {}),
                            'timestamp': datetime.now().isoformat()
                        }, to=sid)
                        
                        print(f"✅ Speech-to-speech completed for {sid}")
                    else:
                        # Send error response
                        await sio.emit('speech_response', {
                            'success': False,
                            'error': result.get('error', 'Unknown error'),
                            'processing_times': result.get('processing_times', {}),
                            'timestamp': datetime.now().isoformat()
                        }, to=sid)
                        
                        print(f"❌ Speech-to-speech failed for {sid}: {result.get('error')}")
                else:
                    error_text = await resp.text()
                    await sio.emit('error', {
                        'message': f'Orchestrator returned {resp.status}: {error_text}',
                        'timestamp': datetime.now().isoformat()
                    }, to=sid)
                    
                    print(f"❌ Orchestrator error for {sid}: {resp.status} - {error_text}")
                    
    except Exception as e:
        error_msg = f'Speech-to-speech processing failed: {str(e)}'
        print(f"❌ Error processing speech-to-speech for {sid}: {e}")
        
        await sio.emit('error', {
            'message': error_msg,
            'timestamp': datetime.now().isoformat()
        }, to=sid)

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(title="WebSocket Service", version="1.0.0", description="Real-time Socket.IO communication service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Socket.IO
app.mount("/socket.io", socketio.ASGIApp(sio))

# ============================================================================
# REST API Routes
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    stats = connection_manager.get_stats() if connection_manager else None

    return HealthResponse(
        status="healthy" if connection_manager else "unhealthy",
        service="websocket",
        version="1.0.0",
        websocket_enabled=True,
        connections={
            "total": stats.total_connections if stats else 0,
            "active": stats.active_connections if stats else 0,
            "messages": stats.total_messages if stats else 0
        },
        timestamp=datetime.now()
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "WebSocket Gateway",
        "description": "Real-time communication via Socket.IO",
        "version": "1.0.0",
        "protocol": "Socket.IO",
        "endpoints": {
            "socket.io": f"http://localhost:{config['service']['port']}/socket.io",
            "health": "/health",
            "stats": "/stats",
            "rooms": "/rooms"
        }
    }

@app.get("/stats")
async def get_stats():
    """Get connection statistics"""
    if not connection_manager:
        raise HTTPException(status_code=503, detail="Connection manager not available")

    return connection_manager.get_stats()

@app.get("/rooms")
async def get_rooms():
    """Get information about active rooms"""
    if not connection_manager:
        raise HTTPException(status_code=503, detail="Connection manager not available")

    rooms_info = []
    for room_id, participants in connection_manager.rooms.items():
        rooms_info.append({
            "room_id": room_id,
            "participants": len(participants),
            "participant_sids": list(participants)
        })

    return {"rooms": rooms_info, "total_rooms": len(rooms_info)}

@app.get("/connections")
async def get_connections():
    """Get information about active connections"""
    if not connection_manager:
        raise HTTPException(status_code=503, detail="Connection manager not available")

    connections_info = []
    for sid, info in connection_manager.active_connections.items():
        connections_info.append({
            "sid": sid,
            "user_id": info["user_id"],
            "connected_at": info["connected_at"].isoformat(),
            "last_activity": info["last_activity"].isoformat(),
            "rooms": list(info["rooms"]),
            "metadata": info["metadata"]
        })

    return {
        "connections": connections_info,
        "total_connections": len(connections_info)
    }

@app.delete("/connections/{sid}")
async def disconnect_connection(sid: str):
    """Force disconnect a connection (admin)"""
    if not connection_manager:
        raise HTTPException(status_code=503, detail="Connection manager not available")

    if sid not in connection_manager.active_connections:
        raise HTTPException(status_code=404, detail="Connection not found")

    connection_manager.remove_connection(sid)
    await sio.disconnect(sid)

    return {"message": f"Connection {sid} disconnected"}

@app.get("/message-queues/{conversation_id}")
async def get_message_queue(conversation_id: str):
    """Get message queue for a conversation"""
    if not connection_manager:
        raise HTTPException(status_code=503, detail="Connection manager not available")

    queue = connection_manager.get_message_queue(conversation_id)
    return {
        "conversation_id": conversation_id,
        "queue_length": len(queue),
        "messages": queue
    }

@app.delete("/message-queues/{conversation_id}")
async def clear_message_queue(conversation_id: str):
    """Clear message queue for a conversation"""
    if not connection_manager:
        raise HTTPException(status_code=503, detail="Connection manager not available")

    connection_manager.clear_message_queue(conversation_id)
    return {"message": f"Message queue cleared for conversation {conversation_id}"}

@app.post("/cleanup")
async def cleanup_inactive():
    """Clean up inactive connections"""
    if not connection_manager:
        raise HTTPException(status_code=503, detail="Connection manager not available")

    removed = connection_manager.cleanup_inactive_connections()
    return {"message": f"Cleaned up {removed} inactive connections"}

# Add standard endpoints
router = socketio.ASGIApp(sio)  # Socket.IO is already mounted above
add_standard_endpoints(app.router)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service"""
    print("🚀 Initializing WebSocket Service...")
    print(f"   Socket.IO Port: {config['service']['port']}")
    print(f"   Max Connections: {config['websocket']['max_connections']}")
    print(f"   CORS Origins: {config['websocket']['cors_origins']}")

    if connection_manager:
        stats = connection_manager.get_stats()
        print(f"   Active Connections: {stats.active_connections}")
        print(f"   Total Messages: {stats.total_messages}")

        # Start heartbeat task
        asyncio.create_task(heartbeat_task())

    print("✅ WebSocket Service initialized successfully!")

async def heartbeat_task():
    """Background task for heartbeat and cleanup"""
    while True:
        try:
            await asyncio.sleep(config["websocket"]["heartbeat_interval"])

            if connection_manager:
                # Cleanup inactive connections
                removed = connection_manager.cleanup_inactive_connections(
                    config["websocket"]["connection_timeout"]
                )

                if removed > 0:
                    print(f"🧹 Cleaned up {removed} inactive connections")

        except Exception as e:
            print(f"⚠️  Heartbeat task error: {e}")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8022"))
    print(f"Starting WebSocket Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

