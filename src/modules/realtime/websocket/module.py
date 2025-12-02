"""
WebSocket Module - Real-time communication via Socket.IO
"""

import os
import asyncio
import socketio
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.modules.base_module import BaseModule
from loguru import logger

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "websocket": {
        "ping_timeout": 60,
        "ping_interval": 25,
        "max_connections": 1000,
        "cors_origins": ["*"],
        "heartbeat_interval": 30,
        "connection_timeout": 3600  # 1 hour
    }
}

# ============================================================================
# Pydantic Models
# ============================================================================

class ConnectionStats(BaseModel):
    """Connection statistics"""
    total_connections: int
    active_connections: int
    total_messages: int
    uptime_seconds: int
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

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    websocket_enabled: bool
    connections: Dict[str, int]
    timestamp: datetime

# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections and real-time communication"""

    def __init__(self, max_connections: int = 1000):
        self.max_connections = max_connections
        self.active_connections: Dict[str, Dict[str, Any]] = {}  # sid -> connection_info
        self.rooms: Dict[str, set] = {}  # room_id -> set of sids
        self.user_sessions: Dict[str, str] = {}  # user_id -> sid
        self.total_connections = 0
        self.total_messages = 0
        self.start_time = datetime.now()
        self.message_queues: Dict[str, List[Dict]] = {}

    def add_connection(self, sid: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None):
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

    def remove_connection(self, sid: str):
        if sid in self.active_connections:
            connection_info = self.active_connections[sid]
            user_id = connection_info["user_id"]

            for room_id in list(connection_info["rooms"]):
                if room_id in self.rooms:
                    self.rooms[room_id].discard(sid)
                    if not self.rooms[room_id]:
                        del self.rooms[room_id]

            if user_id and self.user_sessions.get(user_id) == sid:
                del self.user_sessions[user_id]

            del self.active_connections[sid]

    def update_activity(self, sid: str):
        if sid in self.active_connections:
            self.active_connections[sid]["last_activity"] = datetime.now()

    def join_room(self, sid: str, room_id: str):
        if sid not in self.active_connections:
            return False
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(sid)
        self.active_connections[sid]["rooms"].add(room_id)
        return True

    def leave_room(self, sid: str, room_id: str):
        if sid in self.active_connections and room_id in self.rooms:
            self.rooms[room_id].discard(sid)
            self.active_connections[sid]["rooms"].discard(room_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
            return True
        return False

    def add_to_message_queue(self, conversation_id: str, message: Dict):
        if conversation_id not in self.message_queues:
            self.message_queues[conversation_id] = []
        self.message_queues[conversation_id].append({
            **message,
            "queued_at": datetime.now(),
            "status": "queued"
        })

    def get_message_queue(self, conversation_id: str) -> List[Dict]:
        return self.message_queues.get(conversation_id, [])

    def clear_message_queue(self, conversation_id: str):
        if conversation_id in self.message_queues:
            del self.message_queues[conversation_id]

    def get_stats(self) -> ConnectionStats:
        uptime = (datetime.now() - self.start_time).total_seconds()
        return ConnectionStats(
            total_connections=self.total_connections,
            active_connections=len(self.active_connections),
            total_messages=self.total_messages,
            uptime_seconds=int(uptime),
            timestamp=datetime.now()
        )

    def cleanup_inactive_connections(self, max_idle_seconds: int = 3600):
        now = datetime.now()
        to_remove = []
        for sid, info in self.active_connections.items():
            idle_time = (now - info["last_activity"]).total_seconds()
            if idle_time > max_idle_seconds:
                to_remove.append(sid)
        
        for sid in to_remove:
            self.remove_connection(sid)
        return len(to_remove)

# ============================================================================
# WebSocket Module
# ============================================================================

class WebSocketModule(BaseModule):
    """WebSocket Module for real-time communication"""

    def __init__(self):
        super().__init__("websocket")
        self.connection_manager: Optional[ConnectionManager] = None
        self.sio: Optional[socketio.AsyncServer] = None
        self.router = APIRouter(tags=["websocket"])

    async def _initialize(self) -> bool:
        """Initialize WebSocket module"""
        try:
            self.logger.info("🚀 Initializing WebSocket Module...")
            
            # Initialize Connection Manager
            self.connection_manager = ConnectionManager(
                max_connections=DEFAULT_CONFIG["websocket"]["max_connections"]
            )
            
            # Initialize Socket.IO Server
            self.sio = socketio.AsyncServer(
                async_mode='asgi',
                cors_allowed_origins='*',
                ping_timeout=DEFAULT_CONFIG["websocket"]["ping_timeout"],
                ping_interval=DEFAULT_CONFIG["websocket"]["ping_interval"],
                logger=False,
                engineio_logger=False
            )
            
            # Register Event Handlers
            self._register_handlers()
            
            # Register API Routes
            self._register_routes()
            
            # Start heartbeat task
            asyncio.create_task(self._heartbeat_task())
            
            self.logger.info("✅ WebSocket Module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize WebSocket Module: {e}")
            return False

    def _register_handlers(self):
        """Register Socket.IO event handlers"""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            self.logger.debug(f"🔌 Client connected: {sid}")
            user_id = auth.get("user_id") if auth else None
            metadata = auth or {}
            
            try:
                if self.connection_manager:
                    self.connection_manager.add_connection(sid, user_id, metadata)
                
                await self.sio.emit('connected', {
                    'sid': sid,
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Successfully connected to WebSocket service'
                }, to=sid)
            except Exception as e:
                self.logger.error(f"❌ Connection error for {sid}: {e}")
                await self.sio.disconnect(sid)

        @self.sio.event
        async def disconnect(sid):
            self.logger.debug(f"🔌 Client disconnected: {sid}")
            if self.connection_manager:
                self.connection_manager.remove_connection(sid)

        @self.sio.event
        async def join_room(sid, data):
            room_id = data.get("room_id")
            if not room_id:
                await self.sio.emit('error', {'message': 'room_id required'}, to=sid)
                return
            
            if self.connection_manager and self.connection_manager.join_room(sid, room_id):
                await self.sio.emit('room_joined', {'room_id': room_id, 'timestamp': datetime.now().isoformat()}, to=sid)

        @self.sio.event
        async def leave_room(sid, data):
            room_id = data.get("room_id")
            if not room_id:
                return
            
            if self.connection_manager and self.connection_manager.leave_room(sid, room_id):
                await self.sio.emit('room_left', {'room_id': room_id, 'timestamp': datetime.now().isoformat()}, to=sid)

        @self.sio.event
        async def message(sid, data):
            if self.connection_manager:
                self.connection_manager.update_activity(sid)
                self.connection_manager.total_messages += 1
            
            # Echo back
            await self.sio.emit('message_received', {
                'original_message': data,
                'timestamp': datetime.now().isoformat(),
                'processed': True
            }, to=sid)

        @self.sio.event
        async def barge_in(sid, data):
            if not isinstance(data, dict):
                return
            
            barge_in_request = BargeInRequest(**data)
            self.logger.info(f"🚨 Barge-in from {sid} for conversation {barge_in_request.conversation_id}")
            
            if self.connection_manager:
                self.connection_manager.add_to_message_queue(
                    barge_in_request.conversation_id,
                    {
                        "sid": sid,
                        "user_message": barge_in_request.user_message,
                        "priority": barge_in_request.priority,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                await self.sio.emit('barge_in_acknowledged', {
                    'conversation_id': barge_in_request.conversation_id,
                    'queued': True
                }, to=sid)

        @self.sio.event
        async def heartbeat(sid, data):
            if self.connection_manager:
                self.connection_manager.update_activity(sid)
            await self.sio.emit('heartbeat_response', {'status': 'alive', 'timestamp': datetime.now().isoformat()}, to=sid)

    def _register_routes(self):
        """Register REST API routes"""
        
        @self.router.get("/health", response_model=HealthResponse)
        async def health_check():
            stats = self.connection_manager.get_stats() if self.connection_manager else None
            return HealthResponse(
                status="healthy" if self.connection_manager else "unhealthy",
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

        @self.router.get("/stats")
        async def get_stats():
            if not self.connection_manager:
                raise HTTPException(status_code=503, detail="Service unavailable")
            return self.connection_manager.get_stats()

    async def _heartbeat_task(self):
        """Background task for cleanup"""
        while True:
            try:
                await asyncio.sleep(DEFAULT_CONFIG["websocket"]["heartbeat_interval"])
                if self.connection_manager:
                    removed = self.connection_manager.cleanup_inactive_connections(
                        DEFAULT_CONFIG["websocket"]["connection_timeout"]
                    )
                    if removed > 0:
                        self.logger.debug(f"🧹 Cleaned up {removed} inactive connections")
            except Exception as e:
                self.logger.error(f"⚠️ Heartbeat task error: {e}")
