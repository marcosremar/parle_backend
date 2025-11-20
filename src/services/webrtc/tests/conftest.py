"""
Test Configuration and Fixtures for WebRTC Gateway Service.

Provides fixtures for:
- WebRTC peer connections
- Signaling mocks
- Media stream mocks
- ICE candidate handling
- STUN/TURN configuration
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock
import json


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def webrtc_config():
    """WebRTC Gateway configuration for testing."""
    return {
        "signaling_timeout": 30.0,
        "ice_timeout": 10.0,
        "max_peers": 100,
        "stun_servers": ["stun:stun.l.google.com:19302"],
        "turn_servers": [
            {
                "urls": "turn:turn.example.com:3478",
                "username": "test_user",
                "credential": "test_pass"
            }
        ],
        "ice_gathering_timeout": 5.0,
        "enable_data_channel": True
    }


# ============================================================================
# Signaling Fixtures
# ============================================================================

@pytest.fixture
def mock_signaling_service():
    """Mock signaling service."""

    class MockSignalingService:
        def __init__(self):
            self.offers = {}
            self.answers = {}
            self.ice_candidates = {}
            self.state = "stable"

        async def create_offer(self, peer_id: str) -> Dict:
            """Create SDP offer."""
            offer = {
                "type": "offer",
                "sdp": f"v=0\r\no=- {peer_id} 2 IN IP4 127.0.0.1\r\n",
                "peer_id": peer_id,
                "timestamp": datetime.now().isoformat()
            }
            self.offers[peer_id] = offer
            self.state = "have-local-offer"
            return offer

        async def create_answer(self, peer_id: str, offer: Dict) -> Dict:
            """Create SDP answer."""
            answer = {
                "type": "answer",
                "sdp": f"v=0\r\na=- {peer_id} 2 IN IP4 127.0.0.1\r\n",
                "peer_id": peer_id,
                "timestamp": datetime.now().isoformat()
            }
            self.answers[peer_id] = answer
            self.state = "stable"
            return answer

        async def add_ice_candidate(self, peer_id: str, candidate: Dict) -> bool:
            """Add ICE candidate."""
            if peer_id not in self.ice_candidates:
                self.ice_candidates[peer_id] = []
            self.ice_candidates[peer_id].append(candidate)
            return True

        def get_state(self) -> str:
            """Get signaling state."""
            return self.state

    return MockSignalingService()


@pytest.fixture
def sample_sdp_offer():
    """Sample SDP offer for testing."""
    return {
        "type": "offer",
        "sdp": """v=0
o=- 123456789 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE audio video
m=audio 9 UDP/TLS/RTP/SAVPF 111
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:test123
a=ice-pwd:testpassword123
a=rtpmap:111 opus/48000/2
m=video 9 UDP/TLS/RTP/SAVPF 96
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:test123
a=ice-pwd:testpassword123
a=rtpmap:96 VP8/90000
"""
    }


# ============================================================================
# Media Stream Fixtures
# ============================================================================

@pytest.fixture
def mock_media_stream():
    """Mock media stream."""

    class MockMediaStream:
        def __init__(self, stream_type: str = "audio"):
            self.id = f"{stream_type}_stream_{id(self)}"
            self.type = stream_type
            self.active = True
            self.tracks = []

        def add_track(self, track: Dict):
            """Add media track."""
            self.tracks.append(track)

        def get_tracks(self) -> List[Dict]:
            """Get all tracks."""
            return self.tracks

        def stop(self):
            """Stop stream."""
            self.active = False

    return MockMediaStream


@pytest.fixture
def audio_constraints():
    """Audio media constraints."""
    return {
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
            "sampleRate": 48000,
            "channelCount": 2
        }
    }


@pytest.fixture
def video_constraints():
    """Video media constraints."""
    return {
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
            "facingMode": "user"
        }
    }


# ============================================================================
# Peer Connection Fixtures
# ============================================================================

@pytest.fixture
async def mock_peer_connection():
    """Mock RTCPeerConnection."""

    class MockPeerConnection:
        def __init__(self):
            self.local_description = None
            self.remote_description = None
            self.ice_connection_state = "new"
            self.ice_gathering_state = "new"
            self.signaling_state = "stable"
            self.connection_state = "new"
            self.ice_candidates = []
            self.data_channels = {}
            self.tracks = []

        async def create_offer(self) -> Dict:
            """Create offer."""
            self.signaling_state = "have-local-offer"
            return {
                "type": "offer",
                "sdp": "v=0\r\no=- test 2 IN IP4 127.0.0.1\r\n"
            }

        async def create_answer(self) -> Dict:
            """Create answer."""
            self.signaling_state = "have-local-answer"
            return {
                "type": "answer",
                "sdp": "v=0\r\na=- test 2 IN IP4 127.0.0.1\r\n"
            }

        async def set_local_description(self, description: Dict):
            """Set local description."""
            self.local_description = description
            self.signaling_state = "stable"

        async def set_remote_description(self, description: Dict):
            """Set remote description."""
            self.remote_description = description

        async def add_ice_candidate(self, candidate: Dict):
            """Add ICE candidate."""
            self.ice_candidates.append(candidate)
            if self.ice_connection_state == "new":
                self.ice_connection_state = "checking"

        def add_track(self, track: Dict):
            """Add media track."""
            self.tracks.append(track)

        def create_data_channel(self, label: str, options: Dict = None) -> Dict:
            """Create data channel."""
            channel = {
                "label": label,
                "id": len(self.data_channels),
                "readyState": "open",
                "bufferedAmount": 0
            }
            self.data_channels[label] = channel
            return channel

        async def close(self):
            """Close connection."""
            self.connection_state = "closed"
            self.ice_connection_state = "closed"

    return MockPeerConnection()


# ============================================================================
# ICE Candidate Fixtures
# ============================================================================

@pytest.fixture
def sample_ice_candidate():
    """Sample ICE candidate."""
    return {
        "candidate": "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host",
        "sdpMid": "audio",
        "sdpMLineIndex": 0,
        "usernameFragment": "test123"
    }


@pytest.fixture
def ice_server_config():
    """ICE server configuration."""
    return {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {
                "urls": "turn:turn.example.com:3478",
                "username": "testuser",
                "credential": "testpass"
            }
        ],
        "iceTransportPolicy": "all",
        "bundlePolicy": "balanced",
        "rtcpMuxPolicy": "require"
    }


# ============================================================================
# Data Channel Fixtures
# ============================================================================

@pytest.fixture
def mock_data_channel():
    """Mock data channel."""

    class MockDataChannel:
        def __init__(self, label: str):
            self.label = label
            self.id = 0
            self.readyState = "open"
            self.bufferedAmount = 0
            self.messages = []

        def send(self, data: str):
            """Send data."""
            self.messages.append({
                "data": data,
                "timestamp": datetime.now().isoformat()
            })

        def close(self):
            """Close channel."""
            self.readyState = "closed"

    return MockDataChannel


# ============================================================================
# Quality Metrics Fixtures
# ============================================================================

@pytest.fixture
def media_quality_metrics():
    """Media quality metrics."""
    return {
        "audio": {
            "bitrate": 128000,
            "packetLoss": 0.01,
            "jitter": 5,
            "roundTripTime": 50
        },
        "video": {
            "bitrate": 2500000,
            "frameRate": 30,
            "width": 1280,
            "height": 720,
            "packetLoss": 0.02,
            "jitter": 10,
            "roundTripTime": 50
        }
    }


# ============================================================================
# Error Simulation Fixtures
# ============================================================================

@pytest.fixture
def network_interruption():
    """Simulate network interruption."""

    class NetworkInterruption:
        def __init__(self):
            self.interrupted = False

        def trigger(self):
            """Trigger interruption."""
            self.interrupted = True

        def restore(self):
            """Restore connection."""
            self.interrupted = False

        def is_interrupted(self) -> bool:
            """Check if interrupted."""
            return self.interrupted

    return NetworkInterruption()
