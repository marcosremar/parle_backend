"""
Pytest fixtures for Viber Gateway tests
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.orchestrator.utils.context import ServiceContext
from src.services.viber_gateway.service import ViberGatewayService
from src.services.viber_gateway.config import get_config


@pytest.fixture
def mock_config():
    """Mock Viber configuration"""
    return {
        "service": {
            "name": "viber_gateway",
            "port": 8806,
            "host": "0.0.0.0",
            "version": "1.0.0"
        },
        "viber": {
            "auth_token": "test_viber_token_123",
            "bot_name": "Test Ultravox Bot",
            "avatar": "https://example.com/avatar.jpg",
            "webhook_url": "https://example.com/viber/webhook",
            "api_url": "https://chatapi.viber.com/pa",
        },
        "features": {
            "text_messages": True,
            "video_messages": True,
            "image_messages": True,
            "file_messages": True,
            "location_messages": True,
            "stickers": True,
            "rich_media": False,
        },
        "orchestrator": {
            "service_name": "orchestrator",
            "text_endpoint": "/process_text",
            "video_endpoint": "/process_video",
        },
        "media": {
            "tmp_dir": Path("/tmp/viber_test"),
            "max_video_size_mb": 50,
            "max_image_size_mb": 5,
            "supported_video_formats": ["mp4"],
            "supported_image_formats": ["jpeg", "jpg", "png", "gif"],
        },
        "limits": {
            "max_text_length": 7000,
            "min_api_version": 7,
        }
    }


@pytest.fixture
def mock_context():
    """Mock ServiceContext"""
    context = Mock(spec=ServiceContext)

    # Mock logger
    context.logger = Mock()
    context.logger.info = Mock()
    context.logger.error = Mock()
    context.logger.warning = Mock()

    # Mock communication manager
    context.comm = AsyncMock()
    context.comm.call_service = AsyncMock()

    # Mock telemetry
    context.telemetry = Mock()

    return context


@pytest.fixture
def mock_viber_api():
    """Mock Viber API"""
    api = Mock()

    # Mock send methods
    api.send_messages = Mock(return_value=["msg_token_123"])
    api.set_webhook = Mock()

    return api


@pytest.fixture
def viber_service(mock_config, mock_context, mock_viber_api, monkeypatch):
    """Create ViberGatewayService with mocks"""

    # Mock viberbot module and its message classes
    mock_text_message = Mock()
    mock_video_message = Mock()
    mock_image_message = Mock()

    # Create mock module structure
    mock_viberbot_messages = Mock()
    mock_viberbot_messages.TextMessage = mock_text_message
    mock_viberbot_messages.VideoMessage = mock_video_message
    mock_viberbot_messages.PictureMessage = mock_image_message

    # Inject mocks into sys.modules
    monkeypatch.setitem(sys.modules, 'viberbot', Mock())
    monkeypatch.setitem(sys.modules, 'viberbot.api', Mock())
    monkeypatch.setitem(sys.modules, 'viberbot.api.messages', mock_viberbot_messages)

    # Mock get_config
    monkeypatch.setattr(
        "src.services.viber_gateway.service.get_config",
        lambda: mock_config
    )

    # Create service
    service = ViberGatewayService(config=mock_config, context=mock_context)

    # Mock Viber API
    service.viber = mock_viber_api

    return service


@pytest.fixture
def mock_viber_user():
    """Mock Viber user"""
    user = Mock()
    user.id = "viber_user_123"
    user.name = "Test User"
    user.avatar = "https://example.com/user.jpg"
    user.country = "BR"
    user.language = "pt-BR"

    return user


@pytest.fixture
def mock_viber_text_message(mock_viber_user):
    """Mock Viber text message"""
    from viberbot.api.messages import TextMessage

    message = TextMessage(text="Hello, Viber Bot!")
    return message


@pytest.fixture
def mock_viber_request(mock_viber_user, mock_viber_text_message):
    """Mock Viber webhook request"""
    request = Mock()
    request.message = mock_viber_text_message
    request.sender = mock_viber_user
    request.message_token = "msg_token_123"

    return request
