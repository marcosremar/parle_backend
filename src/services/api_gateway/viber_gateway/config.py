"""
Viber Gateway Configuration
Supports: Text, Video, Images, Files (receive only), Location, Stickers
Note: Audio as separate type is not supported - use video with audio track
"""

import os
from typing import Dict
from pathlib import Path


def get_config() -> Dict:
    """Load Viber Gateway configuration"""

    return {
        "service": {
            "name": "viber_gateway",
            "port": int(os.getenv("VIBER_GATEWAY_PORT", "8806")),
            "host": "0.0.0.0",
            "version": "1.0.0"
        },

        "viber": {
            # Viber Bot credentials (from Viber Admin Panel)
            "auth_token": os.getenv("VIBER_AUTH_TOKEN"),
            "bot_name": os.getenv("VIBER_BOT_NAME", "Ultravox Bot"),
            "avatar": os.getenv("VIBER_BOT_AVATAR", ""),

            # Webhook
            "webhook_url": os.getenv("VIBER_WEBHOOK_URL"),  # Must be HTTPS

            # API settings
            "api_url": "https://chatapi.viber.com/pa",
        },

        "features": {
            # Enable/disable features
            "text_messages": True,
            "video_messages": True,  # MP4 videos
            "image_messages": True,  # Pictures/photos
            "file_messages": True,   # Receive files from users (cannot send)
            "location_messages": True,
            "stickers": True,
            "rich_media": False,  # Carousel/Rich Media (future)
        },

        "orchestrator": {
            "service_name": "orchestrator",
            "text_endpoint": "/process_text",
            "video_endpoint": "/process_video",
        },

        "media": {
            # Media storage
            "tmp_dir": Path(__file__).parent / "tmp" / "media",
            "max_video_size_mb": 50,  # Viber limit
            "max_image_size_mb": 5,
            "supported_video_formats": ["mp4"],
            "supported_image_formats": ["jpeg", "jpg", "png", "gif"],
        },

        "limits": {
            # Viber API limits
            "max_text_length": 7000,  # Viber allows up to 7000 chars
            "min_api_version": 7,  # Minimum API version required
        }
    }
