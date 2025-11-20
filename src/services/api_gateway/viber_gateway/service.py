"""
Viber Gateway Service - Multi-modal messaging
Supports: Text, Video (.mp4), Images, Files (receive), Location, Stickers
Uses viberbot library for Viber Bot API integration
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Logging setup
, increment_metric
from loguru import logger

.parent / "tmp" / "metrics"
)

from .utils.base_service import BaseService
from src.services.orchestrator.utils.context import ServiceContext

from .config import get_config
from .models import SendTextRequest, SendVideoRequest, SendImageRequest, SendMessageResponse

class ViberGatewayService(BaseService):
    """
    Viber Gateway - Multi-modal messaging via Viber Bot API

    Features:
    - Text messages (send/receive)
    - Video messages (send/receive .mp4)
    - Image messages (send/receive)
    - Files (receive only)
    - Location, stickers
    - Integration with Orchestrator for AI processing

    Note: Commercial Viber Bot account required (since 5.02.24)
    """

    def __init__(self, config: Dict = None, context: Optional[ServiceContext] = None):
        super().__init__(context=context, config=config)

        self.config = get_config()
        self.viber = None  # Viber API instance

        self.logger.info("🎯 Viber Gateway Service initialized")

    def _setup_router(self) -> None:
        """Setup FastAPI routes"""
        from .routes import create_router
        router = create_router(service=self)
        self.router.include_router(router)

    async def initialize(self) -> bool:
        """Initialize Viber bot"""
        try:
            config = self.config

            self.logger.info(f"Viber Gateway initializing on port {config['service']['port']}")

            # Validate credentials
            if not config['viber']['auth_token']:
                raise ValueError("VIBER_AUTH_TOKEN not configured!")

            # Initialize Viber API
            try:
                from viberbot import Api
                from viberbot.api.bot_configuration import BotConfiguration

                bot_config = BotConfiguration(
                    name=config['viber']['bot_name'],
                    avatar=config['viber']['avatar'],
                    auth_token=config['viber']['auth_token']
                )

                self.viber = Api(bot_config)

                self.logger.info("✅ Viber API initialized successfully")

                # Set webhook if configured
                if config['viber'].get('webhook_url'):
                    self.viber.set_webhook(config['viber']['webhook_url'])
                    self.logger.info(f"✅ Webhook set: {config['viber']['webhook_url']}")

            except ImportError:
                self.logger.error("❌ viberbot not installed! Run: pip install viberbot")
                return False
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Viber API: {e}")
                return False

            # Create media directory
            config['media']['tmp_dir'].mkdir(parents=True, exist_ok=True)

            self.logger.info("✅ Viber Gateway Service initialized successfully")

            # Log enabled features
            enabled_features = [k for k, v in config['features'].items() if v]
            self.logger.info(f"📋 Enabled features: {', '.join(enabled_features)}")

            self.initialized = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Viber Gateway: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def process_incoming_message(self, viber_request):
        """Process incoming message from Viber webhook"""
        try:
            from viberbot.api.messages import TextMessage as ViberTextMessage
            from viberbot.api.messages import VideoMessage as ViberVideoMessage

            message = viber_request.message
            sender = viber_request.sender

            self.logger.info(f"📨 Message from {sender.name} ({sender.id}): {message}")

            # Handle different message types
            if isinstance(message, ViberTextMessage):
                await self._handle_text_message(sender, message)

            elif isinstance(message, ViberVideoMessage):
                await self._handle_video_message(sender, message)

            # Add handlers for other types as needed

        except Exception as e:
            self.logger.error(f"Error processing incoming message: {e}")

    async def _handle_text_message(self, sender, message):
        """Handle incoming text message"""
        try:
            from viberbot.api.messages import TextMessage as ViberTextMessage

            # Process via Orchestrator
            response = await self.comm.call_service(
                service_name="orchestrator",
                endpoint="/process_text",
                method="POST",
                data={
                    "text": message.text,
                    "user_id": sender.id,
                    "session_id": f"viber_{sender.id}",
                    "source": "viber"
                }
            )

            # Send response
            if response and response.get('success'):
                reply_message = ViberTextMessage(text=response['response_text'])
                self.viber.send_messages(sender.id, [reply_message])

                increment_metric("messages_processed", "viber_gateway", {"type": "text"})

        except Exception as e:
            self.logger.error(f"Error handling text message: {e}")

    async def _handle_video_message(self, sender, message):
        """Handle incoming video message"""
        try:
            from viberbot.api.messages import TextMessage as ViberTextMessage

            self.logger.info(f"🎥 Video from {sender.name}: {message.media}")

            # For now, acknowledge receipt
            # TODO: Download video, process via Orchestrator
            reply = ViberTextMessage(text="Vídeo recebido! Processamento em breve.")
            self.viber.send_messages(sender.id, [reply])

            increment_metric("messages_processed", "viber_gateway", {"type": "video"})

        except Exception as e:
            self.logger.error(f"Error handling video: {e}")

    async def send_text(self, receiver: str, text: str) -> SendMessageResponse:
        """Send text message"""
        try:
            from viberbot.api.messages import TextMessage as ViberTextMessage

            message = ViberTextMessage(text=text)
            result = self.viber.send_messages(receiver, [message])

            return SendMessageResponse(
                success=True,
                message_token=result[0] if result else None,
                status=0,
                status_message="OK"
            )

        except Exception as e:
            self.logger.error(f"Failed to send text: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def send_video(self, receiver: str, video_url: str, size: int,
                        thumbnail: str = None, duration: int = None) -> SendMessageResponse:
        """Send video message"""
        try:
            from viberbot.api.messages import VideoMessage as ViberVideoMessage

            message = ViberVideoMessage(
                media=video_url,
                size=size,
                thumbnail=thumbnail,
                duration=duration
            )
            result = self.viber.send_messages(receiver, [message])

            return SendMessageResponse(
                success=True,
                message_token=result[0] if result else None,
                status=0,
                status_message="OK"
            )

        except Exception as e:
            self.logger.error(f"Failed to send video: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def send_image(self, receiver: str, image_url: str,
                        text: str = None) -> SendMessageResponse:
        """Send image message"""
        try:
            from viberbot.api.messages import PictureMessage

            message = PictureMessage(media=image_url, text=text)
            result = self.viber.send_messages(receiver, [message])

            return SendMessageResponse(
                success=True,
                message_token=result[0] if result else None,
                status=0,
                status_message="OK"
            )

        except Exception as e:
            self.logger.error(f"Failed to send image: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def health_check(self) -> Dict:
        """Health check"""
        return {
            "status": "healthy" if self.initialized else "unhealthy",
            "service": "viber-gateway",
            "timestamp": datetime.now().isoformat(),
            "features": self.config['features'],
            "viber_api": "connected" if self.viber else "disconnected"
        }

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self.logger.info("🛑 Shutting down Viber Gateway...")
        self.initialized = False
        self.logger.info("✅ Viber Gateway shut down successfully")
