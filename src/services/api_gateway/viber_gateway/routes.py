"""
Viber Gateway Routes
Webhook endpoint and sending messages
"""

from fastapi import APIRouter, Request, HTTPException
from typing import Dict

from .models import (
    SendTextRequest,
    SendVideoRequest,
    SendImageRequest,
    SendMessageResponse
)


def create_router(service):
    """Create FastAPI router with service dependency"""
    router = APIRouter(prefix="/viber", tags=["viber"])

    @router.post("/webhook")
    async def viber_webhook(request: Request):
        """
        Viber webhook endpoint
        Receives all incoming messages from Viber

        Important: Must be HTTPS in production
        """
        try:
            from viberbot.api.viber_requests import ViberRequest

            # Parse Viber request
            request_data = await request.body()
            viber_request = ViberRequest.create(request_data.decode())

            service.logger.info(f"📞 Webhook event: {type(viber_request).__name__}")

            # Handle different event types
            if hasattr(viber_request, 'message'):
                # Message event
                await service.process_incoming_message(viber_request)

            return {"status": "ok"}

        except Exception as e:
            service.logger.error(f"Webhook error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/send/text", response_model=SendMessageResponse)
    async def send_text_message(req: SendTextRequest):
        """Send text message via Viber"""
        return await service.send_text(receiver=req.receiver, text=req.text)

    @router.post("/send/video", response_model=SendMessageResponse)
    async def send_video_message(req: SendVideoRequest):
        """Send video message via Viber"""
        return await service.send_video(
            receiver=req.receiver,
            video_url=req.video_url,
            size=req.size,
            thumbnail=req.thumbnail,
            duration=req.duration
        )

    @router.post("/send/image", response_model=SendMessageResponse)
    async def send_image_message(req: SendImageRequest):
        """Send image message via Viber"""
        return await service.send_image(
            receiver=req.receiver,
            image_url=req.image_url,
            text=req.text
        )

    @router.get("/status")
    async def get_status() -> Dict:
        """Get Viber Gateway status"""
        return await service.health_check()

    return router
