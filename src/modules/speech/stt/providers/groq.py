"""
Groq Transcription Provider
"""

import os
import tempfile
from typing import Any

import aiohttp
from fastapi import HTTPException


class GroqTranscriptionProvider:
    """Groq Whisper API transcription provider"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not provided")

        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self.timeout = 30
        self.max_retries = 3

    async def transcribe_audio(
        self, audio_data: bytes, language: str = "pt", model: str = "whisper-large-v3"
    ) -> dict[str, Any]:
        """Transcribe audio using Groq Whisper API"""
        temp_file_path = None
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Prepare multipart form data
            form_data = aiohttp.FormData()
            # Read file asynchronously
            import asyncio

            audio_content = await asyncio.to_thread(lambda: open(temp_file_path, "rb").read())
            form_data.add_field("file", audio_content, filename="audio.wav")
            form_data.add_field("model", model)
            form_data.add_field("language", language)
            form_data.add_field("response_format", "json")

            # Make API request
            from src.core.http_client import HTTPClient

            session = await HTTPClient.get_session()
            # Use custom timeout for this request
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{self.base_url}/audio/transcriptions",
                        data=form_data,
                        headers=self.headers,
                        timeout=timeout,
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return {
                                "text": result.get("text", ""),
                                "language": language,
                                "model": model,
                                "provider": "groq",
                            }
                        else:
                            error_text = await response.text()
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(1)
                                continue
                            else:
                                raise HTTPException(
                                    status_code=response.status,
                                    detail=f"Groq API error: {error_text}",
                                )
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise HTTPException(status_code=500, detail=f"Network error: {e!s}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription error: {e!s}")
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass  # Ignore cleanup errors
