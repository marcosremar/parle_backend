"""
Groq Transcription Provider
"""

import os
import tempfile
import asyncio
import aiohttp
from typing import Dict, Any, Optional
from fastapi import HTTPException


class GroqTranscriptionProvider:
    """Groq Whisper API transcription provider"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not provided")

        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self.timeout = 30
        self.max_retries = 3

    async def transcribe_audio(self, audio_data: bytes, language: str = "pt", model: str = "whisper-large-v3") -> Dict[str, Any]:
        """Transcribe audio using Groq Whisper API"""
        temp_file_path = None
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Prepare multipart form data
            form_data = aiohttp.FormData()
            with open(temp_file_path, 'rb') as audio_file:
                audio_content = audio_file.read()
                form_data.add_field('file', audio_content, filename='audio.wav')
            form_data.add_field('model', model)
            form_data.add_field('language', language)
            form_data.add_field('response_format', 'json')

            # Make API request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                for attempt in range(self.max_retries):
                    try:
                        async with session.post(
                            f"{self.base_url}/audio/transcriptions",
                            data=form_data,
                            headers=self.headers
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                return {
                                    "text": result.get("text", ""),
                                    "language": language,
                                    "model": model,
                                    "provider": "groq"
                                }
                            else:
                                error_text = await response.text()
                                if attempt < self.max_retries - 1:
                                    await asyncio.sleep(1)
                                    continue
                                else:
                                    raise HTTPException(
                                        status_code=response.status,
                                        detail=f"Groq API error: {error_text}"
                                    )
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        else:
                            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass  # Ignore cleanup errors
