#!/usr/bin/env python3
"""
Audio conversion endpoint for converting WebM to WAV
"""

import asyncio
import tempfile
import os
import numpy as np
import wave
from aiohttp import web
import logging

logger = logging.getLogger(__name__)

async def convert_audio_handler(request):
    """Convert WebM audio to WAV format with optional amplification"""
    try:
        # Read multipart form data
        reader = await request.multipart()
        audio_file = None
        output_format = 'wav'
        amplify = False

        # Parse form data
        async for field in reader:
            if field.name == 'audio':
                # Save uploaded file to temp location
                audio_data = await field.read()
                audio_file = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
                audio_file.write(audio_data)
                audio_file.flush()
                audio_file.close()

            elif field.name == 'output_format':
                output_format = (await field.read()).decode('utf-8')

            elif field.name == 'amplify':
                amplify = (await field.read()).decode('utf-8').lower() == 'true'

        if not audio_file:
            return web.json_response({'error': 'No audio file provided'}, status=400)

        # Convert WebM to WAV using ffmpeg
        output_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_file.close()

        # Use ffmpeg to convert and resample to 16kHz
        cmd = f'ffmpeg -i {audio_file.name} -ar 16000 -ac 1 -f wav {output_file.name} -y'

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {stderr.decode()}")
            return web.json_response({'error': 'Audio conversion failed'}, status=500)

        # If amplification requested, process the WAV file
        if amplify:
            # Read WAV file
            with wave.open(output_file.name, 'rb') as wav_in:
                params = wav_in.getparams()
                frames = wav_in.readframes(params.nframes)

            # Convert to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16)

            # Find max amplitude
            max_amplitude = np.max(np.abs(audio_data))

            if max_amplitude > 0:
                # Calculate amplification factor (normalize to 90% of max)
                target_amplitude = int(32767 * 0.9)
                amplification_factor = min(target_amplitude / max_amplitude, 10)

                # Apply amplification
                audio_data = np.clip(audio_data * amplification_factor, -32768, 32767).astype(np.int16)

                # Write back to file
                with wave.open(output_file.name, 'wb') as wav_out:
                    wav_out.setparams(params)
                    wav_out.writeframes(audio_data.tobytes())

                logger.info(f"Audio amplified by factor: {amplification_factor:.2f}")

        # Read the converted file
        with open(output_file.name, 'rb') as f:
            wav_data = f.read()

        # Cleanup temp files
        try:
            os.unlink(audio_file.name)
            os.unlink(output_file.name)
        except OSError:
            # Temp files already deleted or not accessible
            pass

        # Return WAV file
        return web.Response(
            body=wav_data,
            content_type='audio/wav',
            headers={
                'Content-Disposition': 'attachment; filename="converted.wav"'
            }
        )

    except Exception as e:
        logger.error(f"Error in audio conversion: {e}", exc_info=True)
        return web.json_response({'error': str(e)}, status=500)

def setup_audio_converter_routes(app):
    """Setup audio converter routes"""
    app.router.add_post('/api/convert_audio', convert_audio_handler)
    logger.info("Audio converter endpoint registered at /api/convert_audio")