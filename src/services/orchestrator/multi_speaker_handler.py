"""
Multi-Speaker Handler for Orchestrator

Handles speaker diarization and voice mapping for multi-speaker conversations.
Integrates with diarization service to identify speakers and map them to appropriate TTS voices.
"""

from typing import Dict, List, Optional, Any
from loguru import logger


class MultiSpeakerHandler:
    """
    Handles multi-speaker logic for orchestrator.

    Responsibilities:
    - Call diarization service to identify speakers
    - Map speakers to TTS voices
    - Track speaker assignments in session
    - Merge speaker segments with transcriptions
    """

    def __init__(self, comm_manager, session_client, tts_client):
        """
        Initialize multi-speaker handler.

        Args:
            comm_manager: Communication manager for service calls
            session_client: Session service client
            tts_client: TTS service client
        """
        self.comm = comm_manager
        self.session_client = session_client
        self.tts_client = tts_client

        # Default voice mappings (fallback)
        self.default_voices = {
            "SPEAKER_00": "af_heart",  # Female voice 1
            "SPEAKER_01": "am_adam",  # Male voice 1
            "SPEAKER_02": "af_nicole",  # Female voice 2
            "SPEAKER_03": "am_alex",  # Male voice 2
            "SPEAKER_04": "af_alloy",  # Female voice 3
        }

    async def diarize_audio(
        self, audio: str, encoding: str = "base64", num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call diarization service to identify speakers in audio.

        Args:
            audio: Audio data (base64/hex/URL)
            encoding: Audio encoding method
            num_speakers: Expected number of speakers (optional)

        Returns:
            Diarization response with speaker segments

        Example response:
            {
                "success": true,
                "speakers": [
                    {"speaker_id": "SPEAKER_00", "start": 0.5, "end": 3.2},
                    {"speaker_id": "SPEAKER_01", "start": 3.5, "end": 6.8}
                ],
                "stats": {"num_speakers": 2, ...},
                "processing_time": 1.23
            }
        """
        try:
            logger.info(f"Calling diarization service (num_speakers={num_speakers})")

            # Call diarization service
            response = await self.comm.call_service(
                service_name="diarization",
                endpoint="/diarize",
                method="POST",
                data={
                    "audio": audio,
                    "encoding": encoding,
                    "num_speakers": num_speakers,
                    "include_timestamps": True,
                    "merge_same_speaker": True,
                },
            )

            if not response.get("success"):
                logger.error(f"Diarization failed: {response.get('error')}")
                return None

            logger.info(
                f"Diarization successful: {response['stats']['num_speakers']} speakers, "
                f"{len(response['speakers'])} segments"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to call diarization service: {e}")
            return None

    async def get_or_assign_voice(
        self, session_id: str, speaker_id: str, auto_assign: bool = True
    ) -> Optional[str]:
        """
        Get TTS voice for a speaker from session, or auto-assign if not set.

        Args:
            session_id: Session ID
            speaker_id: Speaker identifier (e.g., 'SPEAKER_00')
            auto_assign: Auto-assign voice if not found

        Returns:
            Voice ID (e.g., 'af_heart') or None
        """
        try:
            # Get session to check existing speaker mappings
            session = await self.session_client.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found")
                return None

            # Check if speaker already has a voice assigned
            speakers = session.get("speakers", {})
            if speaker_id in speakers:
                voice_id = speakers[speaker_id]
                logger.debug(f"Speaker {speaker_id} already mapped to {voice_id}")
                return voice_id

            # Auto-assign voice if enabled
            if auto_assign:
                voice_id = self.default_voices.get(speaker_id, "af_heart")
                logger.info(f"Auto-assigning voice {voice_id} to {speaker_id}")

                # Update session with new speaker mapping
                speakers[speaker_id] = voice_id
                await self.session_client.update_session(
                    session_id, update_data={"speakers": speakers}
                )

                return voice_id

            return None

        except Exception as e:
            logger.error(f"Failed to get/assign voice for speaker {speaker_id}: {e}")
            return None

    async def process_multi_speaker_turn(
        self, audio: str, session_id: str, encoding: str = "base64"
    ) -> Dict[str, Any]:
        """
        Process a multi-speaker audio turn.

        Workflow:
        1. Diarize audio to identify speakers
        2. For each speaker segment:
           - Transcribe segment (call STT)
           - Get/assign TTS voice for speaker
           - Save message with speaker_id
        3. Generate LLM response (assistant)
        4. Synthesize response with appropriate voice
        5. Return processed turn with speaker info

        Args:
            audio: Audio data (base64/hex/URL)
            session_id: Session ID
            encoding: Audio encoding method

        Returns:
            Processed turn with speaker information

        Example return:
            {
                "speakers": [
                    {
                        "speaker_id": "SPEAKER_00",
                        "text": "Hello, how are you?",
                        "voice_id": "af_heart",
                        "start": 0.5,
                        "end": 3.2
                    }
                ],
                "response": {
                    "text": "I'm doing well, thank you!",
                    "audio": "base64...",
                    "voice_id": "am_adam"
                }
            }
        """
        try:
            logger.info(f"Processing multi-speaker turn for session {session_id}")

            # Step 1: Diarize audio
            diarization = await self.diarize_audio(audio, encoding)
            if not diarization:
                logger.warning("Diarization failed, falling back to single-speaker mode")
                return None

            # Step 2: Process each speaker segment
            speakers_info = []
            for segment in diarization["speakers"]:
                speaker_id = segment["speaker_id"]
                start = segment["start"]
                end = segment["end"]

                # Get/assign voice for this speaker
                voice_id = await self.get_or_assign_voice(session_id, speaker_id)

                speakers_info.append(
                    {
                        "speaker_id": speaker_id,
                        "voice_id": voice_id,
                        "start": start,
                        "end": end,
                        # Note: Transcription would be done separately per segment
                        # This is a placeholder for future integration
                        "text": None,  # Would be filled by STT per segment
                    }
                )

            result = {
                "diarization": diarization,
                "speakers": speakers_info,
                "num_speakers": diarization["stats"]["num_speakers"],
            }

            logger.info(f"Multi-speaker turn processed: {result['num_speakers']} speakers")
            return result

        except Exception as e:
            logger.error(f"Failed to process multi-speaker turn: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def synthesize_with_speaker_voice(
        self, text: str, session_id: str, speaker_id: str = "assistant"
    ) -> Optional[str]:
        """
        Synthesize text with appropriate voice for speaker.

        Args:
            text: Text to synthesize
            session_id: Session ID
            speaker_id: Speaker identifier (default: 'assistant')

        Returns:
            Audio data (base64) or None
        """
        try:
            # Get voice for speaker
            voice_id = await self.get_or_assign_voice(session_id, speaker_id)
            if not voice_id:
                logger.warning(f"No voice found for speaker {speaker_id}, using default")
                voice_id = "af_heart"

            logger.info(f"Synthesizing with voice {voice_id} for speaker {speaker_id}")

            # Call TTS service
            audio = await self.tts_client.synthesize(text, voice_id=voice_id)

            return audio

        except Exception as e:
            logger.error(f"Failed to synthesize with speaker voice: {e}")
            return None

    def get_default_voice_for_speaker(self, speaker_index: int) -> str:
        """
        Get default voice for a speaker by index.

        Args:
            speaker_index: Speaker index (0, 1, 2, ...)

        Returns:
            Voice ID
        """
        speaker_id = f"SPEAKER_{speaker_index:02d}"
        return self.default_voices.get(speaker_id, "af_heart")
