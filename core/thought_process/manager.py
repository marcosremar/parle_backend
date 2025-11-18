"""
Thought Process Manager - Central Interface.

Provides a unified, easy-to-use interface for the entire Thought Process Framework.
This is the main entry point for all services that want to use thought processes.
"""

from typing import Dict, Any, Optional, List, Tuple
import json
from src.core.thought_process.prompt_manager import ThoughtProcessPromptManager
from src.core.thought_process.validator import ThoughtProcessValidator
from src.core.thought_process.structurer import ResponseMetadataStructurer
from src.core.thought_process.models import PromptConfig


class ThoughtProcessManager:
    """
    Central manager for the entire Thought Process Framework.

    This class provides a simple, unified interface for all thought process operations.
    Other services should use this manager instead of importing individual components.

    Example usage:
        manager = ThoughtProcessManager()

        # 1. Get customized prompt for LLM
        prompt = manager.get_prompt(language="portuguese", level="A2")

        # 2. After LLM generates response, structure it
        structured = manager.structure_response(
            llm_response="Você vai para a praia!",
            user_text="Eu vai para praia",
            metadata_json=llm_metadata
        )

        # 3. Add latency metrics
        structured = manager.add_latencies(
            response=structured,
            stt_time_ms=100,
            llm_time_ms=500,
            tts_time_ms=200
        )

        # 4. Validate the response
        is_valid, errors = manager.validate(structured)

        # 5. Convert to NDJSON for streaming
        ndjson = manager.to_ndjson(structured)
    """

    def __init__(self):
        """Initialize the manager with all required components."""
        self._prompt_manager = ThoughtProcessPromptManager()
        self._validator = ThoughtProcessValidator()
        self._structurer = ResponseMetadataStructurer()

    # ============================================================================
    # PROMPT MANAGEMENT
    # ============================================================================

    def get_prompt(
        self,
        language: str = "portuguese",
        level: str = "A2",
        region: Optional[str] = None,
        learning_context: Optional[str] = None,
        session_number: Optional[int] = None,
    ) -> str:
        """
        Get a customized system prompt for the LLM.

        This prompt instructs the LLM to generate all 9 thought processes.

        Args:
            language: Target language (portuguese, spanish, french, english, etc.)
            level: CEFR proficiency level (A1, A2, B1, B2, C1, C2)
            region: Optional region for dialect (brazil, portugal, spain, mexico)
            learning_context: Optional context (conversational, business, academic)
            session_number: Session number for progress tracking

        Returns:
            Customized system prompt string ready for LLM

        Example:
            prompt = manager.get_prompt(language="portuguese", level="B1")
            response = llm_service.generate(system_prompt=prompt, user_input=text)
        """
        return self._prompt_manager.get_prompt(
            language=language,
            level=level,
            region=region,
            learning_context=learning_context,
            session_number=session_number,
        )

    def get_config(
        self,
        language: str = "portuguese",
        level: str = "A2",
        region: Optional[str] = None,
        learning_context: Optional[str] = None,
        session_number: Optional[int] = None,
    ) -> PromptConfig:
        """
        Get structured configuration object.

        Args:
            language: Target language
            level: CEFR level
            region: Optional region
            learning_context: Optional context
            session_number: Session number

        Returns:
            PromptConfig Pydantic model with validated configuration
        """
        return self._prompt_manager.get_config(
            language=language,
            level=level,
            region=region,
            learning_context=learning_context,
            session_number=session_number,
        )

    def list_available_languages(self) -> List[str]:
        """List all available language templates."""
        return self._prompt_manager.list_available_languages()

    # ============================================================================
    # RESPONSE STRUCTURING
    # ============================================================================

    def structure_response(
        self,
        llm_response: str,
        user_text: str,
        metadata_json: Optional[Dict[str, Any]] = None,
        turn_number: int = 1,
        classification: str = "GOOD",
    ) -> Dict[str, Any]:
        """
        Structure a complete response with metadata.

        This creates the full response object that combines the main response
        with all thought process metadata.

        Args:
            llm_response: Main conversational response
            user_text: Original user input
            metadata_json: Thought process and hints metadata (from LLM)
            turn_number: Turn number in conversation
            classification: Performance classification (EXCELLENT/GOOD/ACCEPTABLE/SLOW)

        Returns:
            Complete structured response dict

        Example:
            response = manager.structure_response(
                llm_response="Você vai para a praia!",
                user_text="Eu vai para praia",
                metadata_json=llm_metadata,
                turn_number=1
            )
        """
        return self._structurer.structure(
            llm_response=llm_response,
            user_text=user_text,
            metadata_json=metadata_json,
            turn_number=turn_number,
            classification=classification,
        )

    def add_latencies(
        self,
        response: Dict[str, Any],
        stt_time_ms: float,
        llm_time_ms: float,
        tts_time_ms: float,
    ) -> Dict[str, Any]:
        """
        Add latency metrics to response.

        This adds performance measurements and updates classification based on
        perceived latency (LLM + TTS time).

        Args:
            response: Response dict from structure_response()
            stt_time_ms: Speech-to-text processing time
            llm_time_ms: LLM processing time
            tts_time_ms: Text-to-speech processing time

        Returns:
            Response with latency metrics and updated classification

        Example:
            response = manager.add_latencies(
                response=response,
                stt_time_ms=100,
                llm_time_ms=500,
                tts_time_ms=200
            )
            # Classification is now auto-updated based on latency
        """
        return self._structurer.add_latency_metrics(
            response=response,
            stt_time_ms=stt_time_ms,
            llm_time_ms=llm_time_ms,
            tts_time_ms=tts_time_ms,
        )

    def create_session_summary(
        self,
        responses: List[Dict[str, Any]],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Create a summary of the session from multiple responses.

        Aggregates metrics, latencies, and classifications across all turns.

        Args:
            responses: List of response dicts from previous turns
            session_id: Session identifier

        Returns:
            Session summary with aggregated statistics

        Example:
            summary = manager.create_session_summary(
                responses=all_turn_responses,
                session_id="session_xyz"
            )
        """
        return self._structurer.add_session_summary(
            responses=responses,
            session_id=session_id,
        )

    # ============================================================================
    # VALIDATION
    # ============================================================================

    def validate(
        self,
        response: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a response has all required 9 thought processes.

        Checks:
        - Main response fields (turn_number, timestamp, user_text, llm_response)
        - Response metadata structure
        - All 9 thought processes with correct names and IDs
        - Latency metrics with valid values

        Args:
            response: Response dict to validate

        Returns:
            Tuple of (is_valid, error_list)

        Example:
            is_valid, errors = manager.validate(response)
            if not is_valid:
                print("Validation errors:", errors)
        """
        is_valid = self._validator.validate(response)
        errors = self._validator.get_errors()
        return is_valid, errors

    def validate_from_json(
        self,
        json_string: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a response from JSON string.

        Args:
            json_string: JSON response as string

        Returns:
            Tuple of (is_valid, error_list)

        Example:
            is_valid, errors = manager.validate_from_json(json_str)
        """
        return self._validator.validate_from_json_string(json_string)

    # ============================================================================
    # SERIALIZATION & STREAMING
    # ============================================================================

    def to_ndjson(self, response: Dict[str, Any]) -> str:
        """
        Convert response to NDJSON format for streaming.

        NDJSON (Newline-Delimited JSON) is used for streaming responses
        to the client without buffering the entire response.

        Args:
            response: Response dict

        Returns:
            Single-line JSON string (no newlines)

        Example:
            ndjson_line = manager.to_ndjson(response)
            # Send to client over WebSocket/HTTP stream
        """
        return self._structurer.to_ndjson(response)

    def from_ndjson(self, ndjson_line: str) -> Dict[str, Any]:
        """
        Parse NDJSON line back to response dict.

        Args:
            ndjson_line: Single NDJSON line

        Returns:
            Parsed response dict

        Example:
            response = manager.from_ndjson(ndjson_line)
        """
        return self._structurer.from_ndjson(ndjson_line)

    # ============================================================================
    # CONVENIENCE METHODS
    # ============================================================================

    def get_process_names(self) -> Dict[int, str]:
        """
        Get mapping of process IDs to names.

        Returns:
            Dict mapping process ID (1-9) to process name

        Example:
            names = manager.get_process_names()
            # {1: "Error Detection", 2: "Grammar Analysis", ...}
        """
        return self._validator.REQUIRED_PROCESSES.copy()

    def get_latency_classification(self, perceived_latency_ms: float) -> str:
        """
        Get performance classification for a latency value.

        Args:
            perceived_latency_ms: Perceived latency (LLM + TTS time)

        Returns:
            Classification: EXCELLENT (<500ms), GOOD (<800ms), ACCEPTABLE (<1200ms), SLOW (>=1200ms)

        Example:
            classification = manager.get_latency_classification(600)
            # "GOOD"
        """
        return self._structurer._classify_latency(perceived_latency_ms)

    def extract_thought_processes(
        self,
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract just the thought process data from a response.

        Args:
            response: Full response dict

        Returns:
            Thought process metadata (processes array, timings, etc.)

        Example:
            tp = manager.extract_thought_processes(response)
            for process in tp['processes']:
                print(f"Process {process['id']}: {process['name']}")
        """
        return self._structurer.extract_thought_processes(response)

    def extract_hints(
        self,
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract hints for next turn from a response.

        Args:
            response: Full response dict

        Returns:
            Hints metadata with suggestions for next input

        Example:
            hints = manager.extract_hints(response)
            print("Next hints:", hints['hints'])
        """
        return self._structurer.extract_hints(response)

    # ============================================================================
    # COMPLETE WORKFLOW (High-level convenience)
    # ============================================================================

    def process_student_input(
        self,
        user_text: str,
        llm_response: str,
        llm_metadata: Optional[Dict[str, Any]],
        language: str = "portuguese",
        level: str = "A2",
        stt_time_ms: float = 0,
        llm_time_ms: float = 0,
        tts_time_ms: float = 0,
        turn_number: int = 1,
    ) -> Tuple[Dict[str, Any], bool, List[str]]:
        """
        Complete end-to-end processing of student input.

        This high-level method handles the entire workflow:
        1. Structure the response
        2. Add latency metrics
        3. Validate all requirements
        4. Return structured response and validation status

        Args:
            user_text: Original student input
            llm_response: LLM's conversational response
            llm_metadata: Response metadata with 9 thought processes (from LLM)
            language: Student's target language
            level: Student's proficiency level
            stt_time_ms: Speech-to-text time (if using STT)
            llm_time_ms: LLM processing time
            tts_time_ms: Text-to-speech time
            turn_number: Turn in conversation

        Returns:
            Tuple of (structured_response, is_valid, validation_errors)

        Example:
            response, is_valid, errors = manager.process_student_input(
                user_text="Eu vai para praia",
                llm_response="Você vai para a praia!",
                llm_metadata={"thought_process": {...}, ...},
                language="portuguese",
                level="A2",
                stt_time_ms=100,
                llm_time_ms=500,
                tts_time_ms=200,
                turn_number=1
            )

            if is_valid:
                # Send response to student
                ndjson = manager.to_ndjson(response)
            else:
                # Log validation errors
                print("Validation failed:", errors)
        """
        # 1. Structure the response
        structured = self.structure_response(
            llm_response=llm_response,
            user_text=user_text,
            metadata_json=llm_metadata,
            turn_number=turn_number,
        )

        # 2. Add latency metrics
        structured = self.add_latencies(
            response=structured,
            stt_time_ms=stt_time_ms,
            llm_time_ms=llm_time_ms,
            tts_time_ms=tts_time_ms,
        )

        # 3. Validate
        is_valid, errors = self.validate(structured)

        return structured, is_valid, errors

    # ============================================================================
    # INFORMATION & DOCUMENTATION
    # ============================================================================

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the Thought Process Framework.

        Returns:
            Dict with framework info, version, processes, etc.
        """
        return {
            "name": "Thought Process Framework",
            "version": "1.0.0",
            "description": "Language teaching AI with transparent 9-process reasoning",
            "num_processes": 9,
            "process_names": self.get_process_names(),
            "supported_languages": self.list_available_languages() or ["portuguese", "english"],
            "cefr_levels": ["A1", "A2", "B1", "B2", "C1", "C2"],
            "latency_classifications": {
                "EXCELLENT": "< 500ms",
                "GOOD": "< 800ms",
                "ACCEPTABLE": "< 1200ms",
                "SLOW": ">= 1200ms",
            },
        }
