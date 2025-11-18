"""
Response Metadata Structurer.

Structures and formats response metadata from LLM output.
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime


class ResponseMetadataStructurer:
    """Structures response metadata into proper format."""

    def structure(
        self,
        llm_response: str,
        user_text: str,
        metadata_json: Optional[Dict[str, Any]] = None,
        turn_number: int = 1,
        classification: str = "GOOD",
    ) -> Dict[str, Any]:
        """
        Structure complete response with metadata.

        Args:
            llm_response: Main conversational response
            user_text: Original user input
            metadata_json: Thought process and hints metadata
            turn_number: Turn number in conversation
            classification: Performance classification (EXCELLENT/GOOD/ACCEPTABLE/SLOW)

        Returns:
            Structured response dict
        """
        response = {
            "turn_number": turn_number,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_text": user_text,
            "llm_response": llm_response,
            "response_metadata": metadata_json or self._default_metadata(),
            "classification": classification,
        }

        return response

    def _default_metadata(self) -> Dict[str, Any]:
        """Get default metadata structure."""
        return {
            "thought_process": {
                "processes": [],
                "total_processes": 0,
                "total_generation_time_ms": 0.0,
            },
            "hints_for_next_turn": {
                "hints": [],
                "count": 0,
                "generation_time_ms": 0.0,
            },
        }

    def extract_thought_processes(
        self, metadata_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract thought processes from metadata.

        Args:
            metadata_json: Metadata dict

        Returns:
            Thought process data
        """
        if "response_metadata" in metadata_json:
            return metadata_json["response_metadata"].get("thought_process", {})
        elif "thought_process" in metadata_json:
            return metadata_json["thought_process"]
        return {}

    def extract_hints(self, metadata_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hints from metadata."""
        if "response_metadata" in metadata_json:
            return metadata_json["response_metadata"].get("hints_for_next_turn", {})
        elif "hints_for_next_turn" in metadata_json:
            return metadata_json["hints_for_next_turn"]
        return {}

    def add_latency_metrics(
        self,
        response: Dict[str, Any],
        stt_time_ms: float,
        llm_time_ms: float,
        tts_time_ms: float,
    ) -> Dict[str, Any]:
        """
        Add latency metrics to response.

        Args:
            response: Response dict
            stt_time_ms: Speech-to-text processing time
            llm_time_ms: LLM processing time
            tts_time_ms: Text-to-speech processing time

        Returns:
            Response with latency metrics added
        """
        total_time = stt_time_ms + llm_time_ms + tts_time_ms
        perceived_latency = llm_time_ms + tts_time_ms  # What user perceives

        response["latencies"] = {
            "stt_time_ms": stt_time_ms,
            "llm_time_ms": llm_time_ms,
            "tts_time_ms": tts_time_ms,
            "total_time_ms": total_time,
            "perceived_latency_ms": perceived_latency,
        }

        # Update classification based on perceived latency
        response["classification"] = self._classify_latency(perceived_latency)

        return response

    def _classify_latency(self, perceived_latency_ms: float) -> str:
        """
        Classify latency performance.

        Args:
            perceived_latency_ms: Perceived latency in milliseconds

        Returns:
            Classification string
        """
        if perceived_latency_ms < 500:
            return "EXCELLENT"
        elif perceived_latency_ms < 800:
            return "GOOD"
        elif perceived_latency_ms < 1200:
            return "ACCEPTABLE"
        else:
            return "SLOW"

    def add_session_summary(
        self,
        responses: list,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Create session summary from multiple responses.

        Args:
            responses: List of response dicts
            session_id: Session ID

        Returns:
            Session summary dict
        """
        if not responses:
            return {"session_id": session_id, "total_turns": 0}

        latencies = [r.get("latencies", {}).get("perceived_latency_ms", 0) for r in responses]
        classifications = [r.get("classification", "UNKNOWN") for r in responses]

        summary = {
            "session_id": session_id,
            "total_turns": len(responses),
            "average_perceived_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "min_perceived_latency_ms": min(latencies) if latencies else 0,
            "max_perceived_latency_ms": max(latencies) if latencies else 0,
            "classifications": {
                "excellent": classifications.count("EXCELLENT"),
                "good": classifications.count("GOOD"),
                "acceptable": classifications.count("ACCEPTABLE"),
                "slow": classifications.count("SLOW"),
            },
        }

        return summary

    def to_ndjson(self, response: Dict[str, Any]) -> str:
        """
        Convert response to NDJSON format.

        Args:
            response: Response dict

        Returns:
            JSON string (single line)
        """
        return json.dumps(response, ensure_ascii=False)

    def from_ndjson(self, ndjson_line: str) -> Dict[str, Any]:
        """
        Parse NDJSON line to response dict.

        Args:
            ndjson_line: Single NDJSON line

        Returns:
            Parsed response dict
        """
        return json.loads(ndjson_line)
