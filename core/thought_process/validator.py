"""
Thought Process Validator.

Validates that LLM responses contain all 7 thought processes (optimized for speech-only learning).
Removed: Error Detection (Process 1) and Pronunciation Evaluation (Process 6).
"""

from typing import Dict, Any, List, Tuple
import json


class ThoughtProcessValidator:
    """Validates response structure and content (7 processes for speech-only learning)."""

    REQUIRED_PROCESSES = {
        1: "Grammar Analysis",
        2: "Vocabulary Assessment",
        3: "Pedagogical Strategy",
        4: "Conversation Flow",
        5: "Learning Progress",
        6: "Cultural Context",
        7: "Learning Recommendation",
    }

    def __init__(self):
        """Initialize validator."""
        self.errors: List[str] = []

    def validate(self, response: Dict[str, Any]) -> bool:
        """
        Validate response structure.

        Args:
            response: LLM response dict

        Returns:
            True if valid, False otherwise
        """
        self.errors = []

        # Check main structure
        if not self._validate_main_response(response):
            return False

        # Check metadata
        if not self._validate_response_metadata(response):
            return False

        # Check latencies
        if not self._validate_latencies(response):
            return False

        # Check all 7 processes
        if not self._validate_all_processes(response):
            return False

        return len(self.errors) == 0

    def _validate_main_response(self, response: Dict[str, Any]) -> bool:
        """Validate main response fields."""
        required_fields = ["turn_number", "timestamp", "user_text", "llm_response"]

        for field in required_fields:
            if field not in response:
                self.errors.append(f"Missing required field: {field}")
                return False

        if not isinstance(response["llm_response"], str) or not response["llm_response"]:
            self.errors.append("llm_response must be non-empty string")
            return False

        return True

    def _validate_response_metadata(self, response: Dict[str, Any]) -> bool:
        """Validate response metadata structure."""
        if "response_metadata" not in response:
            self.errors.append("Missing response_metadata")
            return False

        metadata = response["response_metadata"]

        if "thought_process" not in metadata:
            self.errors.append("Missing thought_process in metadata")
            return False

        if "hints_for_next_turn" not in metadata:
            self.errors.append("Missing hints_for_next_turn in metadata")
            return False

        # Validate thought_process structure
        tp = metadata["thought_process"]
        if "processes" not in tp:
            self.errors.append("Missing processes array in thought_process")
            return False

        if "total_processes" not in tp or tp["total_processes"] != 7:
            self.errors.append(f"total_processes must be 7, got {tp.get('total_processes')}")
            return False

        return True

    def _validate_latencies(self, response: Dict[str, Any]) -> bool:
        """Validate latency metrics."""
        if "latencies" not in response:
            self.errors.append("Missing latencies")
            return False

        latencies = response["latencies"]
        required_fields = ["stt_time_ms", "llm_time_ms", "tts_time_ms", "perceived_latency_ms"]

        for field in required_fields:
            if field not in latencies:
                self.errors.append(f"Missing latency field: {field}")
                return False

            if not isinstance(latencies[field], (int, float)) or latencies[field] < 0:
                self.errors.append(f"{field} must be non-negative number")
                return False

        return True

    def _validate_all_processes(self, response: Dict[str, Any]) -> bool:
        """Validate all 7 processes are present."""
        metadata = response.get("response_metadata", {})
        thought_process = metadata.get("thought_process", {})
        processes = thought_process.get("processes", [])

        if len(processes) != 7:
            self.errors.append(f"Expected 7 processes, got {len(processes)}")
            return False

        # Check each process
        process_ids = set()
        for i, process in enumerate(processes):
            if not isinstance(process, dict):
                self.errors.append(f"Process {i} is not a dict")
                return False

            # Validate process ID
            if "id" not in process:
                self.errors.append(f"Process {i} missing id")
                return False

            process_id = process["id"]
            if process_id not in range(1, 8):
                self.errors.append(f"Invalid process id: {process_id}")
                return False

            process_ids.add(process_id)

            # Validate process name
            if "name" not in process:
                self.errors.append(f"Process {process_id} missing name")
                return False

            expected_name = self.REQUIRED_PROCESSES.get(process_id)
            if process["name"] != expected_name:
                self.errors.append(
                    f"Process {process_id} name mismatch: "
                    f"expected '{expected_name}', got '{process['name']}'"
                )

            # Validate content
            if "content" not in process:
                self.errors.append(f"Process {process_id} missing content")
                return False

            if not isinstance(process["content"], dict):
                self.errors.append(f"Process {process_id} content must be dict")
                return False

            # Validate generation_time_ms
            if "generation_time_ms" not in process:
                self.errors.append(f"Process {process_id} missing generation_time_ms")
                return False

        # Verify all 7 processes are present (no duplicates or missing)
        if process_ids != set(range(1, 8)):
            missing = set(range(1, 8)) - process_ids
            if missing:
                self.errors.append(f"Missing processes: {missing}")
            return False

        return True

    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.errors

    def print_errors(self) -> None:
        """Print all validation errors."""
        if not self.errors:
            print("✅ Validation passed!")
            return

        print("❌ Validation failed:")
        for i, error in enumerate(self.errors, 1):
            print(f"  {i}. {error}")

    def validate_from_json_string(self, json_string: str) -> Tuple[bool, List[str]]:
        """
        Validate from JSON string.

        Args:
            json_string: JSON response as string

        Returns:
            Tuple of (is_valid, error_list)
        """
        try:
            response = json.loads(json_string)
            is_valid = self.validate(response)
            return is_valid, self.errors
        except json.JSONDecodeError as e:
            self.errors = [f"Invalid JSON: {str(e)}"]
            return False, self.errors
