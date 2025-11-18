"""
Turn Manager Utility

Manages turn-taking in group conversations with queue-based allocation,
timeouts, and different turn modes (sequential, simultaneous, free).
"""

from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.group_session.models import (
    Participant,
    GroupSession,
    TurnMode,
)


class TurnPriority(str, Enum):
    """Turn priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TurnRequest:
    """Represents a turn request with metadata"""

    def __init__(
        self,
        participant_id: str,
        priority: TurnPriority = TurnPriority.NORMAL,
        requested_at: datetime = None
    ):
        self.participant_id = participant_id
        self.priority = priority
        self.requested_at = requested_at or datetime.now()


class TurnManager:
    """
    Manages turn-taking in group conversations

    Responsibilities:
    - Queue-based turn allocation
    - Turn timeouts and expiration
    - Priority-based queueing
    - Simultaneous vs sequential mode handling
    - Turn statistics and monitoring
    """

    def __init__(
        self,
        turn_timeout_seconds: int = 60,
        max_queue_size: int = 50
    ):
        """
        Initialize TurnManager

        Args:
            turn_timeout_seconds: Seconds before a turn expires
            max_queue_size: Maximum number of participants in queue
        """
        self._logger = logger
        self.turn_timeout_seconds = turn_timeout_seconds
        self.max_queue_size = max_queue_size

        # Statistics
        self._total_turns_processed = 0
        self._turn_start_times: Dict[str, datetime] = {}

    # ========================================================================
    # Turn Request Management
    # ========================================================================

    def can_request_turn(
        self,
        session: GroupSession,
        participant: Participant
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if participant can request a turn

        Args:
            session: Group session
            participant: Participant requesting turn

        Returns:
            (can_request: bool, reason: Optional[str])
        """
        # Check if participant can send
        if not participant.can_send:
            return False, "Participant does not have send permission"

        # Check if participant is active
        if not participant.is_active():
            return False, f"Participant is {participant.status.value}"

        # Check if queue is full
        if len(session.turn_queue) >= self.max_queue_size:
            return False, f"Turn queue is full ({self.max_queue_size} max)"

        # Check if already in queue
        if participant.participant_id in session.turn_queue:
            return False, "Participant already in turn queue"

        # Check if currently speaking
        if session.current_speaker_id == participant.participant_id:
            return False, "Participant is already speaking"

        return True, None

    def request_turn(
        self,
        session: GroupSession,
        participant: Participant,
        priority: TurnPriority = TurnPriority.NORMAL
    ) -> Tuple[bool, Optional[str], int]:
        """
        Request a turn for participant

        Args:
            session: Group session
            participant: Participant requesting turn
            priority: Turn priority (default: NORMAL)

        Returns:
            (success: bool, error: Optional[str], queue_position: int)
        """
        # Validate can request
        can_request, reason = self.can_request_turn(session, participant)
        if not can_request:
            return False, reason, -1

        participant_id = participant.participant_id

        # Add to queue based on priority
        if priority == TurnPriority.HIGH:
            # High priority - add to front (after current speaker)
            session.turn_queue.insert(0, participant_id)
            position = 1
        elif priority == TurnPriority.LOW:
            # Low priority - add to back
            session.turn_queue.append(participant_id)
            position = len(session.turn_queue)
        else:
            # Normal priority - add to back
            session.turn_queue.append(participant_id)
            position = len(session.turn_queue)

        session.last_activity_at = datetime.now()

        self._logger.info(
            f"🎤 Participant {participant.display_name} requested turn "
            f"(priority: {priority.value}, position: {position})"
        )

        return True, None, position

    def cancel_turn_request(
        self,
        session: GroupSession,
        participant_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Cancel a turn request

        Args:
            session: Group session
            participant_id: Participant ID to remove from queue

        Returns:
            (success: bool, error: Optional[str])
        """
        if participant_id not in session.turn_queue:
            return False, "Participant not in turn queue"

        session.turn_queue.remove(participant_id)
        session.last_activity_at = datetime.now()

        self._logger.info(f"🎤 Turn request cancelled for participant {participant_id}")

        return True, None

    # ========================================================================
    # Turn Allocation
    # ========================================================================

    def next_turn(self, session: GroupSession) -> Optional[str]:
        """
        Allocate next turn from queue

        Args:
            session: Group session

        Returns:
            participant_id of next speaker (None if queue empty)
        """
        # Check if queue is empty
        if not session.turn_queue:
            session.current_speaker_id = None
            return None

        # Get next participant
        next_speaker_id = session.turn_queue.pop(0)
        session.current_speaker_id = next_speaker_id
        session.last_activity_at = datetime.now()

        # Record turn start time
        self._turn_start_times[next_speaker_id] = datetime.now()
        self._total_turns_processed += 1

        self._logger.info(
            f"🎤 Next turn: {next_speaker_id} "
            f"(queue: {len(session.turn_queue)} remaining)"
        )

        return next_speaker_id

    def end_turn(self, session: GroupSession) -> Tuple[bool, Optional[str]]:
        """
        End the current turn

        Args:
            session: Group session

        Returns:
            (success: bool, error: Optional[str])
        """
        if session.current_speaker_id is None:
            return False, "No active speaker"

        speaker_id = session.current_speaker_id

        # Calculate turn duration
        if speaker_id in self._turn_start_times:
            start_time = self._turn_start_times[speaker_id]
            duration = (datetime.now() - start_time).total_seconds()
            self._logger.debug(f"🎤 Turn ended: {speaker_id} (duration: {duration:.1f}s)")
            del self._turn_start_times[speaker_id]

        session.current_speaker_id = None
        session.last_activity_at = datetime.now()

        return True, None

    def force_end_turn(
        self,
        session: GroupSession,
        reason: str = "Turn timeout"
    ) -> Tuple[bool, Optional[str]]:
        """
        Forcefully end current turn (e.g., timeout)

        Args:
            session: Group session
            reason: Reason for forced end

        Returns:
            (success: bool, error: Optional[str])
        """
        if session.current_speaker_id is None:
            return False, "No active speaker"

        speaker_id = session.current_speaker_id

        self._logger.warning(
            f"🎤 Turn force-ended: {speaker_id} (reason: {reason})"
        )

        return self.end_turn(session)

    # ========================================================================
    # Turn Mode Handling
    # ========================================================================

    def should_queue_request(
        self,
        session: GroupSession,
        turn_mode: TurnMode
    ) -> bool:
        """
        Determine if turn request should be queued based on mode

        Args:
            session: Group session
            turn_mode: Current turn mode

        Returns:
            True if should queue, False if allow immediate processing
        """
        if turn_mode == TurnMode.SEQUENTIAL:
            # Queue if someone is speaking or processing
            return session.current_speaker_id is not None or session.is_processing

        elif turn_mode == TurnMode.SIMULTANEOUS:
            # Queue if processing (allow multiple speakers, but sequential processing)
            return session.is_processing

        elif turn_mode == TurnMode.FREE:
            # Never queue - allow chaos
            return False

        return True  # Default: queue

    def can_process_immediately(
        self,
        session: GroupSession,
        turn_mode: TurnMode
    ) -> bool:
        """
        Check if next turn can be processed immediately

        Args:
            session: Group session
            turn_mode: Current turn mode

        Returns:
            True if can process now, False if must wait
        """
        if turn_mode == TurnMode.SEQUENTIAL:
            # Can process if no one speaking and not processing
            return session.current_speaker_id is None and not session.is_processing

        elif turn_mode == TurnMode.SIMULTANEOUS:
            # Can process if not currently processing (allows overlap)
            return not session.is_processing

        elif turn_mode == TurnMode.FREE:
            # Always can process
            return True

        return False  # Default: cannot process

    # ========================================================================
    # Turn Timeout Management
    # ========================================================================

    def is_turn_expired(self, session: GroupSession) -> bool:
        """
        Check if current turn has expired

        Args:
            session: Group session

        Returns:
            True if turn expired, False otherwise
        """
        if session.current_speaker_id is None:
            return False

        speaker_id = session.current_speaker_id
        if speaker_id not in self._turn_start_times:
            return False

        start_time = self._turn_start_times[speaker_id]
        elapsed = (datetime.now() - start_time).total_seconds()

        return elapsed > self.turn_timeout_seconds

    def check_and_expire_turn(self, session: GroupSession) -> bool:
        """
        Check and expire current turn if timeout exceeded

        Args:
            session: Group session

        Returns:
            True if turn was expired, False otherwise
        """
        if self.is_turn_expired(session):
            speaker_id = session.current_speaker_id
            self._logger.warning(
                f"🎤 Turn expired for {speaker_id} "
                f"(timeout: {self.turn_timeout_seconds}s)"
            )
            self.force_end_turn(session, reason="Turn timeout")
            return True

        return False

    # ========================================================================
    # Queue Management
    # ========================================================================

    def get_queue_position(
        self,
        session: GroupSession,
        participant_id: str
    ) -> int:
        """
        Get participant's position in turn queue

        Args:
            session: Group session
            participant_id: Participant ID

        Returns:
            Queue position (1-indexed), or -1 if not in queue
        """
        try:
            return session.turn_queue.index(participant_id) + 1
        except ValueError:
            return -1

    def clear_queue(self, session: GroupSession):
        """Clear entire turn queue"""
        cleared_count = len(session.turn_queue)
        session.turn_queue = []
        session.last_activity_at = datetime.now()

        self._logger.info(f"🎤 Turn queue cleared ({cleared_count} requests removed)")

    def remove_participant_from_queue(
        self,
        session: GroupSession,
        participant_id: str
    ) -> bool:
        """
        Remove participant from queue

        Args:
            session: Group session
            participant_id: Participant to remove

        Returns:
            True if removed, False if not in queue
        """
        if participant_id in session.turn_queue:
            session.turn_queue.remove(participant_id)
            session.last_activity_at = datetime.now()
            self._logger.debug(f"🎤 Removed {participant_id} from turn queue")
            return True
        return False

    def reorder_queue(
        self,
        session: GroupSession,
        new_order: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Reorder turn queue

        Args:
            session: Group session
            new_order: New queue order (list of participant IDs)

        Returns:
            (success: bool, error: Optional[str])
        """
        # Validate all participants in new_order are in current queue
        current_set = set(session.turn_queue)
        new_set = set(new_order)

        if current_set != new_set:
            return False, "New order contains different participants than current queue"

        session.turn_queue = new_order
        session.last_activity_at = datetime.now()

        self._logger.info(f"🎤 Turn queue reordered: {new_order}")

        return True, None

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_queue_length(self, session: GroupSession) -> int:
        """Get current queue length"""
        return len(session.turn_queue)

    def get_current_turn_duration(self, session: GroupSession) -> Optional[float]:
        """
        Get duration of current turn in seconds

        Args:
            session: Group session

        Returns:
            Duration in seconds, or None if no active turn
        """
        if session.current_speaker_id is None:
            return None

        speaker_id = session.current_speaker_id
        if speaker_id not in self._turn_start_times:
            return None

        start_time = self._turn_start_times[speaker_id]
        return (datetime.now() - start_time).total_seconds()

    def get_statistics(self) -> Dict[str, any]:
        """
        Get turn manager statistics

        Returns:
            Statistics dictionary
        """
        return {
            "total_turns_processed": self._total_turns_processed,
            "active_turns": len(self._turn_start_times),
            "turn_timeout_seconds": self.turn_timeout_seconds,
            "max_queue_size": self.max_queue_size,
        }

    def get_turn_info(self, session: GroupSession) -> Dict[str, any]:
        """
        Get current turn information

        Args:
            session: Group session

        Returns:
            Turn info dictionary
        """
        return {
            "current_speaker_id": session.current_speaker_id,
            "current_turn_duration": self.get_current_turn_duration(session),
            "is_processing": session.is_processing,
            "queue_length": self.get_queue_length(session),
            "turn_queue": session.turn_queue.copy(),
            "is_expired": self.is_turn_expired(session),
        }
