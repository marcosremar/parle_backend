"""
Participant Manager Utility

Manages participant lifecycle, permissions, and state transitions
in group conversations.
"""

from typing import List, Dict, Optional, Set
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.group_session.models import (
    Participant,
    ParticipantRole,
    ParticipantStatus,
    GroupConversation,
)


class ParticipantManager:
    """
    Manages participant operations in group conversations

    Responsibilities:
    - Join/leave participant logic
    - Permission validation
    - Status transitions
    - Role management
    - Participant count validation
    """

    def __init__(self):
        """Initialize ParticipantManager"""
        self._logger = logger

    # ========================================================================
    # Join/Leave Operations
    # ========================================================================

    def can_join(
        self,
        group: GroupConversation,
        user_id: str,
        existing_participants: List[Participant]
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a user can join a group

        Args:
            group: Group conversation
            user_id: User ID attempting to join
            existing_participants: Current participants in group

        Returns:
            (can_join: bool, reason: Optional[str])
        """
        # Check if group is active
        if not group.is_active:
            return False, "Group is inactive"

        # Check if group is full
        if not group.can_add_participant():
            return False, f"Group is full (max {group.max_participants} participants)"

        # Check if user is already a participant
        for participant in existing_participants:
            if participant.user_id == user_id:
                # User is already in group
                if participant.status == ParticipantStatus.BANNED:
                    return False, "User is banned from this group"
                elif participant.status == ParticipantStatus.DISCONNECTED:
                    # Allow reconnection
                    return True, "Reconnection allowed"
                else:
                    return False, "User is already a participant"

        # All checks passed
        return True, None

    def join_group(
        self,
        group: GroupConversation,
        user_id: str,
        display_name: str,
        role: ParticipantRole = ParticipantRole.MEMBER,
        existing_participants: List[Participant] = None
    ) -> tuple[Optional[Participant], Optional[str]]:
        """
        Add a user as a participant to a group

        Args:
            group: Group conversation
            user_id: User ID to add
            display_name: Display name for participant
            role: Participant role (default: MEMBER)
            existing_participants: Current participants (for validation)

        Returns:
            (participant: Optional[Participant], error: Optional[str])
        """
        existing_participants = existing_participants or []

        # Validate can join
        can_join, reason = self.can_join(group, user_id, existing_participants)
        if not can_join:
            return None, reason

        # Check if reconnecting
        for participant in existing_participants:
            if participant.user_id == user_id and participant.status == ParticipantStatus.DISCONNECTED:
                # Reconnect existing participant
                participant.status = ParticipantStatus.ACTIVE
                participant.last_active_at = datetime.now()
                self._logger.info(f"👤 User {user_id} reconnected to group {group.group_conversation_id}")
                return participant, None

        # Create new participant
        participant = Participant(
            user_id=user_id,
            group_conversation_id=group.group_conversation_id,
            display_name=display_name,
            role=role,
            status=ParticipantStatus.ACTIVE,
        )

        # Set permissions based on role
        if role == ParticipantRole.OWNER:
            participant.can_send = True
            participant.can_receive = True
            participant.can_invite = True
        elif role == ParticipantRole.MODERATOR:
            participant.can_send = True
            participant.can_receive = True
            participant.can_invite = True
        elif role == ParticipantRole.MEMBER:
            participant.can_send = True
            participant.can_receive = True
            participant.can_invite = False
        elif role == ParticipantRole.OBSERVER:
            participant.can_send = False
            participant.can_receive = True
            participant.can_invite = False

        self._logger.info(
            f"👤 User {user_id} ({display_name}) joined group {group.group_conversation_id} "
            f"as {role.value}"
        )

        return participant, None

    def leave_group(
        self,
        participant: Participant,
        hard_remove: bool = False
    ) -> tuple[bool, Optional[str]]:
        """
        Remove a participant from a group

        Args:
            participant: Participant to remove
            hard_remove: If True, delete participant. If False, mark as disconnected.

        Returns:
            (success: bool, error: Optional[str])
        """
        # Check if owner
        if participant.role == ParticipantRole.OWNER and not hard_remove:
            return False, "Owner cannot leave group (transfer ownership first)"

        if hard_remove:
            # Hard remove (delete participant)
            self._logger.info(
                f"👤 Participant {participant.participant_id} removed from group "
                f"{participant.group_conversation_id}"
            )
            return True, None
        else:
            # Soft remove (mark as disconnected)
            participant.status = ParticipantStatus.DISCONNECTED
            participant.last_active_at = datetime.now()
            self._logger.info(
                f"👤 Participant {participant.participant_id} disconnected from group "
                f"{participant.group_conversation_id}"
            )
            return True, None

    # ========================================================================
    # Permission Validation
    # ========================================================================

    def can_send_message(self, participant: Participant) -> tuple[bool, Optional[str]]:
        """
        Check if participant can send messages

        Args:
            participant: Participant to check

        Returns:
            (can_send: bool, reason: Optional[str])
        """
        # Check status
        if participant.status != ParticipantStatus.ACTIVE:
            return False, f"Participant is {participant.status.value}"

        # Check permission
        if not participant.can_send:
            return False, "Participant does not have send permission"

        return True, None

    def can_receive_message(self, participant: Participant) -> tuple[bool, Optional[str]]:
        """
        Check if participant can receive messages

        Args:
            participant: Participant to check

        Returns:
            (can_receive: bool, reason: Optional[str])
        """
        # Banned users cannot receive
        if participant.status == ParticipantStatus.BANNED:
            return False, "Participant is banned"

        # Check permission
        if not participant.can_receive:
            return False, "Participant does not have receive permission"

        return True, None

    def can_invite_participant(self, participant: Participant) -> tuple[bool, Optional[str]]:
        """
        Check if participant can invite others

        Args:
            participant: Participant to check

        Returns:
            (can_invite: bool, reason: Optional[str])
        """
        # Check status
        if participant.status != ParticipantStatus.ACTIVE:
            return False, f"Participant is {participant.status.value}"

        # Check permission
        if not participant.can_invite:
            return False, "Participant does not have invite permission"

        return True, None

    def can_moderate(self, participant: Participant) -> tuple[bool, Optional[str]]:
        """
        Check if participant can perform moderation actions

        Args:
            participant: Participant to check

        Returns:
            (can_moderate: bool, reason: Optional[str])
        """
        # Check status
        if participant.status != ParticipantStatus.ACTIVE:
            return False, f"Participant is {participant.status.value}"

        # Check role
        if not participant.can_moderate():
            return False, f"Participant role {participant.role.value} cannot moderate"

        return True, None

    # ========================================================================
    # Status Management
    # ========================================================================

    def set_status(
        self,
        participant: Participant,
        new_status: ParticipantStatus
    ) -> tuple[bool, Optional[str]]:
        """
        Change participant status

        Args:
            participant: Participant to update
            new_status: New status

        Returns:
            (success: bool, error: Optional[str])
        """
        old_status = participant.status

        # Validate status transitions
        if old_status == ParticipantStatus.BANNED and new_status != ParticipantStatus.ACTIVE:
            return False, "Banned participants can only be unbanned (set to ACTIVE)"

        # Update status
        participant.status = new_status
        participant.last_active_at = datetime.now()

        self._logger.info(
            f"👤 Participant {participant.participant_id} status changed: "
            f"{old_status.value} → {new_status.value}"
        )

        return True, None

    def mark_active(self, participant: Participant):
        """Mark participant as active"""
        participant.status = ParticipantStatus.ACTIVE
        participant.last_active_at = datetime.now()

    def mark_idle(self, participant: Participant):
        """Mark participant as idle"""
        participant.status = ParticipantStatus.IDLE
        participant.last_active_at = datetime.now()

    def mark_disconnected(self, participant: Participant):
        """Mark participant as disconnected"""
        participant.status = ParticipantStatus.DISCONNECTED
        participant.last_active_at = datetime.now()

    def ban_participant(self, participant: Participant):
        """Ban participant from group"""
        participant.status = ParticipantStatus.BANNED
        participant.can_send = False
        participant.can_receive = False
        participant.can_invite = False
        participant.last_active_at = datetime.now()

        self._logger.warning(
            f"👤 Participant {participant.participant_id} BANNED from group "
            f"{participant.group_conversation_id}"
        )

    # ========================================================================
    # Role Management
    # ========================================================================

    def change_role(
        self,
        participant: Participant,
        new_role: ParticipantRole,
        requester: Participant
    ) -> tuple[bool, Optional[str]]:
        """
        Change participant role

        Args:
            participant: Participant to promote/demote
            new_role: New role
            requester: Participant requesting the change

        Returns:
            (success: bool, error: Optional[str])
        """
        # Only moderators and owners can change roles
        can_mod, reason = self.can_moderate(requester)
        if not can_mod:
            return False, f"Requester cannot moderate: {reason}"

        # Cannot change owner role
        if participant.role == ParticipantRole.OWNER:
            return False, "Cannot change owner role (transfer ownership instead)"

        # Cannot promote to owner
        if new_role == ParticipantRole.OWNER:
            return False, "Cannot promote to owner (transfer ownership instead)"

        old_role = participant.role
        participant.role = new_role

        # Update permissions based on new role
        if new_role == ParticipantRole.MODERATOR:
            participant.can_invite = True
        elif new_role == ParticipantRole.MEMBER:
            participant.can_send = True
            participant.can_receive = True
            participant.can_invite = False
        elif new_role == ParticipantRole.OBSERVER:
            participant.can_send = False
            participant.can_receive = True
            participant.can_invite = False

        self._logger.info(
            f"👤 Participant {participant.participant_id} role changed: "
            f"{old_role.value} → {new_role.value}"
        )

        return True, None

    def transfer_ownership(
        self,
        current_owner: Participant,
        new_owner: Participant
    ) -> tuple[bool, Optional[str]]:
        """
        Transfer group ownership

        Args:
            current_owner: Current owner
            new_owner: Participant to become new owner

        Returns:
            (success: bool, error: Optional[str])
        """
        # Validate current owner
        if current_owner.role != ParticipantRole.OWNER:
            return False, "Requester is not the owner"

        # Validate new owner is in same group
        if current_owner.group_conversation_id != new_owner.group_conversation_id:
            return False, "Participants are not in the same group"

        # Validate new owner is active
        if new_owner.status != ParticipantStatus.ACTIVE:
            return False, f"New owner is {new_owner.status.value}"

        # Transfer ownership
        current_owner.role = ParticipantRole.MEMBER
        new_owner.role = ParticipantRole.OWNER
        new_owner.can_send = True
        new_owner.can_receive = True
        new_owner.can_invite = True

        self._logger.info(
            f"👤 Ownership transferred: {current_owner.participant_id} → {new_owner.participant_id} "
            f"in group {current_owner.group_conversation_id}"
        )

        return True, None

    # ========================================================================
    # Query Helpers
    # ========================================================================

    def get_active_participants(self, participants: List[Participant]) -> List[Participant]:
        """Get all active participants"""
        return [p for p in participants if p.status == ParticipantStatus.ACTIVE]

    def get_participants_by_role(
        self,
        participants: List[Participant],
        role: ParticipantRole
    ) -> List[Participant]:
        """Get participants with specific role"""
        return [p for p in participants if p.role == role]

    def get_owner(self, participants: List[Participant]) -> Optional[Participant]:
        """Get group owner"""
        owners = self.get_participants_by_role(participants, ParticipantRole.OWNER)
        return owners[0] if owners else None

    def count_active(self, participants: List[Participant]) -> int:
        """Count active participants"""
        return len(self.get_active_participants(participants))

    def get_participant_names(self, participants: List[Participant]) -> List[str]:
        """Get list of participant display names"""
        return [p.display_name for p in participants if p.status != ParticipantStatus.BANNED]
