"""
Context Loader Engine

Loads conversation context in parallel for better performance.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging

logger = logging.getLogger(__name__)


class ContextLoader:
    """Loads session, scenario, history, and student data in parallel."""

    def __init__(self, clients: Dict[str, Any]) -> None:
        """
        Initialize context loader.
        
        Args:
            clients: Dictionary of service clients
        """
        self.clients = clients

    async def load_context(
        self,
        session_id: str,
        session_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[
        Optional[Dict[str, Any]],  # scenario_data
        List[Dict[str, Any]],  # conversation_history
        Optional[Dict[str, Any]],  # student_cefr_progress
        Optional[Dict[str, Any]],  # target_skill
        Optional[str],  # conversation_id
        Optional[str],  # scenario_id
        str  # user_id
    ]:
        """
        Load all context data in parallel.
        
        Args:
            session_id: Session identifier
            session_data: Optional pre-loaded session data
            
        Returns:
            Tuple of (scenario_data, conversation_history, student_cefr_progress, 
                     target_skill, conversation_id, scenario_id, user_id)
        """
        # Load session if not provided
        if session_data is None:
            if "session" in self.clients:
                session_data = await self.clients["session"].get_session(session_id)
            else:
                session_data = None

        if not session_data:
            logger.warning(f"⚠️ Session {session_id} not found, using defaults")
            return None, [], None, None, None, None, session_id

        conversation_id = session_data.get("conversation_id")
        scenario_id = session_data.get("scenario_id")
        user_id = session_data.get("user_id", session_id)

        # Build parallel tasks
        tasks: List[Optional[Any]] = []

        # Scenario task
        if scenario_id and "scenarios" in self.clients:
            tasks.append(self.clients["scenarios"].get_scenario(scenario_id))
        else:
            tasks.append(None)

        # History task
        if conversation_id and "conversation_store" in self.clients:
            tasks.append(
                self.clients["conversation_store"].get_context(
                    conversation_id,
                    limit=10
                )
            )
        else:
            tasks.append(None)

        # Student profile task
        if "student_model" in self.clients:
            tasks.append(self.clients["student_model"].get_cefr_progress(user_id))
        else:
            tasks.append(None)

        # Next skill task
        if "learning_path" in self.clients:
            tasks.append(self.clients["learning_path"].get_next_skill(user_id))
        else:
            tasks.append(None)

        # Execute in parallel
        if any(task is not None for task in tasks):
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process scenario result
            scenario_data = None
            if scenario_id and not isinstance(results[0], Exception) and results[0]:
                scenario_data = results[0]
                logger.info(f"📝 Using scenario: {scenario_data.get('name', scenario_id)}")
            elif scenario_id and isinstance(results[0], Exception):
                logger.warning(f"⚠️ Failed to load scenario: {results[0]}")

            # Process history result
            conversation_history: List[Dict[str, Any]] = []
            if conversation_id and not isinstance(results[1], Exception) and results[1]:
                messages = results[1]
                # Format for LLM (simple format)
                for msg in messages:
                    role = "user" if msg.get("sender") == "user" else "assistant"
                    content = msg.get("text", msg.get("content", ""))
                    if content:
                        conversation_history.append({"role": role, "content": content})
                logger.info(f"📚 Loaded {len(messages)} previous messages for context")
            elif conversation_id and isinstance(results[1], Exception):
                logger.warning(f"⚠️ Failed to load conversation history: {results[1]}")

            # Process student CEFR result
            student_cefr_progress = None
            if "student_model" in self.clients and not isinstance(results[2], Exception) and results[2]:
                student_cefr_progress = results[2]
                current_level = (
                    student_cefr_progress.get("current_estimated_level")
                    or student_cefr_progress.get("current_level")
                    or "A1"
                )
                logger.info(f"👤 Loaded student CEFR progress: {current_level}")

            # Process next skill result
            target_skill = None
            if "learning_path" in self.clients and not isinstance(results[3], Exception) and results[3]:
                target_skill = results[3]
                logger.info(f"🎯 Target skill: {target_skill.get('skill_name', target_skill.get('skill_id', 'unknown'))}")

            return (
                scenario_data,
                conversation_history,
                student_cefr_progress,
                target_skill,
                conversation_id,
                scenario_id,
                user_id
            )

        return None, [], None, None, conversation_id, scenario_id, user_id
