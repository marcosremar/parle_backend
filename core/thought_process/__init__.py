"""
Thought Process Framework - Language Teaching AI Module.

This module provides a complete framework for generating and managing
thought processes in language teaching applications.

RECOMMENDED: Use ThoughtProcessManager for easy unified interface
-----------------------------------------------------------------

    from src.core.thought_process import ThoughtProcessManager

    manager = ThoughtProcessManager()

    # 1. Get system prompt for LLM
    prompt = manager.get_prompt(language="portuguese", level="A2")

    # 2. Process student input
    response, is_valid, errors = manager.process_student_input(
        user_text="Eu vai para praia",
        llm_response="Você vai para a praia!",
        llm_metadata=llm_metadata,
        language="portuguese",
        level="A2",
        stt_time_ms=100,
        llm_time_ms=500,
        tts_time_ms=200
    )

    # 3. Send to client
    if is_valid:
        ndjson = manager.to_ndjson(response)


Alternative: Use individual components if you need more control
---------------------------------------------------------------

    from src.core.thought_process import (
        ThoughtProcessPromptManager,
        ThoughtProcessValidator,
        ResponseMetadataStructurer,
    )
"""

# Models
from src.core.thought_process.models import (
    # Individual process models (7 for speech-only learning)
    GrammarAnalysis,
    VocabularyAssessment,
    PedagogicalStrategy,
    ConversationFlow,
    LearningProgress,
    CulturalContext,
    LearningRecommendation,
    # Composite models
    ThoughtProcess,
    ThoughtProcessCollection,
    HintsForNextTurn,
    ResponseMetadata,
    LatencyMetrics,
    StreamingConversationResponse,
    # Config models
    PromptConfig,
    ThoughtProcessConfig,
)

# Managers
from src.core.thought_process.prompt_manager import ThoughtProcessPromptManager
from src.core.thought_process.validator import ThoughtProcessValidator
from src.core.thought_process.structurer import ResponseMetadataStructurer
from src.core.thought_process.manager import ThoughtProcessManager

__all__ = [
    # Central Interface (RECOMMENDED)
    "ThoughtProcessManager",
    # Individual Components (Advanced Use)
    "ThoughtProcessPromptManager",
    "ThoughtProcessValidator",
    "ResponseMetadataStructurer",
    # Models (7 processes for speech-only learning)
    "GrammarAnalysis",
    "VocabularyAssessment",
    "PedagogicalStrategy",
    "ConversationFlow",
    "LearningProgress",
    "CulturalContext",
    "LearningRecommendation",
    "ThoughtProcess",
    "ThoughtProcessCollection",
    "HintsForNextTurn",
    "ResponseMetadata",
    "LatencyMetrics",
    "StreamingConversationResponse",
    "PromptConfig",
    "ThoughtProcessConfig",
]

__version__ = "1.0.0"
__author__ = "Ultravox"
__description__ = "Thought Process Framework for Language Teaching AI"
