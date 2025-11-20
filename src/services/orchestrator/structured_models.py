#!/usr/bin/env python3
"""
Pydantic Models for Structured LLM Output
Guarantees response format without regex/parsing
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ScenarioValidation(BaseModel):
    """Validation metrics for scenario coherence"""

    coherence_score: float = Field(
        ge=0.0, le=1.0,
        description="Coherence score: 0.0 (completely off-topic) to 1.0 (perfectly on-topic)"
    )

    in_scope: bool = Field(
        description="Whether the message is related to the scenario context"
    )

    should_redirect: bool = Field(
        description="Whether the assistant should redirect user back to scenario topics"
    )

    found_topics: List[str] = Field(
        default_factory=list,
        description="List of expected topics found in user message"
    )

    missing_topics: List[str] = Field(
        default_factory=list,
        description="List of expected topics NOT mentioned yet"
    )

    reason: str = Field(
        description="Brief explanation of the validation result"
    )


class ScenarioMetadata(BaseModel):
    """Additional metadata about user message and conversation"""

    sentiment: Optional[str] = Field(
        None,
        description="User sentiment: positive, negative, neutral, mixed"
    )

    intent: Optional[str] = Field(
        None,
        description="User intent: asking, browsing, buying, complaining, chatting, other"
    )

    language_quality: Optional[str] = Field(
        None,
        description="Language quality: beginner, intermediate, advanced, native"
    )

    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Confidence in validation accuracy (0.0-1.0)"
    )


class ScenarioValidatedResponse(BaseModel):
    """
    Complete LLM response with embedded validation

    This is the structured output format that the LLM must return.
    No parsing or regex needed - Pydantic validates everything.
    """

    validation: ScenarioValidation = Field(
        description="Validation metrics for this user message"
    )

    assistant_response: str = Field(
        min_length=10,
        description="Assistant's response to the user in the scenario language"
    )

    metadata: Optional[ScenarioMetadata] = Field(
        None,
        description="Additional metadata about the conversation (optional)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "validation": {
                    "coherence_score": 0.95,
                    "in_scope": True,
                    "should_redirect": False,
                    "found_topics": ["camisa", "azul", "tamanho"],
                    "missing_topics": ["cor", "preço"],
                    "reason": "User is asking about shopping for clothes, perfectly on-topic"
                },
                "assistant_response": "Ótimo! Temos várias camisas azuis no tamanho M. Você prefere manga curta ou longa?",
                "metadata": {
                    "sentiment": "neutral",
                    "intent": "browsing",
                    "language_quality": "intermediate",
                    "confidence": 0.98
                }
            }
        }
