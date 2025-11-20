#!/usr/bin/env python3
"""
Structured LLM Client - Uses Groq with JSON mode for guaranteed structured output
No parsing or regex needed - Pydantic validates everything
"""

import os
import json
import logging
import aiohttp
from typing import List, Optional, Dict, Any
from .structured_models import ScenarioValidatedResponse

logger = logging.getLogger(__name__)


class StructuredLLMClient:
    """
    LLM client with structured output (Pydantic validation)
    Guarantees response format without parsing/regex

    Uses Groq API with JSON mode to ensure structured responses
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize structured LLM client

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            base_url: Base URL for Groq API (defaults to official URL)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url or "https://api.groq.com/openai/v1"
        # Model: llama-3.1-8b-instant (faster and cheaper alternative to 70B models)
        # Supports JSON mode for structured output with 128k context
        self.model = "llama-3.1-8b-instant"  # Groq model
        self.session: Optional[aiohttp.ClientSession] = None

        if not self.api_key:
            logger.warning("⚠️ GROQ_API_KEY not set - structured LLM will fail")

    async def initialize(self, session: Optional[aiohttp.ClientSession] = None):
        """Initialize HTTP session"""
        if session:
            self.session = session
        else:
            self.session = aiohttp.ClientSession()
        logger.info("✅ StructuredLLMClient initialized")

    async def generate_with_validation(
        self,
        user_message: str,
        scenario_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> ScenarioValidatedResponse:
        """
        Generate response with embedded validation (structured output)

        Args:
            user_message: User's message text
            scenario_context: Scenario info (type, topics, roles, system_prompt)
            conversation_history: Previous conversation turns
            temperature: LLM temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            ScenarioValidatedResponse with validation + response (typed!)

        Raises:
            Exception if API call fails or response is invalid
        """

        if not self.session:
            await self.initialize()

        # Build system prompt
        system_prompt = self._build_system_prompt(scenario_context)

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Get JSON schema for structured output
        json_schema = ScenarioValidatedResponse.model_json_schema()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_object"
            }
        }

        try:
            # Call Groq API
            logger.info(f"🤖 Calling Groq with structured output (model: {self.model})...")

            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Groq API error ({response.status}): {error_text}")
                    raise Exception(f"Groq API error: {error_text}")

                result = await response.json()

                # Extract response content
                content = result["choices"][0]["message"]["content"]

                logger.debug(f"📄 Raw LLM response: {content[:200]}...")

                # Parse with Pydantic (validates structure automatically)
                validated_response = ScenarioValidatedResponse.model_validate_json(content)

                logger.info(
                    f"✅ Structured response: "
                    f"coherence={validated_response.validation.coherence_score:.2f}, "
                    f"in_scope={validated_response.validation.in_scope}, "
                    f"redirect={validated_response.validation.should_redirect}"
                )

                return validated_response

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            raise Exception(f"LLM returned invalid JSON: {e}")

        except Exception as e:
            logger.error(f"❌ Structured LLM error: {e}")
            raise

    def _build_system_prompt(self, scenario_context: Dict[str, Any]) -> str:
        """
        Build system prompt from scenario context

        The prompt instructs the LLM to return JSON matching the Pydantic schema
        """

        expected_topics = scenario_context.get("expected_topics", [])
        scenario_type = scenario_context.get("type", "conversation")
        ai_role = scenario_context.get("ai_role", "assistant")
        user_role = scenario_context.get("user_role", "user")
        language = scenario_context.get("language", "pt-BR")
        base_prompt = scenario_context.get("system_prompt", "")

        topics_str = ", ".join(expected_topics) if expected_topics else "general conversation"

        return f"""{base_prompt}

═══════════════════════════════════════════════════════════════
SCENARIO VALIDATION & RESPONSE INSTRUCTIONS
═══════════════════════════════════════════════════════════════

You must analyze the user's message and respond with a JSON object.

SCENARIO CONTEXT:
• Type: {scenario_type}
• Expected topics: {topics_str}
• Your role: {ai_role}
• User role: {user_role}
• Language: {language}

JSON STRUCTURE (you MUST follow this exactly):
{{
  "validation": {{
    "coherence_score": <float 0.0-1.0>,
    "in_scope": <boolean>,
    "should_redirect": <boolean>,
    "found_topics": [<list of strings>],
    "missing_topics": [<list of strings>],
    "reason": "<string>"
  }},
  "assistant_response": "<your response in {language}>",
  "metadata": {{
    "sentiment": "<positive/negative/neutral/mixed or null>",
    "intent": "<asking/browsing/buying/complaining/chatting/other or null>",
    "language_quality": "<beginner/intermediate/advanced/native or null>",
    "confidence": <float 0.0-1.0 or null>
  }}
}}

VALIDATION RULES:
1. Calculate coherence_score:
   • 0.9-1.0: User is perfectly on-topic
   • 0.7-0.9: User is mostly on-topic
   • 0.5-0.7: User is somewhat related
   • 0.3-0.5: User is loosely related
   • 0.0-0.3: User is off-topic

2. Set in_scope:
   • true: if coherence_score >= 0.3
   • false: if coherence_score < 0.3

3. Set should_redirect:
   • true: if coherence_score < 0.3 (user is off-topic)
   • false: if coherence_score >= 0.3 (user is on-topic)

4. found_topics:
   • List topics from expected_topics that the user mentioned
   • Example: ["camisa", "tamanho"] if user said "camisa tamanho M"

5. missing_topics:
   • List topics from expected_topics that user hasn't mentioned yet
   • Example: ["cor", "preço"] if user hasn't asked about these

6. reason:
   • Brief explanation of why you assigned this coherence_score
   • Example: "User is asking about shopping for clothes"

RESPONSE RULES:
1. If should_redirect is true:
   • Gently redirect user back to scenario topics
   • Example: "Vamos focar nas compras! Posso ajudar a encontrar alguma roupa?"

2. If should_redirect is false:
   • Respond naturally to their message
   • Stay in character as {ai_role}
   • Use language: {language}

METADATA (optional but helpful):
• sentiment: How does the user feel? (positive/negative/neutral/mixed)
• intent: What does the user want to do?
• language_quality: How good is their language? (for learning scenarios)
• confidence: How confident are you in your validation? (0.0-1.0)

CRITICAL: Respond ONLY with valid JSON. No text before or after.
"""

    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("✅ StructuredLLMClient session closed")
