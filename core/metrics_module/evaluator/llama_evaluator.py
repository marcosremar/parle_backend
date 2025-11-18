"""
Llama-4 Scout Evaluator for Response Quality
Forces structured JSON output for metrics
"""

import json
import logging
from typing import Dict, Any, Optional
from groq import Groq
import os

logger = logging.getLogger(__name__)


class LlamaEvaluator:
    """
    Evaluates response quality using Llama-4-Scout
    Forces JSON output for structured metrics
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Llama evaluator with Groq API"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY required for Llama evaluation")

        self.client = Groq(api_key=self.api_key)
        self.model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    def evaluate_response(self, question: str, response: str, language: str = "pt") -> Dict[str, Any]:
        """
        Evaluate response quality with forced JSON output

        Returns structured metrics:
        - coherence_score: 0.0-1.0
        - directness_score: 0.0-1.0
        - educational_value: 0.0-1.0
        - engagement_score: 0.0-1.0
        - naturalness_score: 0.0-1.0
        - language_quality: 0.0-1.0
        - tokens_quality: qualitative assessment
        - detailed_feedback: string feedback
        """

        system_prompt = """You are an AI response evaluator. Evaluate the quality of responses in Portuguese.

        You MUST respond with ONLY a valid JSON object with this exact structure:
        {
            "coherence_score": 0.0-1.0,
            "directness_score": 0.0-1.0,
            "educational_value": 0.0-1.0,
            "engagement_score": 0.0-1.0,
            "naturalness_score": 0.0-1.0,
            "language_quality": 0.0-1.0,
            "tokens_quality": "poor|fair|good|excellent",
            "response_completeness": 0.0-1.0,
            "grammar_correctness": 0.0-1.0,
            "detailed_feedback": "Brief feedback in Portuguese"
        }

        Scoring guidelines:
        - coherence_score: Logical flow and consistency
        - directness_score: How directly it answers the question
        - educational_value: Information quality and usefulness
        - engagement_score: Conversational quality
        - naturalness_score: Natural Portuguese language use
        - language_quality: Grammar, vocabulary, fluency
        - tokens_quality: Overall token usage efficiency
        - response_completeness: How complete the answer is
        - grammar_correctness: Grammar accuracy

        IMPORTANT: Return ONLY the JSON object, no additional text."""

        user_prompt = f"""Evaluate this response in Portuguese:

Question: {question}
Response: {response}
Expected Language: {"Portuguese" if language == "pt" else "English"}

Return the evaluation as a JSON object ONLY."""

        try:
            # Request JSON response from Llama
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}  # Force JSON output
            )

            response_text = completion.choices[0].message.content

            # Parse JSON response
            try:
                evaluation = json.loads(response_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from Llama: {response_text}")
                # Return default scores if parsing fails
                evaluation = self._default_evaluation()

            # Ensure all required fields exist
            evaluation = self._validate_evaluation(evaluation)

            # Add metadata
            evaluation['evaluator'] = 'Llama-4-Scout'
            evaluation['model'] = self.model

            return evaluation

        except Exception as e:
            logger.error(f"Llama evaluation error: {e}")
            return self._default_evaluation()

    def _validate_evaluation(self, evaluation: Dict) -> Dict:
        """Validate and fill missing fields in evaluation"""
        required_fields = {
            'coherence_score': 0.5,
            'directness_score': 0.5,
            'educational_value': 0.5,
            'engagement_score': 0.5,
            'naturalness_score': 0.5,
            'language_quality': 0.5,
            'tokens_quality': 'fair',
            'response_completeness': 0.5,
            'grammar_correctness': 0.5,
            'detailed_feedback': 'Avaliação não disponível'
        }

        for field, default in required_fields.items():
            if field not in evaluation:
                evaluation[field] = default
            elif field != 'tokens_quality' and field != 'detailed_feedback':
                # Ensure numeric fields are floats between 0 and 1
                try:
                    value = float(evaluation[field])
                    evaluation[field] = max(0.0, min(1.0, value))
                except (ValueError, TypeError):
                    evaluation[field] = default

        return evaluation

    def _default_evaluation(self) -> Dict:
        """Return default evaluation when API fails"""
        return {
            'coherence_score': 0.5,
            'directness_score': 0.5,
            'educational_value': 0.5,
            'engagement_score': 0.5,
            'naturalness_score': 0.5,
            'language_quality': 0.5,
            'tokens_quality': 'unknown',
            'response_completeness': 0.5,
            'grammar_correctness': 0.5,
            'detailed_feedback': 'Avaliação indisponível',
            'evaluator': 'Llama-4-Scout',
            'model': self.model,
            'error': 'Evaluation failed'
        }

    def batch_evaluate(self, qa_pairs: list) -> list:
        """Evaluate multiple question-response pairs"""
        results = []
        for qa in qa_pairs:
            evaluation = self.evaluate_response(
                qa.get('question', ''),
                qa.get('response', ''),
                qa.get('language', 'pt')
            )
            results.append(evaluation)
        return results