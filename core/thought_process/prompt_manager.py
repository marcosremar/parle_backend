"""
Thought Process Prompt Manager.

Manages system prompts for language teaching with thought process generation.
"""

import os
from typing import Optional
from pathlib import Path
from src.core.thought_process.models import PromptConfig


class ThoughtProcessPromptManager:
    """Manages and customizes system prompts for language teaching AI."""

    def __init__(self):
        """Initialize the prompt manager."""
        self.prompt_dir = Path(__file__).parent / "prompt_templates"
        self._base_prompts = {}  # Cache for language-specific base prompts
        self._language_specific = {}
        self._level_specific = {}

    def _load_base_prompt(self, language: str = "portuguese") -> str:
        """
        Load the base system prompt template for a specific language.

        Args:
            language: The language code (e.g., 'portuguese', 'spanish', 'english')

        Returns:
            The base prompt template as a string
        """
        # Check cache first
        if language in self._base_prompts:
            return self._base_prompts[language]

        # Map language to file
        language_map = {
            "portuguese": "base_pt.md",
            "pt": "base_pt.md",
            "spanish": "base_es.md",
            "es": "base_es.md",
            "french": "base_fr.md",
            "fr": "base_fr.md",
            "english": "base.md",
            "en": "base.md",
        }

        # Get the filename for this language
        filename = language_map.get(language.lower(), "base.md")
        file_path = self.prompt_dir / filename

        # Try to load the language-specific file
        if file_path.exists():
            prompt = file_path.read_text()
        else:
            # Fallback to English base prompt
            base_file = self.prompt_dir / "base.md"
            if base_file.exists():
                prompt = base_file.read_text()
            else:
                prompt = self._get_default_base_prompt()

        # Cache for future use
        self._base_prompts[language] = prompt
        return prompt

    def _get_default_base_prompt(self) -> str:
        """Get default base prompt if file not found."""
        return """You are an advanced AI language teaching assistant designed for speech-to-speech language learning.

## CORE CONTEXT: LANGUAGE LEARNING ENVIRONMENT

This is a PEDAGOGICAL SYSTEM where:
- Every interaction is designed to teach and improve language skills
- Your responses must balance natural conversation WITH teaching effectiveness
- You provide not just answers, but learning opportunities

## TARGET CONFIGURATION

Language: [LANGUAGE]
Level: [LEVEL]
Region: [REGION]
Session: [SESSION_NUMBER]

## YOUR RESPONSIBILITIES

### OUTPUT 1: MAIN RESPONSE
Provide a natural, conversational response at or slightly above the student's level.

### OUTPUT 2: RESPONSE METADATA
Generate JSON with 9 Thought Processes (see documentation).

See SYSTEM_PROMPT_LANGUAGE_TEACHING.md for complete instructions."""

    def get_prompt(
        self,
        language: str = "portuguese",
        level: str = "A2",
        region: Optional[str] = None,
        learning_context: Optional[str] = None,
        session_number: Optional[int] = None,
    ) -> str:
        """
        Get customized system prompt for specific student configuration.

        Args:
            language: Target language (portuguese, spanish, french, etc.)
            level: CEFR level (A1, A2, B1, B2, C1, C2)
            region: Optional region (brazil, portugal, spain, mexico)
            learning_context: Optional context (conversational, business, academic)
            session_number: Session number for progress tracking

        Returns:
            Customized system prompt string
        """
        # Load base prompt in the target language
        prompt = self._load_base_prompt(language)

        # Replace placeholders
        prompt = prompt.replace("[LANGUAGE]", self._format_language(language))
        prompt = prompt.replace("[LEVEL]", level)
        prompt = prompt.replace("[REGION]", region or "neutral")
        prompt = prompt.replace("[SESSION_NUMBER]", str(session_number or 1))

        # Load language-specific customizations
        language_specific = self._load_language_customizations(language)
        if language_specific:
            prompt += "\n\n## LANGUAGE-SPECIFIC GUIDELINES\n\n" + language_specific

        # Load level-specific customizations
        level_specific = self._load_level_customizations(level)
        if level_specific:
            prompt += "\n\n## LEVEL-SPECIFIC CUSTOMIZATIONS\n\n" + level_specific

        # Add context-specific guidance
        if learning_context:
            prompt += f"\n\n## LEARNING CONTEXT\n\nFocus: {learning_context}\n"

        return prompt

    def _format_language(self, language: str) -> str:
        """Format language name nicely."""
        language_names = {
            "portuguese": "Portuguese",
            "spanish": "Spanish",
            "french": "French",
            "german": "German",
            "italian": "Italian",
            "mandarin": "Mandarin Chinese",
            "japanese": "Japanese",
        }
        return language_names.get(language.lower(), language.capitalize())

    def _load_language_customizations(self, language: str) -> Optional[str]:
        """Load language-specific customizations."""
        if language in self._language_specific:
            return self._language_specific[language]

        lang_file = self.prompt_dir / f"{language.lower()}.md"
        if lang_file.exists():
            content = lang_file.read_text()
            self._language_specific[language] = content
            return content

        return None

    def _load_level_customizations(self, level: str) -> Optional[str]:
        """Load level-specific customizations."""
        if level in self._level_specific:
            return self._level_specific[level]

        # Group levels: A1-A2, B1-B2, C1-C2
        level_group = f"{level[0]}{level[1]}"  # Extract A/B/C and 1/2
        level_file = self.prompt_dir / f"level_{level_group}.md"
        if level_file.exists():
            content = level_file.read_text()
            self._level_specific[level] = content
            return content

        return None

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

        Returns:
            PromptConfig Pydantic model
        """
        return PromptConfig(
            language=language,
            level=level,
            region=region,
            learning_context=learning_context,
            session_number=session_number,
        )

    def validate_config(self, config: PromptConfig) -> bool:
        """
        Validate prompt configuration.

        Args:
            config: PromptConfig to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        valid_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        if config.level not in valid_levels:
            raise ValueError(f"Invalid level: {config.level}. Must be one of {valid_levels}")

        if not config.language:
            raise ValueError("Language must be specified")

        return True

    def list_available_languages(self) -> list:
        """List available language templates."""
        if not self.prompt_dir.exists():
            return []

        languages = []
        for file in self.prompt_dir.glob("*.md"):
            if file.name not in ["base.md"]:
                lang = file.stem
                if not lang.startswith("level_"):
                    languages.append(lang)

        return languages

    def reload_templates(self) -> None:
        """Reload all templates from disk."""
        self._base_prompt = self._load_base_prompt()
        self._language_specific = {}
        self._level_specific = {}
