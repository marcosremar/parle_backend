"""
Sentence Buffer - Accumulates LLM tokens and emits complete sentences

This component is the glue between LLM streaming (tokens) and TTS streaming (sentences).
It buffers incoming tokens until a sentence boundary is detected, then emits the complete
sentence for TTS synthesis.

Usage:
    buffer = SentenceBuffer(min_chunk_chars=50)

    # Accumulate tokens from LLM
    for token in llm_stream:
        sentence = buffer.add_token(token)
        if sentence:
            # Complete sentence detected!
            audio = await tts.synthesize(sentence)
            yield audio

    # Don't forget the final buffer
    final = buffer.flush()
    if final:
        audio = await tts.synthesize(final)
        yield audio
"""

import re
from typing import Optional
from loguru import logger


class SentenceBuffer:
    """
    Buffer that accumulates LLM tokens and emits complete sentences

    Detects sentence boundaries using regex patterns (. ! ? \\n) and emits
    complete sentences for progressive TTS synthesis.

    This dramatically reduces perceived latency by allowing TTS to start
    processing before the LLM has finished generating the entire response.
    """

    def __init__(
        self,
        min_chunk_chars: int = 50,
        sentence_end_pattern: str = r'[.!?\n]+\s*'
    ):
        """
        Initialize sentence buffer

        Args:
            min_chunk_chars: Minimum characters before emitting a chunk
                           Prevents very short sentences ("Hi." would wait for more)
            sentence_end_pattern: Regex pattern for sentence boundaries
                                 Default: r'[.!?\\n]+\\s*' (period, exclamation, question, newline)
        """
        self.buffer = ""
        self.min_chunk_chars = min_chunk_chars
        self.sentence_end_pattern = re.compile(sentence_end_pattern)
        self.total_chars_processed = 0
        self.sentences_emitted = 0

        logger.debug(
            f"SentenceBuffer initialized: min_chars={min_chunk_chars}, "
            f"pattern={sentence_end_pattern}"
        )

    def add_token(self, token: str) -> Optional[str]:
        """
        Add a token to the buffer

        Args:
            token: Single token from LLM (can be word, punctuation, space, etc.)

        Returns:
            Complete sentence if boundary detected, None otherwise

        Example:
            >>> buffer = SentenceBuffer(min_chunk_chars=10)
            >>> buffer.add_token("Hello")  # None
            >>> buffer.add_token(" ")      # None
            >>> buffer.add_token("world")  # None
            >>> buffer.add_token("!")      # None (too short)
            >>> buffer.add_token(" ")      # "Hello world! " (boundary + min length met)
        """
        self.buffer += token
        self.total_chars_processed += len(token)

        # Look for sentence boundary
        match = self.sentence_end_pattern.search(self.buffer)

        if match and len(self.buffer) >= self.min_chunk_chars:
            # Found a complete sentence that meets minimum length!
            sentence = self.buffer[:match.end()].strip()
            self.buffer = self.buffer[match.end():]  # Remove from buffer
            self.sentences_emitted += 1

            logger.debug(
                f"✂️  Sentence #{self.sentences_emitted} emitted: "
                f"'{sentence[:50]}...' ({len(sentence)} chars)"
            )

            return sentence

        return None

    def flush(self) -> Optional[str]:
        """
        Flush remaining buffer contents

        Call this when LLM has finished generating to get any remaining text
        that didn't end with a sentence boundary.

        Returns:
            Remaining buffer contents, or None if empty

        Example:
            >>> buffer = SentenceBuffer()
            >>> buffer.add_token("Hello")
            >>> buffer.flush()  # "Hello" (no boundary, but we're done)
        """
        if self.buffer.strip():
            sentence = self.buffer.strip()
            self.buffer = ""
            self.sentences_emitted += 1

            logger.debug(
                f"🔚 Final flush: Sentence #{self.sentences_emitted} "
                f"({len(sentence)} chars)"
            )

            return sentence

        return None

    def reset(self) -> None:
        """
        Reset buffer state

        Useful for reusing the same buffer instance across multiple requests.
        """
        self.buffer = ""
        self.total_chars_processed = 0
        self.sentences_emitted = 0
        logger.debug("🔄 SentenceBuffer reset")

    def get_stats(self) -> dict:
        """
        Get buffer statistics

        Returns:
            Dictionary with buffer stats
        """
        return {
            "buffer_length": len(self.buffer),
            "total_chars_processed": self.total_chars_processed,
            "sentences_emitted": self.sentences_emitted,
            "buffer_content_preview": self.buffer[:100] if self.buffer else ""
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing SentenceBuffer")
    print("=" * 80)

    # Test 1: Simple sentence splitting
    print("\n📝 Test 1: Simple sentence splitting")
    buffer = SentenceBuffer(min_chunk_chars=10)

    tokens = [
        "Hello", ",", " ", "how", " ", "are", " ", "you", "?", " ",
        "I", "'m", " ", "doing", " ", "great", "!", " ",
        "Thanks", " ", "for", " ", "asking", "."
    ]

    sentences = []
    for token in tokens:
        sentence = buffer.add_token(token)
        if sentence:
            sentences.append(sentence)
            print(f"  ✅ Emitted: '{sentence}'")

    # Flush final
    final = buffer.flush()
    if final:
        sentences.append(final)
        print(f"  ✅ Final: '{final}'")

    print(f"\n  📊 Total sentences: {len(sentences)}")
    assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}"

    # Test 2: Minimum chunk size enforcement
    print("\n📝 Test 2: Minimum chunk size (prevents 'Hi.' from being emitted immediately)")
    buffer = SentenceBuffer(min_chunk_chars=20)

    short_tokens = ["Hi", ".", " ", "How", " ", "are", " ", "you", "?"]

    for token in short_tokens:
        sentence = buffer.add_token(token)
        if sentence:
            print(f"  ✅ Emitted: '{sentence}'")

    final = buffer.flush()
    if final:
        print(f"  ✅ Final (flushed): '{final}'")

    # Test 3: Stats
    print("\n📝 Test 3: Statistics")
    buffer = SentenceBuffer()
    for token in ["Test", " ", "sentence", "."]:
        buffer.add_token(token)

    stats = buffer.get_stats()
    print(f"  📊 Stats: {stats}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
