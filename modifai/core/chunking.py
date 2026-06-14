"""
chunking.py — Splits extracted text into overlapping chunks for dataset generation.

Strategy: Sliding window over sentences/paragraphs.
  - Target: ~512 tokens per chunk (approximated as word_count / 0.75)
  - Overlap: ~64 tokens between consecutive chunks (helps the Critic ground samples)
  - Minimum chunk size: 50 words (skip chunks that are too short to generate from)

Usage:
    from modifai.core.chunking import chunk_text

    chunks = chunk_text(raw_text, target_tokens=512, overlap_tokens=64)
    # returns List[str], each element is a chunk of text
"""
from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Approximation: average English word ≈ 0.75 tokens (GPT-style BPE tokenizer)
_WORDS_PER_TOKEN = 0.75
_MIN_CHUNK_WORDS = 50


def chunk_text(
    text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[str]:
    """
    Split text into overlapping chunks suitable for LLM dataset generation.

    Args:
        text:           Raw text string (from Textract or manual input).
        target_tokens:  Target chunk size in tokens (default 512).
                        Actual word count = target_tokens * 0.75 ≈ 384 words.
        overlap_tokens: Overlap between consecutive chunks in tokens (default 64).
                        Helps the Critic evaluate grounding across chunk boundaries.

    Returns:
        List of text chunk strings. Empty list if text is too short.

    Notes:
        - Splits on paragraph boundaries first (double newlines), then falls back
          to sentence splitting. This keeps related sentences together.
        - Chunks shorter than _MIN_CHUNK_WORDS words are discarded.
    """
    if not text or not text.strip():
        logger.warning("chunk_text called with empty text — returning empty list.")
        return []

    target_words = int(target_tokens * _WORDS_PER_TOKEN)
    overlap_words = int(overlap_tokens * _WORDS_PER_TOKEN)

    # Split into sentences, preserving paragraph breaks
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_words: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        # If adding this sentence would exceed the target, flush current chunk
        if current_word_count + sentence_word_count > target_words and current_words:
            chunk_text_str = " ".join(current_words).strip()
            if len(current_words) >= _MIN_CHUNK_WORDS:
                chunks.append(chunk_text_str)
            else:
                logger.debug(
                    "Discarding short chunk (%d words < %d min)",
                    len(current_words), _MIN_CHUNK_WORDS,
                )

            # Keep overlap: retain the last `overlap_words` words for context
            if overlap_words > 0 and len(current_words) > overlap_words:
                current_words = current_words[-overlap_words:]
                current_word_count = len(current_words)
            else:
                current_words = []
                current_word_count = 0

        current_words.extend(sentence_words)
        current_word_count += sentence_word_count

    # Flush remaining text
    if current_words and len(current_words) >= _MIN_CHUNK_WORDS:
        chunks.append(" ".join(current_words).strip())

    logger.info(
        "Chunking complete: %d chunks from %d words (target=%d tokens, overlap=%d tokens)",
        len(chunks),
        len(text.split()),
        target_tokens,
        overlap_tokens,
    )
    return chunks


def estimate_token_count(text: str) -> int:
    """
    Rough token count estimate for a text string.
    Uses word count / 0.75 approximation (good enough for chunking decisions).
    """
    return int(len(text.split()) / _WORDS_PER_TOKEN)


# ── Private helpers ────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentence-level units, preserving paragraph breaks.

    Strategy:
    1. Split on double-newline (paragraph boundaries) first.
    2. Within each paragraph, split on sentence-ending punctuation.
    3. Return list of sentence strings, empty strings filtered out.
    """
    paragraphs = re.split(r"\n{2,}", text)
    sentences: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split on sentence-ending punctuation followed by whitespace + capital letter
        # Handles: "Mr. Smith went." edge cases by requiring capital after space
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
        for s in raw_sentences:
            s = s.strip()
            if s:
                sentences.append(s)

    return sentences
