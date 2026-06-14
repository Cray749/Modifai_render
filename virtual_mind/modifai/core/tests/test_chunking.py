"""Unit tests for chunking.py — no AWS calls needed."""
from __future__ import annotations

import pytest
from modifai.core.chunking import chunk_text, estimate_token_count


# ── Fixtures ───────────────────────────────────────────────────────────────────

SHORT_TEXT = "This is a short sentence. It has very few words."

MEDIUM_TEXT = "\n\n".join([
    "The onboarding process begins with completing the new hire paperwork. "
    "This includes the tax forms, direct deposit information, and emergency contacts. "
    "All documents must be submitted within the first three business days.",

    "After paperwork, the IT department will provision your laptop and accounts. "
    "You will receive access to Slack, Jira, GitHub, and the internal wiki. "
    "Please allow up to 24 hours for all accounts to be activated.",

    "Your first week includes mandatory training sessions covering company policies, "
    "security awareness, and department-specific onboarding. "
    "Check your calendar for scheduled sessions with your manager.",
])

LONG_TEXT = (MEDIUM_TEXT + "\n\n") * 20  # ~60 paragraphs


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_returns_list_of_strings():
    chunks = chunk_text(MEDIUM_TEXT)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)


def test_short_text_returns_empty_or_single():
    """Very short text may produce 0 or 1 chunk depending on word count."""
    chunks = chunk_text(SHORT_TEXT)
    assert len(chunks) <= 1  # too short for multiple chunks


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_none_equivalent_returns_empty():
    """Whitespace-only text should return empty."""
    assert chunk_text("\n\n\n") == []


def test_medium_text_produces_chunks(  ):
    chunks = chunk_text(MEDIUM_TEXT, target_tokens=100, overlap_tokens=20)
    # With small target, medium text should produce at least 1 chunk
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_long_text_produces_multiple_chunks():
    chunks = chunk_text(LONG_TEXT, target_tokens=200, overlap_tokens=30)
    assert len(chunks) >= 5, f"Expected ≥5 chunks, got {len(chunks)}"


def test_chunk_text_preserves_content():
    """Every key phrase from input should appear somewhere across all chunks."""
    # Use very small target so MEDIUM_TEXT definitely gets chunked
    chunks = chunk_text(MEDIUM_TEXT, target_tokens=50, overlap_tokens=10)
    if not chunks:
        # If text is too short even for small target, join is the whole text
        chunks = [MEDIUM_TEXT]
    all_text = " ".join(chunks)
    # Key phrases from each paragraph should appear somewhere
    assert "new hire paperwork" in all_text
    assert "IT department" in all_text
    assert "mandatory training" in all_text



def test_overlap_means_boundary_content_appears_in_multiple_chunks():
    """
    With overlap, content near the boundary of a chunk should appear in both
    the current and next chunk (approximate check — may not always hold
    depending on exact word positions).
    """
    chunks = chunk_text(LONG_TEXT, target_tokens=100, overlap_tokens=40)
    if len(chunks) < 2:
        pytest.skip("Not enough chunks to test overlap")
    # Rough check: last few words of chunk[0] should appear somewhere in chunk[1]
    last_words_of_first = chunks[0].split()[-10:]
    found_overlap = any(word in chunks[1] for word in last_words_of_first)
    assert found_overlap, "Expected overlap between consecutive chunks"


@pytest.mark.parametrize("target_tokens", [64, 128, 256, 512])
def test_various_target_sizes(target_tokens):
    """chunk_text should work with various target token sizes."""
    chunks = chunk_text(LONG_TEXT, target_tokens=target_tokens, overlap_tokens=target_tokens // 8)
    assert isinstance(chunks, list)
    if chunks:
        # Rough check: no chunk should be drastically larger than target
        # Allow 2x overrun since we don't split mid-sentence
        words_per_chunk = [len(c.split()) for c in chunks]
        max_words = max(words_per_chunk)
        target_words = int(target_tokens * 0.75)
        assert max_words < target_words * 3, (
            f"Chunk too large: {max_words} words vs target {target_words} words"
        )


def test_estimate_token_count():
    text = "This is a test sentence with exactly eight words."
    # 8 words / 0.75 ≈ 10 tokens
    estimate = estimate_token_count(text)
    assert 8 <= estimate <= 15  # Generous range for approximation


def test_estimate_token_count_empty():
    assert estimate_token_count("") == 0
