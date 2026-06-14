"""
Dataset generation module — generates synthetic training samples from text chunks.

TODO: Replace stub implementation with the actual Bedrock generation call.
      The `custom_prompt` parameter is injected by the CurriculumAgent's
      targeted_prompt when the loop detects quality gaps.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from modifai.core.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

# ── Generation system prompt (base — custom_prompt is appended when provided) ──

_BASE_SYSTEM_PROMPTS = {
    "QA": (
        "You are a training data generator for a question-answering fine-tuning dataset. "
        "Given a source text chunk, generate realistic question-answer pairs that could be "
        "asked by a user querying a knowledge base. Each pair must be grounded entirely in "
        "the provided chunk. Return JSON array."
    ),
    "instruction": (
        "You are a training data generator for an instruction-following fine-tuning dataset. "
        "Given a source text chunk, generate instruction-input-output triples where the "
        "instruction is a task and the output demonstrates how to complete it using information "
        "from the chunk. Return JSON array."
    ),
    "tutor": (
        "You are a training data generator for a tutoring fine-tuning dataset. "
        "Given a source text chunk, generate instructional dialogue examples that teach the "
        "concepts in the chunk step by step. Return JSON array."
    ),
}

_SAMPLE_SCHEMA = """
Each sample must follow this JSON schema:
{
  "instruction": "<the task or question>",
  "input": "<context or empty string>",
  "output": "<the expected response>",
  "chunk_id": <int — the chunk index this sample came from>
}
Return a JSON array of samples. No markdown, no preamble.
"""


def generate_dataset(
    chunks: List,
    mode: str = "QA",
    samples_per_chunk: int = 4,
    custom_prompt: Optional[str] = None,
) -> List[dict]:
    """
    Generate synthetic training samples from text chunks via Bedrock.

    Args:
        chunks:            List of text chunks. Each element can be either:
                             - str: raw text (chunk index used as chunk_id)
                             - dict: {"chunk_id": int, "text": str}
        mode:              Generation mode: "QA" | "instruction" | "tutor"
        samples_per_chunk: Number of samples to generate per chunk (3–8)
        custom_prompt:     Optional targeted improvement prompt from CurriculumAgent.
                           Appended to the base system prompt to guide generation.

    Returns:
        List of sample dicts: {instruction, input, output, chunk_id}
    """
    provider = get_llm_provider()

    base_prompt = _BASE_SYSTEM_PROMPTS.get(mode, _BASE_SYSTEM_PROMPTS["QA"])
    system_prompt = base_prompt + _SAMPLE_SCHEMA
    if custom_prompt:
        system_prompt += f"\n\nADDITIONAL QUALITY REQUIREMENTS:\n{custom_prompt}"

    all_samples: List[dict] = []

    for i, chunk in enumerate(chunks):
        # Normalise chunk to (chunk_id, text)
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id", i)
            chunk_text = chunk.get("text", "")
        else:
            chunk_id = i
            chunk_text = str(chunk)

        if not chunk_text.strip():
            logger.warning("Chunk %d is empty — skipping.", chunk_id)
            continue

        user_message = (
            f"SOURCE CHUNK (chunk_id={chunk_id}):\n{chunk_text}\n\n"
            f"Generate exactly {samples_per_chunk} training samples from this chunk."
        )

        try:
            raw = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_message,
                temperature=0.7,
                return_raw=True
            )
            samples = _parse_generation_response(raw, chunk_id, samples_per_chunk)
            all_samples.extend(samples)
            logger.debug("Chunk %d: generated %d samples.", chunk_id, len(samples))

        except Exception as e:
            logger.error("Generation failed for chunk %d: %s", chunk_id, e)

    logger.info(
        "Dataset generation complete: %d samples from %d chunks.",
        len(all_samples),
        len(chunks),
    )
    return all_samples


def _parse_generation_response(
    raw: str, chunk_id: int, expected_count: int
) -> List[dict]:
    """Parse and validate Bedrock generation response."""
    import json
    import re

    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()

    # Strip anything before the first '[' to handle preamble
    bracket_pos = text.find("[")
    if bracket_pos != -1:
        text = text[bracket_pos:]

    try:
        samples = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse generation response for chunk %d: %s", chunk_id, e)
        return []

    if not isinstance(samples, list):
        logger.error("Generation response for chunk %d was not a list.", chunk_id)
        return []

    valid = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        # Ensure required fields
        s.setdefault("instruction", "")
        s.setdefault("input", "")
        s.setdefault("output", "")
        s["chunk_id"] = chunk_id  # Always set/overwrite chunk_id
        valid.append(s)

    return valid[:expected_count]
