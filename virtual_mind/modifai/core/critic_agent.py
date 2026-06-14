"""
Critic Agent — Modifai Agentic Pipeline
========================================
Evaluates each (instruction, input, response) sample against its source chunk.
Returns one of three verdicts:
  - accept   : sample is good, keep it as-is
  - rewrite  : sample has salvageable content; a corrected output is provided
  - reject   : sample is too bad to fix; discard it

Output schema (locked — do NOT change without team sync):
{
    "verdict":          "accept" | "rewrite" | "reject",
    "reason":           str,          # one sentence explaining the decision
    "rewritten_output": str | None,   # only present when verdict == "rewrite"
    "scores": {
        "specificity":    float,      # 0.0–1.0
        "grounding":      float,      # 0.0–1.0
        "format":         float       # 0.0–1.0
    }
}
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from modifai.core.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (locked after Day 2 — agreed integration contract)
# ─────────────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are a rigorous training-data quality critic for an LLM fine-tuning pipeline.

Your job is to evaluate a single training example — an (instruction, input, response) triple — against the source document chunk it was generated from.

You must judge three dimensions:

1. SPECIFICITY  (0.0 – 1.0)
   Does the response give a precise, concrete answer, or is it vague and generic?
   - 1.0 = highly specific, mentions entities/numbers/steps from the chunk
   - 0.5 = partially specific but misses key details
   - 0.0 = completely generic ("It depends", "Please refer to the document", etc.)

2. GROUNDING  (0.0 – 1.0)
   Is every claim in the response supported by the source chunk?
   - 1.0 = every fact comes directly from the chunk
   - 0.5 = mostly grounded but includes one unsupported inference
   - 0.0 = fabricates facts not in the chunk

3. FORMAT  (0.0 – 1.0)
   Is the response a complete, well-formed answer to the instruction?
   - 1.0 = grammatically complete, directly answers the question
   - 0.5 = partially answers or has minor truncation/awkward phrasing
   - 0.0 = incomplete sentence, just a list of keywords, or does not answer

VERDICT RULES (apply in order):
- If grounding < 0.4: REJECT — hallucination risk is too high to fix safely
- If specificity < 0.4 AND format < 0.5: REJECT — not worth fixing
- If any score < 0.6: REWRITE — produce a corrected response grounded in the chunk
- Otherwise: ACCEPT

REWRITE RULES (critical — failure here breaks the pipeline):
- ONLY use information present in the source chunk
- Do NOT invent facts, examples, or numbers not in the chunk
- Keep the same instruction; only fix the response
- If you cannot write a grounded rewrite, REJECT instead

Respond with ONLY a valid JSON object. No markdown, no explanation outside the JSON.

JSON schema:
{
  "verdict": "accept" | "rewrite" | "reject",
  "reason": "<one sentence>",
  "rewritten_output": "<corrected response string>" | null,
  "scores": {
    "specificity": <float 0.0-1.0>,
    "grounding": <float 0.0-1.0>,
    "format": <float 0.0-1.0>
  }
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample critique
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_message(sample: Dict[str, Any], chunk_text: str) -> str:
    """Constructs the user turn sent to the Critic LLM."""
    return (
        f"SOURCE CHUNK:\n{chunk_text}\n\n"
        f"INSTRUCTION: {sample.get('instruction', '')}\n"
        f"INPUT: {sample.get('input', '')}\n"
        f"RESPONSE: {sample.get('response', '')}"
    )


def _parse_critic_response(raw: str) -> Optional[Dict[str, Any]]:
    """
    Strips markdown fences if present, then JSON-parses the response.
    Returns None if parsing fails (caller handles retry/fallback).
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    required = {"verdict", "reason", "scores"}
    if not required.issubset(parsed.keys()):
        return None
    if parsed.get("verdict") not in ("accept", "rewrite", "reject"):
        return None

    parsed.setdefault("rewritten_output", None)
    if parsed["verdict"] != "rewrite":
        parsed["rewritten_output"] = None

    return parsed


def critique_sample(
    sample: Dict[str, Any],
    chunk_text: str,
    aws_region: str,
    model_id: str,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Calls the Critic LLM on a single sample.

    Returns a verdict dict matching the locked output schema.
    On unrecoverable failure, returns a safe REJECT verdict.

    Args:
        sample:      dict with keys instruction, input, response (or output), chunk_id
        chunk_text:  the raw source text the sample was generated from
        aws_region:  AWS region for Bedrock
        model_id:    Bedrock model ID
        max_retries: re-prompt once on malformed JSON before giving up
    """
    provider = get_llm_provider()
    user_msg = _build_user_message(sample, chunk_text)

    for attempt in range(max_retries + 1):
        try:
            result = provider.generate(
                system_prompt=CRITIC_SYSTEM_PROMPT,
                user_prompt=user_msg,
                response_schema=None, # CRITIC uses standard generation parsed from string
            )

            if result is not None and result.get("verdict") in ("accept", "rewrite", "reject"):
                logger.debug(
                    "Critic verdict for chunk %s: %s — %s",
                    sample.get("chunk_id"),
                    result["verdict"],
                    result.get("reason", ""),
                )
                return result

            logger.warning(
                "Critic returned malformed output on attempt %d: %s",
                attempt + 1,
                result,
            )

        except Exception as e:
            logger.error("Provider call failed on attempt %d: %s", attempt + 1, e)

    logger.error(
        "Critic failed after %d attempts for chunk %s. Defaulting to REJECT.",
        max_retries + 1,
        sample.get("chunk_id"),
    )
    return {
        "verdict": "reject",
        "reason": "Critic LLM failed to return a valid response after retries.",
        "rewritten_output": None,
        "scores": {"specificity": 0.0, "grounding": 0.0, "format": 0.0},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode
# ─────────────────────────────────────────────────────────────────────────────

def run_critic_batch(
    dataset: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    aws_region: str,
    model_id: str,
) -> Dict[str, Any]:
    """
    Runs the Critic over every sample in the dataset.

    Returns a dict with:
      - "results":   list of per-sample dicts (original sample + verdict info)
      - "stats":     aggregate statistics across the batch
      - "survivors": samples that passed (accepted or rewritten), ready for fine-tuning

    Args:
        dataset:    list of samples, each with keys instruction/input/response/chunk_id
        chunks:     list of chunk dicts: [{"chunk_id": int, "text": str}, ...]
        aws_region: AWS region for Bedrock
        model_id:   Bedrock model ID
    """
    logger.info("Running Critic batch over %d samples.", len(dataset))

    chunk_lookup: Dict[int, str] = {c["chunk_id"]: c["text"] for c in chunks}

    results: List[Dict[str, Any]] = []
    survivors: List[Dict[str, Any]] = []

    accept_count = rewrite_count = reject_count = 0

    for i, sample in enumerate(dataset):
        chunk_id = sample.get("chunk_id")
        chunk_text = chunk_lookup.get(chunk_id, "")

        if not chunk_text:
            logger.warning(
                "Sample %d references unknown chunk_id=%s. Rejecting.", i, chunk_id
            )
            verdict_dict = {
                "verdict": "reject",
                "reason": "No source chunk found for this sample.",
                "rewritten_output": None,
                "scores": {"specificity": 0.0, "grounding": 0.0, "format": 0.0},
            }
        else:
            verdict_dict = critique_sample(sample, chunk_text, aws_region, model_id)

        verdict = verdict_dict["verdict"]

        if verdict == "accept":
            accept_count += 1
            survivors.append(sample)
        elif verdict == "rewrite":
            rewrite_count += 1
            rewritten = dict(sample)
            rewritten["response"] = verdict_dict["rewritten_output"]
            rewritten["_critic_rewritten"] = True
            survivors.append(rewritten)
        else:
            reject_count += 1

        results.append({
            "sample_index": i,
            "chunk_id": chunk_id,
            "original_sample": sample,
            **verdict_dict,
        })

    total = len(dataset)
    stats = {
        "total":         total,
        "accepted":      accept_count,
        "rewritten":     rewrite_count,
        "rejected":      reject_count,
        "accept_pct":    round(accept_count / total * 100, 1) if total else 0.0,
        "rewrite_pct":   round(rewrite_count / total * 100, 1) if total else 0.0,
        "reject_pct":    round(reject_count / total * 100, 1) if total else 0.0,
        "survivor_count": len(survivors),
    }

    logger.info(
        "Critic batch complete — accept: %s%% | rewrite: %s%% | reject: %s%%",
        stats["accept_pct"],
        stats["rewrite_pct"],
        stats["reject_pct"],
    )

    return {
        "results":   results,
        "stats":     stats,
        "survivors": survivors,
    }
