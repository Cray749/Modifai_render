"""
CriticAgent — adapter class wrapping the core critic_agent module functions.

WHY THIS FILE EXISTS:
The core critic implementation (modifai/core/critic_agent.py) uses module-level
functions (critique_sample, run_critic_batch). This class adapter presents the
interface the pipeline_loop and build manual expect:
  - CriticAgent(model_id, region, chunks)
  - run_single(sample) → {verdict, reason, rewritten_output}
  - run_batch(samples) → {verdicts: [...], stats: {...}}

KEY ADAPTER DECISIONS:
1. chunks must be passed at construction — the core critic needs chunk text for
   grounding evaluation, but the pipeline_loop's run_batch(samples) call doesn't
   carry chunks. The loop passes them at CriticAgent init time.

2. Field normalisation — pipeline samples use "output" as the response field;
   the core critic uses "response". The adapter translates output→response before
   calling the core, and preserves the original sample (with "output") in verdicts
   so _collect_accepted() in pipeline_loop.py works correctly.

3. Stats normalisation — the core run_critic_batch returns extra fields
   (rewrite_pct, reject_pct, survivor_count). The adapter trims to the locked
   CriticStats schema: {total, accepted, rewritten, rejected, accept_pct}.
   Extra fields are kept for logging but not in the returned stats dict.

4. accept_pct — only counts pure "accept" verdicts (not rewrites).
   Rewrites are still collected by _collect_accepted() for the final dataset,
   but the threshold comparison uses accept_pct only. This is by design.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Any, List, Optional

from modifai.core.critic_agent import critique_sample, run_critic_batch
from modifai.agents.schemas import CriticBatchOutput, CriticVerdict, CriticStats

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    LLM-powered quality critic for training samples.

    Evaluates each (instruction, input, output) sample against its source chunk
    and returns one of three verdicts: accept, rewrite, or reject.

    Usage:
        # Convert your List[str] chunks to structured format first:
        structured_chunks = [{"chunk_id": i, "text": c} for i, c in enumerate(chunks)]

        agent = CriticAgent(chunks=structured_chunks)
        batch = agent.run_batch(samples)
        # batch["verdicts"]  → list of {verdict, reason, rewritten_output, original_sample}
        # batch["stats"]     → {total, accepted, rewritten, rejected, accept_pct}
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 1,
    ):
        """
        Args:
            model_id:    Bedrock model ID override (default: env AWS_MODEL_ID or nova-micro).
            region:      AWS region override (default: env AWS_REGION or us-east-1).
            chunks:      Structured chunks list: [{"chunk_id": int, "text": str}, ...].
                         Required for real evaluation. May be omitted in unit tests
                         where run_batch is mocked.
            max_retries: Re-prompt attempts on malformed LLM output (default 1).
        """
        self.model_id = model_id or os.environ.get(
            "AWS_MODEL_ID", "amazon.nova-micro-v1:0"
        )
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.chunks: List[Dict[str, Any]] = chunks or []
        self.max_retries = max_retries

        # Build lookup once for performance
        self._chunk_lookup: Dict[int, str] = {
            c["chunk_id"]: c["text"] for c in self.chunks
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_single(self, sample: dict) -> dict:
        """
        Evaluate a single sample.

        Args:
            sample: Dict with instruction, input, output (or response), and
                    optionally chunk_id (used to look up source chunk text).

        Returns:
            {verdict: str, reason: str, rewritten_output: str|None}
        """
        chunk_id = sample.get("chunk_id")
        chunk_text = self._chunk_lookup.get(chunk_id, "") if chunk_id is not None else ""

        adapted = self._normalize_output_to_response(sample)
        raw_verdict = critique_sample(
            adapted, chunk_text, self.region, self.model_id, self.max_retries
        )
        return {
            "verdict": raw_verdict["verdict"],
            "reason": raw_verdict["reason"],
            "rewritten_output": raw_verdict.get("rewritten_output"),
        }

    def run_batch(self, samples: List[dict]) -> dict:
        """
        Evaluate a batch of samples.

        Args:
            samples: List of sample dicts. Each should have:
                       instruction, input, output (or response), chunk_id.

        Returns:
            CriticBatchOutput: {
                "verdicts": [
                    {verdict, reason, rewritten_output, original_sample}, ...
                ],
                "stats": {total, accepted, rewritten, rejected, accept_pct},
            }

        NOTE: verdicts include "original_sample" (the pre-adaptation sample with
        the "output" field intact) so pipeline_loop._collect_accepted() can
        reconstruct final samples correctly.
        """
        # Translate "output" → "response" for the core critic
        adapted_samples = [self._normalize_output_to_response(s) for s in samples]

        raw_output = run_critic_batch(
            adapted_samples, self.chunks, self.region, self.model_id
        )

        # Build verdicts list: map core results format → pipeline verdicts format.
        # We use the ORIGINAL sample (pre-adaptation) as original_sample so
        # _collect_accepted() sees the "output" field, not "response".
        verdicts = []
        for result, original_sample in zip(raw_output["results"], samples):
            verdicts.append({
                "verdict": result["verdict"],
                "reason": result["reason"],
                "rewritten_output": result.get("rewritten_output"),
                "original_sample": original_sample,
            })

        # Trim stats to locked CriticStats schema (drop rewrite_pct, reject_pct, survivor_count)
        raw_stats = raw_output["stats"]
        stats: CriticStats = {
            "total":      raw_stats.get("total", len(samples)),
            "accepted":   raw_stats["accepted"],
            "rewritten":  raw_stats["rewritten"],
            "rejected":   raw_stats["rejected"],
            "accept_pct": raw_stats["accept_pct"],
        }

        logger.info(
            "CriticAgent.run_batch: total=%d accepted=%d rewritten=%d rejected=%d accept_pct=%.1f%%",
            stats["total"],
            stats["accepted"],
            stats["rewritten"],
            stats["rejected"],
            stats["accept_pct"],
        )

        return {"verdicts": verdicts, "stats": stats}

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_output_to_response(sample: dict) -> dict:
        """
        Translate 'output' → 'response' field for the core critic's _build_user_message.
        If 'response' already exists, it's left unchanged.
        Returns a shallow copy so the original sample is not mutated.
        """
        adapted = dict(sample)
        if "output" in adapted and "response" not in adapted:
            adapted["response"] = adapted["output"]
        return adapted
