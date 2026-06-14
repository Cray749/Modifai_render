"""
run_agentic_loop() — wires Orchestrator → DatasetGeneration → Critic → Curriculum.

The loop runs at most max_iterations times. Early exit if:
  - All samples accepted on first Critic pass (exit_reason = "all_accepted_first_pass")
  - accept_pct >= strategy["quality_threshold"] * 100 (exit_reason = "threshold_met")
  - max_iterations reached (exit_reason = "max_iterations")

ADAPTER NOTES (differences from build manual template):
  - CriticAgent is imported at module level (required for @patch in E2E tests)
  - CriticAgent receives structured chunks at construction; run_batch(samples) only
  - chunks: List[str] is converted to [{"chunk_id": i, "text": c}] internally
  - _generate_samples() annotates each returned sample with chunk_id if missing,
    so the Critic can look up source chunk text for grounding evaluation
  - _collect_accepted() handles both "output" and "response" field names in samples
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from modifai.agents.orchestrator import OrchestratorAgent
from modifai.agents.curriculum import CurriculumAgent
from modifai.agents.critic import CriticAgent          # module-level for test patching
from modifai.agents.logging_utils import AgentEventLogger
from modifai.agents.schemas import (
    DocMetadata,
    OrchestratorOutput,
    PipelineLoopState,
)

logger = logging.getLogger(__name__)


def run_agentic_loop(
    goal: str,
    doc_metadata: DocMetadata,
    chunks: List[str],
    max_iterations: int = 3,
    event_log_path: str = "agent_events.jsonl",
    model_id: Optional[str] = None,
    region: Optional[str] = None,
) -> PipelineLoopState:
    """
    Run the full Modifai agentic loop.

    Args:
        goal:             User's goal string (e.g. "Build a Q&A bot for our HR policy docs")
        doc_metadata:     DocMetadata TypedDict
        chunks:           List of raw text chunks from the chunking step (List[str])
        max_iterations:   Maximum Critic→Curriculum loop iterations (default 3)
        event_log_path:   Path to write JSONL event stream (read by P3 dashboard)
        model_id:         Bedrock model ID override (default from env or nova-micro)
        region:           AWS region override (default from env or us-east-1)

    Returns:
        PipelineLoopState with final_samples, final_stats, events, exit_reason
    """
    event_logger = AgentEventLogger(path=event_log_path)
    curriculum_outputs: List[dict] = []

    # ── Convert raw chunks to structured format for CriticAgent ───────────────
    # CriticAgent needs {"chunk_id": int, "text": str} dicts for chunk lookup.
    # chunk_id == index in the original list.
    structured_chunks = [
        {"chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)
    ]

    # ── Step 1: Orchestrator ───────────────────────────────────────────────────
    logger.info("Running OrchestratorAgent...")
    orchestrator = OrchestratorAgent(model_id=model_id, region=region)
    strategy: OrchestratorOutput = orchestrator.run(
        goal=goal, doc_metadata=doc_metadata
    )

    event_logger.log(
        agent="orchestrator",
        iteration=0,
        decision=(
            f"intent={strategy['intent']}, "
            f"threshold={strategy['quality_threshold']}, "
            f"spc={strategy['samples_per_chunk']}"
        ),
        reason=strategy["reasoning"],
        data=dict(strategy),
    )

    # ── Step 2: Initial dataset generation ────────────────────────────────────
    logger.info("Generating initial dataset with strategy: %s", strategy)
    samples = _generate_samples(
        chunks=chunks,
        intent=strategy["intent"],
        samples_per_chunk=strategy["samples_per_chunk"],
        custom_prompt=None,
    )
    logger.info("Generated %d samples", len(samples))

    # ── Step 3: Agentic loop ───────────────────────────────────────────────────
    critic = CriticAgent(model_id=model_id, region=region, chunks=structured_chunks)
    curriculum_agent = CurriculumAgent(model_id=model_id, region=region)

    iteration = 0
    exit_reason = "max_iterations"
    stats: dict = {"total": 0, "accepted": 0, "rewritten": 0, "rejected": 0, "accept_pct": 0.0}

    while iteration < max_iterations:
        iteration += 1
        logger.info("=== Loop iteration %d / %d ===", iteration, max_iterations)

        # ── Critic pass ────────────────────────────────────────────────────────
        batch_output = critic.run_batch(samples)
        stats = batch_output["stats"]
        accept_pct = stats["accept_pct"]

        logger.info(
            "Critic iter=%d: accepted=%d rewritten=%d rejected=%d accept_pct=%.1f%%",
            iteration,
            stats["accepted"],
            stats["rewritten"],
            stats["rejected"],
            accept_pct,
        )

        event_logger.log(
            agent="critic",
            iteration=iteration,
            decision=(
                f"accepted={stats['accepted']}/{stats['total']} "
                f"({accept_pct:.1f}%), "
                f"rewritten={stats['rewritten']}, rejected={stats['rejected']}"
            ),
            reason=(
                f"accept_pct={accept_pct:.1f}% vs threshold={strategy['quality_threshold'] * 100:.1f}%"
            ),
            data=dict(stats),
        )

        # ── Early exit: all accepted on first pass ─────────────────────────────
        if iteration == 1 and accept_pct == 100.0:
            logger.info("All samples accepted on first pass — skipping Curriculum.")
            exit_reason = "all_accepted_first_pass"
            samples = _collect_accepted(batch_output)
            break

        # ── Threshold met ──────────────────────────────────────────────────────
        if accept_pct >= strategy["quality_threshold"] * 100:
            logger.info(
                "Quality threshold met (%.1f%% >= %.1f%%) — exiting loop.",
                accept_pct,
                strategy["quality_threshold"] * 100,
            )
            exit_reason = "threshold_met"
            samples = _collect_accepted(batch_output)
            break

        # ── Threshold NOT met: run Curriculum (unless last iteration) ──────────
        if iteration >= max_iterations:
            logger.warning(
                "Max iterations (%d) reached. Final accept_pct=%.1f%%.",
                max_iterations,
                accept_pct,
            )
            exit_reason = "max_iterations"
            samples = _collect_accepted(batch_output)
            break

        rejection_reasons = CurriculumAgent.extract_rejection_reasons(batch_output)
        if not rejection_reasons:
            # Paranoia check: stats say threshold not met but no rejection reasons
            logger.warning(
                "No rejection reasons extracted but accept_pct=%.1f%% < threshold. "
                "Exiting loop to avoid infinite loop.",
                accept_pct,
            )
            exit_reason = "max_iterations"
            samples = _collect_accepted(batch_output)
            break

        logger.info(
            "Running CurriculumAgent with %d rejection reasons...",
            len(rejection_reasons),
        )
        curriculum_output = curriculum_agent.run(
            rejection_reasons=rejection_reasons,
            strategy=strategy,
            iteration=iteration,
        )
        curriculum_outputs.append(curriculum_output)

        event_logger.log(
            agent="curriculum",
            iteration=iteration,
            decision=(
                f"identified {len(curriculum_output['gap_categories'])} gap categories, "
                f"priority={curriculum_output['priority_focus']}"
            ),
            reason=curriculum_output["targeted_prompt"][:120] + "...",
            data={
                "gap_categories": curriculum_output["gap_categories"],
                "priority_focus": curriculum_output["priority_focus"],
                "targeted_prompt": curriculum_output["targeted_prompt"],
            },
        )

        # ── Regenerate with targeted prompt ────────────────────────────────────
        logger.info("Regenerating dataset with targeted Curriculum prompt...")
        samples = _generate_samples(
            chunks=chunks,
            intent=strategy["intent"],
            samples_per_chunk=strategy["samples_per_chunk"],
            custom_prompt=curriculum_output["targeted_prompt"],
        )
        logger.info("Regenerated %d samples", len(samples))

    # ── Build and return final state ───────────────────────────────────────────
    all_events = event_logger.read_all()
    final_stats = stats  # last stats from the loop

    return PipelineLoopState(
        iteration=iteration,
        strategy=strategy,
        final_samples=samples,
        final_stats=final_stats,
        curriculum_outputs=curriculum_outputs,
        events=all_events,
        exit_reason=exit_reason,
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _generate_samples(
    chunks: List[str],
    intent: str,
    samples_per_chunk: int,
    custom_prompt: Optional[str],
) -> List[dict]:
    """
    Wrapper around modifai.core.dataset_generation.generate_dataset.

    Ensures every returned sample has a 'chunk_id' field so CriticAgent can
    look up the source chunk text for grounding evaluation. If generate_dataset
    already annotates chunk_id, the post-processing step is a no-op.

    Args:
        chunks:           Raw text chunks (List[str]).
        intent:           "QA" | "instruction" | "tutor"
        samples_per_chunk: Samples to generate per chunk.
        custom_prompt:    CurriculumAgent targeted_prompt (None on first pass).
    """
    from modifai.core.dataset_generation import generate_dataset

    mode_map = {"QA": "QA", "instruction": "instruction", "tutor": "tutor"}
    mode = mode_map.get(intent, "QA")

    samples = generate_dataset(
        chunks=chunks,
        mode=mode,
        samples_per_chunk=samples_per_chunk,
        custom_prompt=custom_prompt,
    )

    # If generate_dataset didn't annotate chunk_id, infer it from sample index.
    # Assumes samples are ordered: samples_per_chunk samples per chunk, in chunk order.
    if samples and "chunk_id" not in samples[0]:
        logger.debug(
            "_generate_samples: annotating chunk_id by index "
            "(samples_per_chunk=%d, total_samples=%d)",
            samples_per_chunk,
            len(samples),
        )
        for i, sample in enumerate(samples):
            sample["chunk_id"] = i // samples_per_chunk

    return samples


def _collect_accepted(batch_output: dict) -> List[dict]:
    """
    Return only the accepted and rewritten samples from a Critic batch output.
    Rewritten samples have their output/response field replaced with rewritten_output.

    Handles both "output" and "response" field names for compatibility with
    samples from different generation sources.
    """
    accepted = []
    for verdict in batch_output.get("verdicts", []):
        if verdict["verdict"] == "reject":
            continue

        sample = verdict.get("original_sample", {}).copy()

        if verdict["verdict"] == "rewrite" and verdict.get("rewritten_output"):
            # Update the response field — handle both naming conventions
            if "output" in sample:
                sample["output"] = verdict["rewritten_output"]
            else:
                sample["response"] = verdict["rewritten_output"]

        accepted.append(sample)
    return accepted
