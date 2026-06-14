"""
End-to-end pipeline tests. All AWS calls are mocked.
These tests exercise the full Orchestrator → Critic → Curriculum loop.

Patching strategy:
  - OrchestratorAgent, CriticAgent, CurriculumAgent are patched at the
    pipeline_loop module level (where they are imported), not at the class level.
  - _generate_samples is patched directly so no real Bedrock dataset generation runs.

Exit reason coverage:
  1. all_accepted_first_pass  — 100% accept on iteration 1, Curriculum never called
  2. threshold_met            — passes threshold after 1 curriculum loop (iter 2)
  3. max_iterations           — never meets threshold across 3 iterations
  4. event log written        — JSONL file written with correct schema
  5. improving accept_pct     — 3 iterations, threshold met at iteration 3
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modifai.agents.pipeline_loop import run_agentic_loop


# ── Shared fixtures ────────────────────────────────────────────────────────────

GOAL = "Build a Q&A bot for our customer support runbook"

DOC_METADATA = {
    "filename": "support_runbook.pdf",
    "page_count": 20,
    "domain": "customer support",
    "estimated_chunk_count": 40,
}

SAMPLE_CHUNKS = [
    "Step 1: Open the support portal. Step 2: Navigate to tickets. Step 3: Assign priority.",
    "Refund policy: customers may request refunds within 30 days of purchase.",
    "Escalation path: Tier 1 → Tier 2 → Manager. Use Slack #escalations channel.",
]

MOCK_STRATEGY = {
    "intent": "QA",
    "quality_threshold": 0.70,
    "samples_per_chunk": 4,
    "reasoning": "Support FAQ domain.",
}

MOCK_SAMPLES = [
    {
        "instruction": "What is step 1?",
        "input": SAMPLE_CHUNKS[0],
        "output": "Open the support portal.",
        "chunk_id": 0,
    },
    {
        "instruction": "What is the refund window?",
        "input": SAMPLE_CHUNKS[1],
        "output": "30 days.",
        "chunk_id": 1,
    },
]

GAP_CATEGORIES = [
    {
        "name": "lacks_steps",
        "description": "Skips intermediate steps.",
        "example_bad": "Restart server.",
        "example_good": "1. SSH in. 2. Run restart. 3. Check logs.",
    },
    {
        "name": "too_vague",
        "description": "No specific entity names.",
        "example_bad": "Use the tool.",
        "example_good": "Use Datadog.",
    },
    {
        "name": "factual_drift",
        "description": "Introduces unsupported facts.",
        "example_bad": "Uses Redis (not in source).",
        "example_good": "Uses the cache described in the doc.",
    },
]

MOCK_CURRICULUM_OUTPUT = {
    "gap_categories": GAP_CATEGORIES,
    "targeted_prompt": "Each answer must enumerate all steps. Name specific tools and systems.",
    "priority_focus": "lacks_steps",
}


# ── Helper: build CriticAgent.run_batch return value ─────────────────────────

def _critic_batch_output(accept_pct: float) -> dict:
    """
    Build a fake CriticBatchOutput with the given accept percentage.
    Samples that don't make the accept cut are marked as "reject".
    accept_pct is a percentage (0–100).
    """
    total = len(MOCK_SAMPLES)
    accepted = round(total * accept_pct / 100)
    rejected = total - accepted

    verdicts = []
    for i, sample in enumerate(MOCK_SAMPLES):
        if i < accepted:
            verdicts.append({
                "verdict": "accept",
                "reason": "good quality",
                "rewritten_output": None,
                "original_sample": sample,
            })
        else:
            verdicts.append({
                "verdict": "reject",
                "reason": "too vague, missing steps",
                "rewritten_output": None,
                "original_sample": sample,
            })

    return {
        "verdicts": verdicts,
        "stats": {
            "total": total,
            "accepted": accepted,
            "rewritten": 0,
            "rejected": rejected,
            "accept_pct": accept_pct,
        },
    }


# ── Test 1: All accepted on first pass — Curriculum never called ──────────────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_all_accepted_first_pass_skips_curriculum(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        # Orchestrator returns valid strategy
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY

        # Critic accepts everything (100%)
        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=100.0)

        # Generator returns MOCK_SAMPLES
        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        assert state["exit_reason"] == "all_accepted_first_pass"
        assert state["iteration"] == 1
        assert state["curriculum_outputs"] == []

        # Curriculum.run must never have been called
        mock_curriculum_cls.return_value.run.assert_not_called()

        # Events: 1 orchestrator + 1 critic = exactly 2
        assert len(state["events"]) == 2
        assert state["events"][0]["agent"] == "orchestrator"
        assert state["events"][1]["agent"] == "critic"

        # final_samples = accepted samples only
        assert len(state["final_samples"]) == len(MOCK_SAMPLES)

        # final_stats matches last critic pass
        assert state["final_stats"]["accept_pct"] == 100.0


# ── Test 2: Threshold met after 1 curriculum loop ─────────────────────────────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_loops_once_when_threshold_met_on_second_critic_pass(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY  # threshold=0.70

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        # First pass: 50% accept (below threshold 70%)
        # Second pass: 100% accept (above threshold)
        mock_critic.run_batch.side_effect = [
            _critic_batch_output(accept_pct=50.0),
            _critic_batch_output(accept_pct=100.0),
        ]

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = MOCK_CURRICULUM_OUTPUT
        # extract_rejection_reasons is a staticmethod on CurriculumAgent;
        # it is called via CurriculumAgent.extract_rejection_reasons(batch_output)
        # which resolves to the real static method — no patching needed here.

        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        assert state["exit_reason"] == "threshold_met"
        assert state["iteration"] == 2
        assert len(state["curriculum_outputs"]) == 1

        # Events: orchestrator + critic(iter1) + curriculum(iter1) + critic(iter2) = 4
        assert len(state["events"]) == 4
        agents_in_order = [e["agent"] for e in state["events"]]
        assert agents_in_order == ["orchestrator", "critic", "curriculum", "critic"]

        # final_stats = last (second) critic pass
        assert state["final_stats"]["accept_pct"] == 100.0

        # Generator called twice: initial + after curriculum
        assert mock_gen.call_count == 2
        # Second call must have custom_prompt set from curriculum output
        second_call_kwargs = mock_gen.call_args_list[1][1]
        assert second_call_kwargs["custom_prompt"] == MOCK_CURRICULUM_OUTPUT["targeted_prompt"]


# ── Test 3: Max iterations exhausted ─────────────────────────────────────────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_exits_after_max_iterations(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY  # threshold=0.70

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        # Always 50% — never meets 70% threshold
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=50.0)

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = MOCK_CURRICULUM_OUTPUT

        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        assert state["exit_reason"] == "max_iterations"
        assert state["iteration"] == 3

        # Critic runs 3 times (one per iteration)
        assert mock_critic.run_batch.call_count == 3

        # Curriculum runs 2 times (iter1 and iter2, NOT on iter3 — last iteration)
        assert mock_curriculum.run.call_count == 2
        assert len(state["curriculum_outputs"]) == 2

        # Generator: 1 initial + 2 curriculum-driven = 3 total
        assert mock_gen.call_count == 3


# ── Test 4: Event log written to disk with correct schema ─────────────────────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_event_log_written_to_disk(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=100.0)

        mock_gen.return_value = MOCK_SAMPLES

        run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        log_path = Path(event_log)
        assert log_path.exists(), "Event log file was not created"

        lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) >= 2, "Expected at least 2 events (orchestrator + critic)"

        required_keys = {"event_id", "timestamp", "agent", "iteration", "decision", "data"}
        valid_agents = {"orchestrator", "critic", "curriculum"}

        for line in lines:
            event = json.loads(line)
            assert required_keys.issubset(event.keys()), (
                f"Event missing required keys. Got: {list(event.keys())}"
            )
            assert event["agent"] in valid_agents, f"Unknown agent: {event['agent']}"
            assert isinstance(event["iteration"], int)
            assert isinstance(event["decision"], str) and len(event["decision"]) > 0
            assert isinstance(event["data"], dict)

        # Orchestrator event must have iteration=0
        orch_events = [e for e in [json.loads(l) for l in lines] if e["agent"] == "orchestrator"]
        assert len(orch_events) == 1
        assert orch_events[0]["iteration"] == 0


# ── Test 5: Accept pct improves across 3 iterations → resolves at iter 3 ─────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_accept_pct_increases_across_iterations(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    """Verifies the loop runs Curriculum twice and resolves at iteration 3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        # Set threshold to 0.90 so 3 iterations are needed
        mock_orch.run.return_value = {**MOCK_STRATEGY, "quality_threshold": 0.90}

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.side_effect = [
            _critic_batch_output(accept_pct=40.0),   # iter 1: below 90%
            _critic_batch_output(accept_pct=65.0),   # iter 2: still below 90%
            _critic_batch_output(accept_pct=100.0),  # iter 3: meets 90% threshold
        ]

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = MOCK_CURRICULUM_OUTPUT

        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        # Resolved before max_iterations
        assert state["exit_reason"] == "threshold_met"
        assert state["iteration"] == 3

        # Curriculum ran twice (after iter 1 and iter 2)
        assert mock_curriculum.run.call_count == 2
        assert len(state["curriculum_outputs"]) == 2

        # Final stats reflect iter 3 result
        assert state["final_stats"]["accept_pct"] == 100.0

        # Check event log: orch + 3 critic + 2 curriculum = 6 events
        critic_events = [e for e in state["events"] if e["agent"] == "critic"]
        curriculum_events = [e for e in state["events"] if e["agent"] == "curriculum"]
        assert len(critic_events) == 3
        assert len(curriculum_events) == 2

        # Verify critic event iteration numbers
        critic_iters = [e["iteration"] for e in critic_events]
        assert critic_iters == [1, 2, 3]

        # Generator called: 1 initial + 2 curriculum-prompted = 3 times
        assert mock_gen.call_count == 3
        # First call: custom_prompt=None
        assert mock_gen.call_args_list[0][1]["custom_prompt"] is None
        # Second and third calls: custom_prompt from curriculum
        assert mock_gen.call_args_list[1][1]["custom_prompt"] == MOCK_CURRICULUM_OUTPUT["targeted_prompt"]
        assert mock_gen.call_args_list[2][1]["custom_prompt"] == MOCK_CURRICULUM_OUTPUT["targeted_prompt"]
