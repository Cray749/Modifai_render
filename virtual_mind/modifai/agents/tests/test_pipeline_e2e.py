"""
End-to-end pipeline tests. All AWS calls are mocked.
These tests exercise the full Orchestrator → Critic → Curriculum → Knowledge → AgentDiscovery → VirtualMind → AutomationDiscovery loop.

Patching strategy:
  - All agent classes are patched at the pipeline_loop module level (where they are imported).
  - VirtualMindBuilder is patched at the pipeline_loop module level.
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

MOCK_KNOWLEDGE_OUTPUT = {
    "knowledge_summary": "Summary of HR.",
    "domains": [{"name": "HR", "description": "Human Resources", "evidence": ["Employee handbook"], "confidence": 0.95}],
    "expertise": [{"name": "Onboarding", "confidence": 0.9}],
    "key_concepts": ["Onboarding"],
    "workflows": [{"name": "Onboarding", "steps": ["Step 1", "Step 2"], "confidence": 0.85}]
}

MOCK_DISCOVERED_AGENTS = [
    {
        "name": "HR Agent",
        "description": "Handles HR queries.",
        "specialization": "Human Resources",
        "reasoning": "Strong evidence.",
        "confidence": 0.92,
        "source_domains": ["HR"],
        "source_expertise": ["Onboarding"],
        "capabilities": [
            {"name": "Answer HR questions", "description": "Responds to HR policy queries."},
        ],
    }
]

MOCK_VIRTUAL_MIND = {
    "name": "Modifai Virtual Mind",
    "description": "A virtual mind for the organization.",
    "domains": ["HR"],
    "key_concepts": ["Onboarding"],
    "agents": MOCK_DISCOVERED_AGENTS,
}

MOCK_AUTOMATION_OUTPUT = {
    "executive_summary": "1 automation opportunity identified in HR with high business impact.",
    "automation_opportunities": [
        {
            "name": "Onboarding Automation",
            "description": "Automates the new hire onboarding workflow.",
            "owning_agent": "HR Agent",
            "automation_score": 88,
            "confidence": 0.90,
            "business_impact": "High",
            "impact_reasoning": "Repetitive, multi-step process with strong documentation.",
            "estimated_benefits": ["3 hours saved per hire (estimated)", "Faster account provisioning"],
            "evidence": ["Employee handbook"],
            "source_workflow": "Onboarding",
            "source_domains": ["HR"],
            "reasoning": "Well-documented, standardised 2-step workflow.",
            "automation_blueprint": {
                "trigger": "New employee record created",
                "actions": ["Create accounts", "Schedule orientation"],
            },
        }
    ],
    "total_opportunities": 1,
    "high_impact_count": 1,
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

@patch("modifai.agents.pipeline_loop.AutomationDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.VirtualMindBuilder")
@patch("modifai.agents.pipeline_loop.AgentDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.KnowledgeAgent")
@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_all_accepted_first_pass_skips_curriculum(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen,
    mock_knowledge_cls, mock_discovery_cls, mock_builder_cls, mock_automation_cls
):
    """Test that curriculum is skipped if all samples pass the threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=100.0)

        mock_knowledge = MagicMock()
        mock_knowledge_cls.return_value = mock_knowledge
        mock_knowledge.run.return_value = MOCK_KNOWLEDGE_OUTPUT

        mock_discovery = MagicMock()
        mock_discovery_cls.return_value = mock_discovery
        mock_discovery.run.return_value = MOCK_DISCOVERED_AGENTS

        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_builder.build.return_value = MOCK_VIRTUAL_MIND

        mock_automation_cls.return_value.run.return_value = MOCK_AUTOMATION_OUTPUT

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
        assert state["knowledge_analysis"] == MOCK_KNOWLEDGE_OUTPUT
        assert state["virtual_mind"] == MOCK_VIRTUAL_MIND
        assert state["virtual_mind"]["agents"][0]["name"] == "HR Agent"
        assert state["automation_discovery_output"] == MOCK_AUTOMATION_OUTPUT

        mock_curriculum_cls.return_value.run.assert_not_called()

        # Events: orchestrator + critic + knowledge + agent_discovery + virtual_mind + automation_discovery = 6
        assert len(state["events"]) == 6
        agents_in_order = [e["agent"] for e in state["events"]]
        assert agents_in_order == [
            "orchestrator", "critic", "knowledge", "agent_discovery",
            "virtual_mind", "automation_discovery"
        ]

        assert len(state["final_samples"]) == len(MOCK_SAMPLES)
        assert state["final_stats"]["accept_pct"] == 100.0


# ── Test 2: Threshold met after 1 curriculum loop ─────────────────────────────

@patch("modifai.agents.pipeline_loop.AutomationDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.VirtualMindBuilder")
@patch("modifai.agents.pipeline_loop.AgentDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.KnowledgeAgent")
@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_loops_once_when_threshold_met_on_second_critic_pass(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen,
    mock_knowledge_cls, mock_discovery_cls, mock_builder_cls, mock_automation_cls
):
    """Test that the loop exits early once the quality threshold is met."""
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.side_effect = [
            _critic_batch_output(accept_pct=50.0),
            _critic_batch_output(accept_pct=100.0),
        ]

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = MOCK_CURRICULUM_OUTPUT

        mock_knowledge = MagicMock()
        mock_knowledge_cls.return_value = mock_knowledge
        mock_knowledge.run.return_value = MOCK_KNOWLEDGE_OUTPUT

        mock_discovery_cls.return_value.run.return_value = MOCK_DISCOVERED_AGENTS
        mock_builder_cls.return_value.build.return_value = MOCK_VIRTUAL_MIND
        mock_automation_cls.return_value.run.return_value = MOCK_AUTOMATION_OUTPUT

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
        assert state["virtual_mind"] == MOCK_VIRTUAL_MIND
        assert state["automation_discovery_output"] == MOCK_AUTOMATION_OUTPUT

        # Events: orch + critic + curriculum + critic + knowledge + discovery + virtual_mind + automation_discovery = 8
        assert len(state["events"]) == 8
        agents_in_order = [e["agent"] for e in state["events"]]
        assert agents_in_order == [
            "orchestrator", "critic", "curriculum", "critic",
            "knowledge", "agent_discovery", "virtual_mind", "automation_discovery"
        ]

        assert state["final_stats"]["accept_pct"] == 100.0
        assert mock_gen.call_count == 2
        second_call_kwargs = mock_gen.call_args_list[1][1]
        assert second_call_kwargs["custom_prompt"] == MOCK_CURRICULUM_OUTPUT["targeted_prompt"]


# ── Test 3: Max iterations exhausted ─────────────────────────────────────────

@patch("modifai.agents.pipeline_loop.AutomationDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.VirtualMindBuilder")
@patch("modifai.agents.pipeline_loop.AgentDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.KnowledgeAgent")
@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_exits_after_max_iterations(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen,
    mock_knowledge_cls, mock_discovery_cls, mock_builder_cls, mock_automation_cls
):
    """Test that the loop stops and exits when max iterations are exhausted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=50.0)

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = MOCK_CURRICULUM_OUTPUT

        mock_knowledge_cls.return_value.run.return_value = MOCK_KNOWLEDGE_OUTPUT
        mock_discovery_cls.return_value.run.return_value = MOCK_DISCOVERED_AGENTS
        mock_builder_cls.return_value.build.return_value = MOCK_VIRTUAL_MIND
        mock_automation_cls.return_value.run.return_value = MOCK_AUTOMATION_OUTPUT

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
        assert mock_critic.run_batch.call_count == 3
        assert mock_curriculum.run.call_count == 2
        assert len(state["curriculum_outputs"]) == 2
        assert mock_gen.call_count == 3
        assert state["virtual_mind"] == MOCK_VIRTUAL_MIND
        assert state["automation_discovery_output"] == MOCK_AUTOMATION_OUTPUT


# ── Test 4: Event log written to disk with correct schema ─────────────────────

@patch("modifai.agents.pipeline_loop.AutomationDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.VirtualMindBuilder")
@patch("modifai.agents.pipeline_loop.AgentDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.KnowledgeAgent")
@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_event_log_written_to_disk(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen,
    mock_knowledge_cls, mock_discovery_cls, mock_builder_cls, mock_automation_cls
):
    """Test that the agent events are logged correctly to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=100.0)

        mock_knowledge_cls.return_value.run.return_value = MOCK_KNOWLEDGE_OUTPUT
        mock_discovery_cls.return_value.run.return_value = MOCK_DISCOVERED_AGENTS
        mock_builder_cls.return_value.build.return_value = MOCK_VIRTUAL_MIND
        mock_automation_cls.return_value.run.return_value = MOCK_AUTOMATION_OUTPUT
        mock_curriculum_cls.return_value.run.return_value = MOCK_CURRICULUM_OUTPUT

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
        assert len(lines) >= 6, "Expected at least 6 events (now includes automation_discovery)"

        required_keys = {"event_id", "timestamp", "agent", "iteration", "decision", "data"}
        valid_agents = {
            "orchestrator", "critic", "curriculum", "knowledge",
            "agent_discovery", "virtual_mind", "automation_discovery"
        }

        for line in lines:
            event = json.loads(line)
            assert required_keys.issubset(event.keys())
            assert event["agent"] in valid_agents, f"Unknown agent: {event['agent']}"
            assert isinstance(event["iteration"], int)
            assert isinstance(event["decision"], str) and len(event["decision"]) > 0
            assert isinstance(event["data"], dict)

        orch_events = [e for e in [json.loads(l) for l in lines] if e["agent"] == "orchestrator"]
        assert len(orch_events) == 1
        assert orch_events[0]["iteration"] == 0

        automation_events = [e for e in [json.loads(l) for l in lines] if e["agent"] == "automation_discovery"]
        assert len(automation_events) == 1


# ── Test 5: Accept pct improves across 3 iterations → resolves at iter 3 ─────

@patch("modifai.agents.pipeline_loop.AutomationDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.VirtualMindBuilder")
@patch("modifai.agents.pipeline_loop.AgentDiscoveryAgent")
@patch("modifai.agents.pipeline_loop.KnowledgeAgent")
@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_accept_pct_increases_across_iterations(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen,
    mock_knowledge_cls, mock_discovery_cls, mock_builder_cls, mock_automation_cls
):
    """Verifies the loop runs Curriculum twice and resolves at iteration 3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = {**MOCK_STRATEGY, "quality_threshold": 0.90}

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.side_effect = [
            _critic_batch_output(accept_pct=40.0),
            _critic_batch_output(accept_pct=65.0),
            _critic_batch_output(accept_pct=100.0),
        ]

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = MOCK_CURRICULUM_OUTPUT

        mock_knowledge_cls.return_value.run.return_value = MOCK_KNOWLEDGE_OUTPUT
        mock_discovery_cls.return_value.run.return_value = MOCK_DISCOVERED_AGENTS
        mock_builder_cls.return_value.build.return_value = MOCK_VIRTUAL_MIND
        mock_automation_cls.return_value.run.return_value = MOCK_AUTOMATION_OUTPUT

        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        assert state["exit_reason"] == "threshold_met"
        assert state["iteration"] == 3
        assert mock_curriculum.run.call_count == 2
        assert len(state["curriculum_outputs"]) == 2
        assert state["final_stats"]["accept_pct"] == 100.0
        assert state["virtual_mind"] == MOCK_VIRTUAL_MIND
        assert state["automation_discovery_output"] == MOCK_AUTOMATION_OUTPUT

        # orch + 3 critic + 2 curriculum + knowledge + agent_discovery + virtual_mind + automation_discovery = 10
        critic_events = [e for e in state["events"] if e["agent"] == "critic"]
        curriculum_events = [e for e in state["events"] if e["agent"] == "curriculum"]
        automation_events = [e for e in state["events"] if e["agent"] == "automation_discovery"]
        assert len(critic_events) == 3
        assert len(curriculum_events) == 2
        assert len(automation_events) == 1
        assert len(state["events"]) == 10

        critic_iters = [e["iteration"] for e in critic_events]
        assert critic_iters == [1, 2, 3]

        assert mock_gen.call_count == 3
        assert mock_gen.call_args_list[0][1]["custom_prompt"] is None
        assert mock_gen.call_args_list[1][1]["custom_prompt"] == MOCK_CURRICULUM_OUTPUT["targeted_prompt"]
        assert mock_gen.call_args_list[2][1]["custom_prompt"] == MOCK_CURRICULUM_OUTPUT["targeted_prompt"]
