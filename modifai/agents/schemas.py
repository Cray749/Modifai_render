"""
Shared schema definitions for the Modifai agentic pipeline.
DO NOT MODIFY field names — P2 (infra) and P3 (frontend) depend on these exact keys.
"""
from __future__ import annotations
from typing import TypedDict, Optional, List, Literal


# ── Orchestrator ──────────────────────────────────────────────────────────────

class DocMetadata(TypedDict):
    filename: str
    page_count: int
    domain: str                   # e.g. "customer support", "HR policy", "engineering runbook"
    estimated_chunk_count: int


class OrchestratorInput(TypedDict):
    goal: str
    doc_metadata: DocMetadata


class OrchestratorOutput(TypedDict):
    intent: Literal["QA", "instruction", "tutor"]
    quality_threshold: float      # 0.5 – 0.95
    samples_per_chunk: int        # 3 – 8
    reasoning: str


# ── Critic (defined here for reference; implementation is in critic_agent.py) ──

class CriticVerdict(TypedDict):
    verdict: Literal["accept", "rewrite", "reject"]
    reason: str
    rewritten_output: Optional[str]


class CriticStats(TypedDict):
    total: int
    accepted: int
    rewritten: int
    rejected: int
    accept_pct: float


class CriticBatchOutput(TypedDict):
    verdicts: List[CriticVerdict]
    stats: CriticStats


# ── Curriculum ────────────────────────────────────────────────────────────────

class GapCategory(TypedDict):
    name: str           # snake_case identifier
    description: str
    example_bad: str
    example_good: str


class CurriculumInput(TypedDict):
    rejection_reasons: List[str]
    strategy: OrchestratorOutput
    iteration: int


class CurriculumOutput(TypedDict):
    gap_categories: List[GapCategory]   # len >= 3
    targeted_prompt: str
    priority_focus: str                 # name of the most critical gap


# ── Logging ───────────────────────────────────────────────────────────────────

class AgentEvent(TypedDict):
    event_id: str           # uuid4
    timestamp: str          # ISO 8601 UTC
    agent: Literal["orchestrator", "critic", "curriculum"]
    iteration: int          # 0 = pre-loop (orchestrator), 1–3 = loop
    decision: str           # human-readable one-liner
    reason: Optional[str]
    data: dict              # full payload


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineLoopState(TypedDict):
    iteration: int
    strategy: OrchestratorOutput
    final_samples: List[dict]
    final_stats: CriticStats
    curriculum_outputs: List[CurriculumOutput]
    events: List[AgentEvent]
    exit_reason: Literal["threshold_met", "max_iterations", "all_accepted_first_pass"]
