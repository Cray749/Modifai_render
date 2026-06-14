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
    agent: Literal["orchestrator", "critic", "curriculum", "knowledge", "agent_discovery", "virtual_mind", "automation_discovery"]
    iteration: int          # 0 = pre-loop (orchestrator), 1–3 = loop
    decision: str           # human-readable one-liner
    reason: Optional[str]
    data: dict              # full payload


# ── Knowledge ─────────────────────────────────────────────────────────────────

class KnowledgeDomain(TypedDict):
    name: str
    description: str
    evidence: List[str]
    confidence: float


class ExpertiseArea(TypedDict):
    name: str
    confidence: float


class WorkflowCandidate(TypedDict):
    name: str
    steps: List[str]
    confidence: float


class KnowledgeAnalysisOutput(TypedDict):
    knowledge_summary: str
    domains: List[KnowledgeDomain]
    expertise: List[ExpertiseArea]
    key_concepts: List[str]
    workflows: List[WorkflowCandidate]


# ── Virtual Mind ──────────────────────────────────────────────────────────────

class AgentCapability(TypedDict):
    name: str               # e.g. "Answer HR policy questions"
    description: str        # one-sentence explanation


class DiscoveredAgent(TypedDict):
    name: str               # e.g. "HR Agent"
    description: str        # what this agent does
    specialization: str     # primary domain
    reasoning: str          # justification for this agent's existence
    confidence: float       # confidence score (0.0 - 1.0)
    source_domains: List[str]        # knowledge domains that justify this agent
    source_expertise: List[str]      # expertise areas mapped to this agent
    capabilities: List[AgentCapability]
    starter_questions: List[str]     # 5 intelligent questions related to the agent's capabilities


class VirtualMind(TypedDict):
    name: str               # e.g. "Modifai Virtual Mind"
    description: str        # one-paragraph purpose statement
    domains: List[str]      # deduplicated list of domain names
    key_concepts: List[str]
    agents: List[DiscoveredAgent]


class AgentPackage(TypedDict):
    name: str               # e.g. "HR Agent"
    system_prompt: str
    description: str
    specialization: str
    capabilities: List[AgentCapability]
    knowledge_domains: List[str]
    instructions: List[str]
    starter_questions: List[str]
    version: str


# ── Automation Discovery ──────────────────────────────────────────────────────

class AutomationBlueprint(TypedDict):
    trigger: str            # event that initiates the automation, e.g. "New employee record created"
    actions: List[str]      # ordered automation steps, e.g. ["Create email account", "Assign manager"]


class AutomationOpportunity(TypedDict):
    name: str                               # e.g. "Employee Onboarding Automation"
    description: str                        # one-paragraph purpose
    owning_agent: str                       # name of DiscoveredAgent that owns this automation
    automation_score: int                   # 0-100 automation potential score
    confidence: float                       # 0.0-1.0
    business_impact: Literal["High", "Medium", "Low"]
    impact_reasoning: str                   # why this level of impact
    estimated_benefits: List[str]           # e.g. ["4 hours saved per employee"]
    evidence: List[str]                     # supporting phrases from knowledge
    source_workflow: str                    # workflow name from KnowledgeAnalysisOutput
    source_domains: List[str]
    reasoning: str                          # why this qualifies for automation
    automation_blueprint: AutomationBlueprint


class AutomationDiscoveryOutput(TypedDict):
    executive_summary: str                          # one-paragraph overview for judges and UI
    automation_opportunities: List[AutomationOpportunity]
    total_opportunities: int
    high_impact_count: int


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineLoopState(TypedDict):
    iteration: int
    strategy: OrchestratorOutput
    final_samples: List[dict]
    final_stats: CriticStats
    curriculum_outputs: List[CurriculumOutput]
    events: List[AgentEvent]
    exit_reason: Literal["threshold_met", "max_iterations", "all_accepted_first_pass"]
    knowledge_analysis: Optional[KnowledgeAnalysisOutput]
    virtual_mind: Optional[VirtualMind]
    automation_discovery_output: Optional[AutomationDiscoveryOutput]
