"""
AutomationDiscoveryAgent — transforms KnowledgeAnalysisOutput + VirtualMind
into an AutomationDiscoveryOutput containing ranked automation opportunities.

Input:  KnowledgeAnalysisOutput (Sprint 1)  +  VirtualMind (Sprint 2)
Output: AutomationDiscoveryOutput

This agent consumes ONLY structured knowledge. It does NOT read documents or chunks.
It uses the Bedrock Converse API with forced tool-use to ensure deterministic output.

Every discovered automation includes:
  - automation_score    (0-100)
  - confidence          (0.0-1.0)
  - business_impact     (High / Medium / Low)
  - impact_reasoning    (plain English justification)
  - estimated_benefits  (plausible, grounded benefit list)
  - evidence            (phrases from domain knowledge)
  - automation_blueprint (trigger + ordered actions)
  - owning_agent        (must match a DiscoveredAgent name from VirtualMind)
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, cast

from modifai.core.llm_provider import get_llm_provider

from modifai.agents.schemas import (
    KnowledgeAnalysisOutput,
    VirtualMind,
    AutomationDiscoveryOutput,
    AutomationOpportunity,
)

logger = logging.getLogger(__name__)

# ── Bedrock tool definition ────────────────────────────────────────────────────

_TOOL_SPEC = {
    "toolSpec": {
        "name": "discover_automations",
        "description": (
            "Analyze structured organizational knowledge and Virtual Mind agents to discover "
            "automation opportunities. Return a ranked catalog with blueprints, ownership, "
            "and business value estimates."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "executive_summary": {
                        "type": "string",
                        "description": (
                            "One paragraph overview of all discovered automations, "
                            "suitable for an executive or judge audience."
                        )
                    },
                    "automation_opportunities": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Short name, e.g. 'Employee Onboarding Automation'"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "One paragraph describing what this automation does"
                                },
                                "owning_agent": {
                                    "type": "string",
                                    "description": (
                                        "Exact name of the DiscoveredAgent that owns this automation. "
                                        "MUST match one of the agent names provided."
                                    )
                                },
                                "automation_score": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100,
                                    "description": (
                                        "Automation potential score 0-100. Score higher for: "
                                        "repetitive processes (25pts), many steps ≥4 (20pts), "
                                        "strong evidence/documentation (20pts), "
                                        "high standardisation (20pts), cross-domain reuse (15pts)."
                                    )
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Confidence that this is a genuine automation opportunity (0.0-1.0)"
                                },
                                "business_impact": {
                                    "type": "string",
                                    "enum": ["High", "Medium", "Low"],
                                    "description": "High if score >= 75, Medium if >= 45, Low otherwise"
                                },
                                "impact_reasoning": {
                                    "type": "string",
                                    "description": "Plain English explanation of why this automation has this impact level"
                                },
                                "estimated_benefits": {
                                    "type": "array",
                                    "minItems": 2,
                                    "items": {"type": "string"},
                                    "description": (
                                        "2-4 plausible, grounded benefits. "
                                        "Prefix with quantities where possible, e.g. '4 hours saved per employee', "
                                        "'60% reduction in manual approvals'. Mark as 'estimated' if uncertain."
                                    )
                                },
                                "evidence": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                    "description": "Supporting phrases drawn from domain evidence or workflow steps"
                                },
                                "source_workflow": {
                                    "type": "string",
                                    "description": "Exact name of the WorkflowCandidate this automation is based on"
                                },
                                "source_domains": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Domain names from knowledge analysis this automation belongs to"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Why this workflow qualifies as an automation opportunity"
                                },
                                "automation_blueprint": {
                                    "type": "object",
                                    "properties": {
                                        "trigger": {
                                            "type": "string",
                                            "description": "The specific event that initiates this automation"
                                        },
                                        "actions": {
                                            "type": "array",
                                            "minItems": 2,
                                            "items": {"type": "string"},
                                            "description": "Ordered list of 3-6 concrete automation steps"
                                        }
                                    },
                                    "required": ["trigger", "actions"]
                                }
                            },
                            "required": [
                                "name", "description", "owning_agent", "automation_score",
                                "confidence", "business_impact", "impact_reasoning",
                                "estimated_benefits", "evidence", "source_workflow",
                                "source_domains", "reasoning", "automation_blueprint"
                            ]
                        }
                    }
                },
                "required": ["executive_summary", "automation_opportunities"]
            }
        }
    }
}

_SYSTEM_PROMPT = """\
You are the Automation Discovery engine for Modifai, an enterprise knowledge intelligence platform.

You receive structured organizational intelligence: workflows, domains, expertise areas, \
key concepts, and a Virtual Mind with specialized agents.

Your task:
  1. Analyze every workflow candidate as a potential automation opportunity.
  2. Score each automation 0-100 based on:
       - Repetitiveness signals in the workflow name and steps (+25)
       - Step count >= 4 is a strong automation signal (+20)
       - Documentation quality: more domain evidence phrases = better (+20)
       - Standardisation: fewer human-decision steps = more automatable (+20)
       - Cross-domain reuse potential (+15)
  3. Assign each automation to the best-matching agent (owning_agent MUST exactly \
match one of the agent names provided).
  4. Generate a concrete automation_blueprint with one specific trigger event \
and 3-6 ordered action steps that describe what the automation would do.
  5. Estimate realistic business benefits grounded in the workflow steps and domain evidence. \
Use quantities where possible ("X hours saved", "Y% reduction"). Mark as "estimated" if uncertain.
  6. Assign business_impact: High (score >= 75), Medium (score >= 45), Low otherwise.
  7. Write an executive_summary that a judge can read in 10 seconds to understand \
the total automation value discovered.

Rules:
  - owning_agent: MUST be an exact name from the Available Agents list.
  - evidence: MUST include at least one phrase drawn from domain evidence or workflow steps.
  - automation_blueprint.actions: 3-6 ordered, concrete steps (not abstract).
  - estimated_benefits: 2-4 items, plausible and defensible.
  - Output ONLY via the discover_automations tool call — no free-form text.
"""


class AutomationDiscoveryAgent:
    """
    Discovers automation opportunities from structured knowledge and Virtual Mind.

    Input:  KnowledgeAnalysisOutput  +  VirtualMind
    Output: AutomationDiscoveryOutput

    This agent does NOT read documents or chunks. It consumes only the
    structured outputs produced by KnowledgeAgent and VirtualMindBuilder.

    Usage:
        agent = AutomationDiscoveryAgent()
        output = agent.run(knowledge=knowledge_output, virtual_mind=virtual_mind)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_retries: int = 1,
    ):
        self.max_retries = max_retries
        self.provider = get_llm_provider()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        knowledge: KnowledgeAnalysisOutput,
        virtual_mind: VirtualMind,
    ) -> AutomationDiscoveryOutput:
        """
        Discover automation opportunities from structured knowledge and Virtual Mind.

        Args:
            knowledge:    KnowledgeAnalysisOutput from the KnowledgeAgent.
            virtual_mind: VirtualMind from VirtualMindBuilder (provides agent names).

        Returns:
            AutomationDiscoveryOutput with ranked automation catalog.
        """
        user_message = self._build_user_message(knowledge, virtual_mind)
        agent_names = [a["name"] for a in virtual_mind.get("agents", [])]
        attempt = 0

        while attempt <= self.max_retries:
            try:
                schema = _TOOL_SPEC["toolSpec"]["inputSchema"]["json"]
                tool_name = _TOOL_SPEC["toolSpec"]["name"]
                output = self.provider.generate(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_message,
                    response_schema=schema,
                    tool_name=tool_name,
                )
                self._validate(output, agent_names)

                # Compute derived counts from actual opportunities
                opportunities = output["automation_opportunities"]
                output["total_opportunities"] = len(opportunities)
                output["high_impact_count"] = sum(
                    1 for o in opportunities if o["business_impact"] == "High"
                )

                logger.info(
                    "AutomationDiscovery found %d opportunities (%d high-impact) across %d workflows.",
                    output["total_opportunities"],
                    output["high_impact_count"],
                    len(knowledge.get("workflows", [])),
                )
                return cast(AutomationDiscoveryOutput, output)

            except (KeyError, ValueError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise ValueError(
                        f"AutomationDiscoveryAgent failed after {self.max_retries + 1} attempt(s): {exc}"
                    ) from exc
                logger.warning(
                    "AutomationDiscovery output malformed (attempt %d/%d): %s — retrying",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )

        # Unreachable; satisfies mypy missing-return check
        raise ValueError("AutomationDiscoveryAgent exhausted retries without a valid result.")

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _build_user_message(
        self,
        knowledge: KnowledgeAnalysisOutput,
        virtual_mind: VirtualMind,
    ) -> str:
        """Serialize knowledge + virtual mind into a clear, structured prompt."""
        workflows_str = "\n".join(
            f"  - {w['name']} (confidence: {w['confidence']:.2f}, {len(w['steps'])} steps)\n"
            f"    Steps: {' → '.join(w['steps'])}"
            for w in knowledge.get("workflows", [])
        )
        domains_str = "\n".join(
            f"  - {d['name']} (confidence: {d['confidence']:.2f})\n"
            f"    Evidence: {', '.join(d['evidence'])}"
            for d in knowledge.get("domains", [])
        )
        agents_str = "\n".join(
            f"  - {a['name']} (specialization: {a['specialization']}, "
            f"domains: {', '.join(a['source_domains'])})"
            for a in virtual_mind.get("agents", [])
        )
        expertise_str = "\n".join(
            f"  - {e['name']} (confidence: {e['confidence']:.2f})"
            for e in knowledge.get("expertise", [])
        )
        concepts_str = ", ".join(knowledge.get("key_concepts", [])) or "None identified"

        return (
            "## Organizational Knowledge for Automation Discovery\n\n"
            f"### Knowledge Summary\n{knowledge.get('knowledge_summary', 'N/A')}\n\n"
            f"### Workflows (Primary Automation Candidates)\n{workflows_str or '  (none)'}\n\n"
            f"### Domain Evidence\n{domains_str or '  (none)'}\n\n"
            f"### Expertise Areas\n{expertise_str or '  (none)'}\n\n"
            f"### Available Agents (owning_agent MUST match one of these names exactly)\n"
            f"{agents_str or '  (none)'}\n\n"
            f"### Key Concepts\n{concepts_str}\n\n"
            "Analyze the workflows and discover what should be automated. "
            "Call the discover_automations tool with your findings."
        )

    def _validate(self, output: dict, agent_names: List[str]) -> None:
        """Validate the AutomationDiscoveryOutput is well-formed."""
        if not isinstance(output.get("executive_summary"), str) or not output["executive_summary"].strip():
            raise ValueError("executive_summary must be a non-empty string")

        opportunities = output.get("automation_opportunities")
        if not isinstance(opportunities, list) or len(opportunities) == 0:
            raise ValueError("automation_opportunities must be a non-empty list")

        required_fields = {
            "name", "description", "owning_agent", "automation_score", "confidence",
            "business_impact", "impact_reasoning", "estimated_benefits", "evidence",
            "source_workflow", "source_domains", "reasoning", "automation_blueprint"
        }
        valid_impact_levels = {"High", "Medium", "Low"}

        for i, opp in enumerate(opportunities):
            missing = required_fields - set(opp.keys())
            if missing:
                raise ValueError(f"AutomationOpportunity[{i}] missing fields: {missing}")

            score = opp.get("automation_score")
            if not isinstance(score, int) or not (0 <= score <= 100):
                raise ValueError(
                    f"AutomationOpportunity[{i}] automation_score must be int 0-100, got {score!r}"
                )

            conf = opp.get("confidence", -1)
            if not (0.0 <= conf <= 1.0):
                raise ValueError(
                    f"AutomationOpportunity[{i}] confidence must be 0.0-1.0, got {conf!r}"
                )

            if opp.get("business_impact") not in valid_impact_levels:
                raise ValueError(
                    f"AutomationOpportunity[{i}] business_impact must be one of {valid_impact_levels}"
                )

            if agent_names and opp.get("owning_agent") not in agent_names:
                raise ValueError(
                    f"AutomationOpportunity[{i}] owning_agent '{opp.get('owning_agent')}' "
                    f"not found in Virtual Mind agents: {agent_names}"
                )

            if not isinstance(opp.get("evidence"), list) or len(opp["evidence"]) == 0:
                raise ValueError(f"AutomationOpportunity[{i}] evidence must be a non-empty list")

            if not isinstance(opp.get("estimated_benefits"), list) or len(opp["estimated_benefits"]) == 0:
                raise ValueError(
                    f"AutomationOpportunity[{i}] estimated_benefits must be a non-empty list"
                )

            blueprint = opp.get("automation_blueprint", {})
            if not isinstance(blueprint.get("trigger"), str) or not blueprint["trigger"].strip():
                raise ValueError(
                    f"AutomationOpportunity[{i}] automation_blueprint.trigger must be a non-empty string"
                )
            if not isinstance(blueprint.get("actions"), list) or len(blueprint["actions"]) < 2:
                raise ValueError(
                    f"AutomationOpportunity[{i}] automation_blueprint.actions must have >= 2 items"
                )
