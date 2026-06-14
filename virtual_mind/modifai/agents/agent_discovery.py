"""
AgentDiscoveryAgent — transforms KnowledgeAnalysisOutput into a list of DiscoveredAgents.

Input:  KnowledgeAnalysisOutput (Sprint 1 output)
Output: List[DiscoveredAgent]

This agent consumes ONLY structured knowledge.  It does NOT read documents or chunks.
It uses the Bedrock Converse API with forced tool-use to ensure deterministic output.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, cast

from modifai.core.llm_provider import get_llm_provider

from modifai.agents.schemas import (
    KnowledgeAnalysisOutput,
    DiscoveredAgent,
)

logger = logging.getLogger(__name__)

# ── Bedrock tool definition ────────────────────────────────────────────────────

_TOOL_SPEC = {
    "toolSpec": {
        "name": "discover_agents",
        "description": (
            "Given a structured knowledge analysis, discover what specialized AI agents "
            "should be created to represent the organization's intelligence.  "
            "Return one agent per major domain or department."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Short display name, e.g. 'HR Agent'"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "One sentence describing what this agent does"
                                },
                                "specialization": {
                                    "type": "string",
                                    "description": "Primary domain this agent covers"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Justification for this agent's existence based on evidence"
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence score for generating this agent (0.0 to 1.0)"
                                },
                                "source_domains": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Domain names from knowledge analysis that justify this agent"
                                },
                                "source_expertise": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Expertise area names mapped to this agent"
                                },
                                "capabilities": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "Short capability label"
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "One sentence description of the capability"
                                            }
                                        },
                                        "required": ["name", "description"]
                                    }
                                },
                                "starter_questions": {
                                    "type": "array",
                                    "minItems": 5,
                                    "maxItems": 5,
                                    "items": {"type": "string"},
                                    "description": "Exactly 5 intelligent, concrete questions a user might ask this agent based on its capabilities."
                                }
                            },
                            "required": [
                                "name", "description", "specialization", "reasoning", "confidence",
                                "source_domains", "source_expertise", "capabilities", "starter_questions"
                            ]
                        }
                    }
                },
                "required": ["agents"]
            }
        }
    }
}

_SYSTEM_PROMPT = """\
You are the Agent Discovery component of Modifai, an automated Virtual Mind generation platform.

You receive structured organizational intelligence (knowledge summary, domains, expertise areas, \
key concepts, and workflows) previously extracted from a document set.

Your task:
  1. Identify distinct areas of organizational knowledge.
  2. Determine what specialized AI agent should represent each area.
  3. Generate a DiscoveredAgent for each area with a name, description, specialization, \
the source knowledge that justifies it, concrete capabilities, and 5 starter questions.

Rules:
  - Create one agent per major domain unless two domains are nearly identical.
  - Agent names must follow the pattern "<Domain> Agent" (e.g. "HR Agent", "Engineering Agent").
  - Each agent must have at least 2 capabilities grounded in the source knowledge.
  - Provide solid reasoning and a confidence score for each generated agent.
  - Output ONLY via the discover_agents tool call — no free-form text.
"""


class AgentDiscoveryAgent:
    """
    Discovers what specialized agents should exist, based purely on KnowledgeAnalysisOutput.

    Usage:
        agent = AgentDiscoveryAgent()
        discovered = agent.run(knowledge=knowledge_output)
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

    def run(self, knowledge: KnowledgeAnalysisOutput) -> List[DiscoveredAgent]:
        """
        Discover specialized agents from structured knowledge.

        Args:
            knowledge: KnowledgeAnalysisOutput produced by Sprint 1 KnowledgeAgent.

        Returns:
            List[DiscoveredAgent] — one agent per major knowledge area.
        """
        user_message = self._build_user_message(knowledge)
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
                agents = output.get("agents", [])
                self._validate(agents)
                logger.info(
                    "AgentDiscovery discovered %d specialized agents from %d domains.",
                    len(agents),
                    len(knowledge.get("domains", [])),
                )
                return cast(List[DiscoveredAgent], agents)
            except (KeyError, ValueError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise ValueError(
                        f"AgentDiscoveryAgent failed after {self.max_retries + 1} attempt(s): {exc}"
                    ) from exc
                logger.warning(
                    "AgentDiscovery output malformed (attempt %d/%d): %s — retrying",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
        # Unreachable; satisfies mypy missing-return check
        raise ValueError("AgentDiscoveryAgent exhausted retries without a valid result.")

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _build_user_message(self, knowledge: KnowledgeAnalysisOutput) -> str:
        """Serialize the KnowledgeAnalysisOutput into a clear prompt."""
        domains_str = "\n".join(
            f"  - {d['name']}: {d['description']}"
            for d in knowledge.get("domains", [])
        )
        expertise_str = "\n".join(
            f"  - {e['name']} (confidence: {e['confidence']:.2f})"
            for e in knowledge.get("expertise", [])
        )
        summary_str = knowledge.get("knowledge_summary", "No summary provided")
        concepts_str = ", ".join(knowledge.get("key_concepts", [])) or "None identified"
        workflows_str = "\n".join(
            f"  - {w['name']}: {' → '.join(w['steps'])}"
            for w in knowledge.get("workflows", [])
        )

        return (
            "## Organizational Knowledge Analysis\n\n"
            f"### Knowledge Summary\n{summary_str}\n\n"
            f"### Knowledge Domains\n{domains_str or '  (none)'}\n\n"
            f"### Expertise Areas\n{expertise_str or '  (none)'}\n\n"
            f"### Key Concepts\n{concepts_str}\n\n"
            f"### Workflows\n{workflows_str or '  (none)'}\n\n"
            "Based on this knowledge, discover what specialized agents should be created. "
            "Call the discover_agents tool with your findings."
        )

    def _validate(self, agents: list) -> None:
        """Validate that the returned agents list is well-formed."""
        if not isinstance(agents, list) or len(agents) == 0:
            raise ValueError("agents must be a non-empty list")
        required = {"name", "description", "specialization", "reasoning", "confidence",
                    "source_domains", "source_expertise", "capabilities", "starter_questions"}
        for i, agent in enumerate(agents):
            missing = required - set(agent.keys())
            if missing:
                raise ValueError(f"Agent[{i}] missing fields: {missing}")
            if not isinstance(agent.get("capabilities"), list) or len(agent["capabilities"]) == 0:
                raise ValueError(f"Agent[{i}] must have at least one capability")
            if not isinstance(agent.get("starter_questions"), list) or len(agent["starter_questions"]) != 5:
                raise ValueError(f"Agent[{i}] must have exactly 5 starter questions")
            if not (0.0 <= agent.get("confidence", -1) <= 1.0):
                raise ValueError(f"Agent[{i}] confidence must be between 0.0 and 1.0")
