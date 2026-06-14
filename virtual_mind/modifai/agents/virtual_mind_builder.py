"""
VirtualMindBuilder — assembles KnowledgeAnalysisOutput + List[DiscoveredAgent]
into a complete VirtualMind artifact.

This is a pure Python transformation (no LLM call required).
All intelligence is already embedded in the knowledge and discovered agents.

Input:
    knowledge:  KnowledgeAnalysisOutput  (Sprint 1)
    agents:     List[DiscoveredAgent]    (Sprint 2, AgentDiscoveryAgent)

Output:
    VirtualMind
"""
from __future__ import annotations

import logging
from typing import List

from modifai.agents.schemas import (
    DiscoveredAgent,
    KnowledgeAnalysisOutput,
    VirtualMind,
)

logger = logging.getLogger(__name__)


class VirtualMindBuilder:
    """
    Assembles discovered agents and knowledge into a VirtualMind artifact.

    This builder is a pure data transformer — it requires no Bedrock call.
    All input data has already been validated by KnowledgeAgent and
    AgentDiscoveryAgent respectively.

    Usage:
        builder = VirtualMindBuilder()
        virtual_mind = builder.build(
            knowledge=knowledge_output,
            agents=discovered_agents,
            mind_name="Acme Corp Virtual Mind",
        )
    """

    def build(
        self,
        knowledge: KnowledgeAnalysisOutput,
        agents: List[DiscoveredAgent],
        mind_name: str = "Modifai Virtual Mind",
    ) -> VirtualMind:
        """
        Assemble the VirtualMind from Sprint 1 knowledge and Sprint 2 agents.

        Args:
            knowledge:  KnowledgeAnalysisOutput from the KnowledgeAgent.
            agents:     List[DiscoveredAgent] from the AgentDiscoveryAgent.
            mind_name:  Display name for the Virtual Mind (default: "Modifai Virtual Mind").

        Returns:
            A fully populated VirtualMind TypedDict.
        """
        # Deduplicate domain names (preserve insertion order)
        domain_names = list(dict.fromkeys(
            d["name"] for d in knowledge.get("domains", [])
        ))
        key_concepts = list(knowledge.get("key_concepts", []))

        # Build a human-readable description from what we know
        agent_count = len(agents)
        domain_count = len(domain_names)
        description = (
            f"{mind_name} is an automatically generated organizational intelligence layer "
            f"comprising {agent_count} specialized agent{'s' if agent_count != 1 else ''} "
            f"covering {domain_count} knowledge domain{'s' if domain_count != 1 else ''}. "
            f"It encodes the expertise, workflows, and key concepts extracted from the "
            f"organization's documentation, enabling instant, structured access to "
            f"institutional knowledge."
        )

        virtual_mind: VirtualMind = {
            "name": mind_name,
            "description": description,
            "domains": domain_names,
            "key_concepts": key_concepts,
            "agents": agents,
        }

        logger.info(
            "VirtualMindBuilder assembled '%s' with %d agents across %d domains.",
            mind_name,
            agent_count,
            domain_count,
        )
        return virtual_mind
