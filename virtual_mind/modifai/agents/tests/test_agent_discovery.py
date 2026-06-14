"""
Unit tests for AgentDiscoveryAgent and VirtualMindBuilder.
All Bedrock calls are mocked — no AWS credentials required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import os
os.environ["LLM_PROVIDER"] = "bedrock"

from modifai.agents.agent_discovery import AgentDiscoveryAgent
from modifai.agents.virtual_mind_builder import VirtualMindBuilder

# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_KNOWLEDGE = {
    "knowledge_summary": "Summary of operations.",
    "domains": [
        {"name": "Human Resources", "description": "Employee policies and management", "evidence": ["Handbook"], "confidence": 0.9},
        {"name": "Engineering", "description": "Software development practices", "evidence": ["Wiki"], "confidence": 0.85},
    ],
    "expertise": [
        {"name": "Employee Onboarding", "confidence": 0.92},
        {"name": "Code Review", "confidence": 0.88},
    ],
    "key_concepts": ["Onboarding", "Performance Review", "CI/CD", "Pull Requests"],
    "workflows": [
        {
            "name": "New Hire Onboarding",
            "steps": ["Send offer", "Complete paperwork", "System access", "Orientation"],
            "confidence": 0.95
        },
        {
            "name": "Code Deployment",
            "steps": ["Write code", "Open PR", "Code review", "Merge", "Deploy"],
            "confidence": 0.90
        },
    ],
}

MOCK_DISCOVERED_AGENTS = [
    {
        "name": "HR Agent",
        "description": "Handles all Human Resources queries and employee management.",
        "specialization": "Human Resources",
        "reasoning": "Strong evidence of HR policies.",
        "confidence": 0.95,
        "source_domains": ["Human Resources"],
        "source_expertise": ["Employee Onboarding"],
        "capabilities": [
            {"name": "Answer HR policy questions", "description": "Responds to queries about HR policies."},
            {"name": "Guide onboarding", "description": "Walks new hires through onboarding steps."},
        ],
    },
    {
        "name": "Engineering Agent",
        "description": "Supports software engineering processes and best practices.",
        "specialization": "Engineering",
        "reasoning": "Strong evidence of engineering workflows.",
        "confidence": 0.92,
        "source_domains": ["Engineering"],
        "source_expertise": ["Code Review"],
        "capabilities": [
            {"name": "Explain CI/CD pipeline", "description": "Describes deployment processes."},
            {"name": "Code review guidance", "description": "Provides PR and review best practices."},
        ],
    },
]


def _make_bedrock_response(agents: list) -> dict:
    """Build a mock Bedrock converse response with the discover_agents tool call."""
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": "discover_agents",
                            "input": {"agents": agents},
                        }
                    }
                ]
            }
        }
    }


# ── AgentDiscoveryAgent tests ─────────────────────────────────────────────────

def test_agent_discovery_success():
    """Agent discovery returns correctly typed DiscoveredAgent list."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(MOCK_DISCOVERED_AGENTS)

        agent = AgentDiscoveryAgent(model_id="test-model", region="us-east-1")
        result = agent.run(knowledge=SAMPLE_KNOWLEDGE)

        assert len(result) == 2
        assert result[0]["name"] == "HR Agent"
        assert result[0]["specialization"] == "Human Resources"
        assert len(result[0]["capabilities"]) == 2
        assert result[1]["name"] == "Engineering Agent"
        assert result[1]["source_domains"] == ["Engineering"]


def test_agent_discovery_validates_empty_agents_list():
    """Empty agents list from the model raises ValueError."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response([])

        agent = AgentDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="non-empty list"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE)


def test_agent_discovery_validates_missing_required_fields():
    """Agent missing required fields raises ValueError."""
    bad_agent = {
        "name": "Broken Agent",
        # missing: description, specialization, source_domains, source_expertise, capabilities
    }
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response([bad_agent])

        agent = AgentDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="missing fields"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE)


def test_agent_discovery_validates_empty_capabilities():
    """Agent with an empty capabilities list raises ValueError."""
    bad_agent = {
        "name": "No-Cap Agent",
        "description": "An agent with no capabilities",
        "specialization": "Finance",
        "reasoning": "Some reasoning.",
        "confidence": 0.9,
        "source_domains": ["Finance"],
        "source_expertise": [],
        "capabilities": [],  # invalid — must have >= 1
    }
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response([bad_agent])

        agent = AgentDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="capability"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE)


def test_agent_discovery_retries_on_wrong_tool_name():
    """If the model calls the wrong tool name, agent retries then raises."""
    bad_response = {
        "output": {
            "message": {
                "content": [
                    {"toolUse": {"name": "wrong_tool", "input": {}}}
                ]
            }
        }
    }
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = bad_response

        agent = AgentDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=1)
        with pytest.raises(ValueError):
            agent.run(knowledge=SAMPLE_KNOWLEDGE)

        # Should have attempted 2 calls (initial + 1 retry)
        assert mock_client.converse.call_count == 2


# ── VirtualMindBuilder tests ──────────────────────────────────────────────────

def test_virtual_mind_builder_assembles_correctly():
    """VirtualMindBuilder produces correct structure from knowledge + agents."""
    builder = VirtualMindBuilder()
    vm = builder.build(knowledge=SAMPLE_KNOWLEDGE, agents=MOCK_DISCOVERED_AGENTS)

    assert vm["name"] == "Modifai Virtual Mind"
    assert "Human Resources" in vm["domains"]
    assert "Engineering" in vm["domains"]
    assert len(vm["agents"]) == 2
    assert vm["agents"][0]["name"] == "HR Agent"
    assert "Onboarding" in vm["key_concepts"]
    assert isinstance(vm["description"], str)
    assert len(vm["description"]) > 0


def test_virtual_mind_builder_custom_name():
    """VirtualMindBuilder respects a custom mind_name."""
    builder = VirtualMindBuilder()
    vm = builder.build(
        knowledge=SAMPLE_KNOWLEDGE,
        agents=MOCK_DISCOVERED_AGENTS,
        mind_name="Acme Corp Virtual Mind",
    )
    assert vm["name"] == "Acme Corp Virtual Mind"
    assert "Acme Corp Virtual Mind" in vm["description"]


def test_virtual_mind_builder_deduplicates_domains():
    """Duplicate domain names in knowledge are deduplicated."""
    duplicate_knowledge = {
        **SAMPLE_KNOWLEDGE,
        "domains": [
            {"name": "HR", "description": "Human Resources"},
            {"name": "HR", "description": "Human Resources again"},
            {"name": "Engineering", "description": "Engineering"},
        ],
    }
    builder = VirtualMindBuilder()
    vm = builder.build(knowledge=duplicate_knowledge, agents=MOCK_DISCOVERED_AGENTS)
    assert vm["domains"].count("HR") == 1
    assert len(vm["domains"]) == 2


def test_virtual_mind_builder_empty_agents():
    """VirtualMindBuilder handles empty agents list gracefully."""
    builder = VirtualMindBuilder()
    vm = builder.build(knowledge=SAMPLE_KNOWLEDGE, agents=[])
    assert vm["agents"] == []
    assert "0 specialized agents" in vm["description"]


def test_virtual_mind_schema_keys():
    """VirtualMind output contains all required schema keys."""
    builder = VirtualMindBuilder()
    vm = builder.build(knowledge=SAMPLE_KNOWLEDGE, agents=MOCK_DISCOVERED_AGENTS)

    required_keys = {"name", "description", "domains", "key_concepts", "agents"}
    assert required_keys.issubset(set(vm.keys()))

    # Each agent must have the full DiscoveredAgent schema
    agent_keys = {"name", "description", "specialization", "reasoning", "confidence",
                  "source_domains", "source_expertise", "capabilities"}
    for agent in vm["agents"]:
        assert agent_keys.issubset(set(agent.keys()))

def test_agent_discovery_validates_confidence():
    """Agent with out-of-range confidence raises ValueError."""
    bad_agent = dict(MOCK_DISCOVERED_AGENTS[0])
    bad_agent["confidence"] = 1.5
    
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response([bad_agent])

        agent = AgentDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="confidence"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE)
