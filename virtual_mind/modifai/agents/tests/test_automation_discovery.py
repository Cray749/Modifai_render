"""
Unit tests for AutomationDiscoveryAgent.
All Bedrock calls are mocked — no AWS credentials required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import os
os.environ["LLM_PROVIDER"] = "bedrock"

from modifai.agents.automation_discovery import AutomationDiscoveryAgent

# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_KNOWLEDGE = {
    "knowledge_summary": "The organization primarily covers HR onboarding and engineering processes.",
    "domains": [
        {
            "name": "Human Resources",
            "description": "Employee lifecycle management",
            "evidence": ["Employee Handbook", "Leave Policy", "Payroll Procedures"],
            "confidence": 0.93,
        },
        {
            "name": "Engineering",
            "description": "Software development practices",
            "evidence": ["CI/CD Pipeline Guide", "Code Review Standards"],
            "confidence": 0.87,
        },
    ],
    "expertise": [
        {"name": "Employee Onboarding", "confidence": 0.92},
        {"name": "Code Review", "confidence": 0.88},
    ],
    "key_concepts": ["Onboarding", "Code Review", "CI/CD", "Leave Requests"],
    "workflows": [
        {
            "name": "New Hire Onboarding",
            "steps": ["Send offer letter", "Complete paperwork", "Create accounts", "Assign equipment", "Schedule orientation"],
            "confidence": 0.95,
        },
        {
            "name": "Code Deployment",
            "steps": ["Write code", "Open PR", "Review", "Merge", "Deploy"],
            "confidence": 0.89,
        },
    ],
}

SAMPLE_VIRTUAL_MIND = {
    "name": "Modifai Virtual Mind",
    "description": "Organizational intelligence layer.",
    "domains": ["Human Resources", "Engineering"],
    "key_concepts": ["Onboarding", "Code Review"],
    "agents": [
        {
            "name": "HR Agent",
            "description": "Handles HR queries.",
            "specialization": "Human Resources",
            "reasoning": "Strong HR evidence.",
            "confidence": 0.93,
            "source_domains": ["Human Resources"],
            "source_expertise": ["Employee Onboarding"],
            "capabilities": [{"name": "Answer HR questions", "description": "Responds to HR queries."}],
        },
        {
            "name": "Engineering Agent",
            "description": "Handles engineering queries.",
            "specialization": "Engineering",
            "reasoning": "Strong engineering evidence.",
            "confidence": 0.88,
            "source_domains": ["Engineering"],
            "source_expertise": ["Code Review"],
            "capabilities": [{"name": "Explain CI/CD", "description": "Describes deployment pipeline."}],
        },
    ],
}

MOCK_AUTOMATION_OUTPUT = {
    "executive_summary": (
        "2 automation opportunities identified across HR and Engineering with high business impact."
    ),
    "automation_opportunities": [
        {
            "name": "Employee Onboarding Automation",
            "description": "Automates the end-to-end new hire onboarding process.",
            "owning_agent": "HR Agent",
            "automation_score": 92,
            "confidence": 0.91,
            "business_impact": "High",
            "impact_reasoning": "Highly repetitive, 5-step process occurring with every new hire.",
            "estimated_benefits": [
                "4 hours saved per employee (estimated)",
                "Reduced manual paperwork",
                "Faster system account provisioning",
            ],
            "evidence": ["Employee Handbook", "Leave Policy"],
            "source_workflow": "New Hire Onboarding",
            "source_domains": ["Human Resources"],
            "reasoning": "5-step standardised workflow with strong evidence and high confidence.",
            "automation_blueprint": {
                "trigger": "New employee record created in HRIS",
                "actions": [
                    "Send welcome email with required documents",
                    "Create email and system accounts",
                    "Assign equipment from inventory",
                    "Assign manager in org chart",
                    "Schedule orientation session",
                ],
            },
        },
        {
            "name": "Code Deployment Pipeline Automation",
            "description": "Automates the code review and deployment process.",
            "owning_agent": "Engineering Agent",
            "automation_score": 85,
            "confidence": 0.87,
            "business_impact": "High",
            "impact_reasoning": "Standardised 5-step process with clear triggers and automated checks.",
            "estimated_benefits": [
                "30% faster deployment cycles (estimated)",
                "Reduced manual intervention in deployments",
            ],
            "evidence": ["CI/CD Pipeline Guide", "Code Review Standards"],
            "source_workflow": "Code Deployment",
            "source_domains": ["Engineering"],
            "reasoning": "Well-documented workflow with standardised steps suitable for automation.",
            "automation_blueprint": {
                "trigger": "Pull request merged to main branch",
                "actions": [
                    "Run automated test suite",
                    "Build deployment artifact",
                    "Deploy to staging environment",
                    "Run smoke tests",
                    "Promote to production",
                ],
            },
        },
    ],
    "total_opportunities": 2,
    "high_impact_count": 2,
}


def _make_bedrock_response(payload: dict) -> dict:
    """Build a mock Bedrock converse response with the discover_automations tool call."""
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": "discover_automations",
                            "input": payload,
                        }
                    }
                ]
            }
        }
    }


# ── Success path ──────────────────────────────────────────────────────────────

def test_automation_discovery_success():
    """Agent returns correctly typed AutomationDiscoveryOutput."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(MOCK_AUTOMATION_OUTPUT)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1")
        result = agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)

        assert result["total_opportunities"] == 2
        assert result["high_impact_count"] == 2
        assert len(result["automation_opportunities"]) == 2
        assert result["automation_opportunities"][0]["name"] == "Employee Onboarding Automation"
        assert result["automation_opportunities"][0]["owning_agent"] == "HR Agent"
        assert result["automation_opportunities"][0]["automation_score"] == 92
        assert result["automation_opportunities"][0]["business_impact"] == "High"
        assert isinstance(result["executive_summary"], str)


def test_automation_blueprint_structure():
    """AutomationBlueprint has trigger and non-empty actions list."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(MOCK_AUTOMATION_OUTPUT)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1")
        result = agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)

        bp = result["automation_opportunities"][0]["automation_blueprint"]
        assert isinstance(bp["trigger"], str) and len(bp["trigger"]) > 0
        assert isinstance(bp["actions"], list) and len(bp["actions"]) >= 2


def test_estimated_benefits_present():
    """Each opportunity has non-empty estimated_benefits list."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(MOCK_AUTOMATION_OUTPUT)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1")
        result = agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)

        for opp in result["automation_opportunities"]:
            assert isinstance(opp["estimated_benefits"], list)
            assert len(opp["estimated_benefits"]) >= 1


def test_owning_agent_assignment():
    """owning_agent values are valid Virtual Mind agent names."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(MOCK_AUTOMATION_OUTPUT)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1")
        result = agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)

        valid_agents = {a["name"] for a in SAMPLE_VIRTUAL_MIND["agents"]}
        for opp in result["automation_opportunities"]:
            assert opp["owning_agent"] in valid_agents


def test_executive_summary_non_empty():
    """executive_summary is a non-empty string."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(MOCK_AUTOMATION_OUTPUT)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1")
        result = agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)

        assert isinstance(result["executive_summary"], str)
        assert len(result["executive_summary"].strip()) > 0


def test_high_impact_count_computed_correctly():
    """total_opportunities and high_impact_count are computed from actual data."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        # Return payload with 1 High and 1 Medium opportunity
        payload = dict(MOCK_AUTOMATION_OUTPUT)
        payload["automation_opportunities"] = list(MOCK_AUTOMATION_OUTPUT["automation_opportunities"])
        payload["automation_opportunities"][1] = dict(payload["automation_opportunities"][1])
        payload["automation_opportunities"][1]["business_impact"] = "Medium"
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1")
        result = agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)

        assert result["total_opportunities"] == 2
        assert result["high_impact_count"] == 1  # only the first is High


# ── Validation tests ──────────────────────────────────────────────────────────

def test_validates_empty_opportunities():
    """Empty automation_opportunities list raises ValueError."""
    payload = dict(MOCK_AUTOMATION_OUTPUT)
    payload["automation_opportunities"] = []

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="non-empty list"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_score_out_of_range():
    """automation_score > 100 raises ValueError."""
    bad = dict(MOCK_AUTOMATION_OUTPUT["automation_opportunities"][0])
    bad["automation_score"] = 150
    payload = {"executive_summary": "summary", "automation_opportunities": [bad]}

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="automation_score"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_confidence_out_of_range():
    """confidence > 1.0 raises ValueError."""
    bad = dict(MOCK_AUTOMATION_OUTPUT["automation_opportunities"][0])
    bad["confidence"] = 1.5
    payload = {"executive_summary": "summary", "automation_opportunities": [bad]}

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="confidence"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_invalid_impact_level():
    """business_impact not in {High, Medium, Low} raises ValueError."""
    bad = dict(MOCK_AUTOMATION_OUTPUT["automation_opportunities"][0])
    bad["business_impact"] = "Critical"
    payload = {"executive_summary": "summary", "automation_opportunities": [bad]}

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="business_impact"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_missing_required_fields():
    """Opportunity missing required fields raises ValueError."""
    bad = {"name": "Incomplete Opportunity"}
    payload = {"executive_summary": "summary", "automation_opportunities": [bad]}

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="missing fields"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_empty_evidence():
    """Empty evidence list raises ValueError."""
    bad = dict(MOCK_AUTOMATION_OUTPUT["automation_opportunities"][0])
    bad["evidence"] = []
    payload = {"executive_summary": "summary", "automation_opportunities": [bad]}

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="evidence"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_empty_estimated_benefits():
    """Empty estimated_benefits list raises ValueError."""
    bad = dict(MOCK_AUTOMATION_OUTPUT["automation_opportunities"][0])
    bad["estimated_benefits"] = []
    payload = {"executive_summary": "summary", "automation_opportunities": [bad]}

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="estimated_benefits"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_empty_blueprint_actions():
    """Blueprint with < 2 actions raises ValueError."""
    bad = dict(MOCK_AUTOMATION_OUTPUT["automation_opportunities"][0])
    bad["automation_blueprint"] = {"trigger": "Some event", "actions": ["Only one action"]}
    payload = {"executive_summary": "summary", "automation_opportunities": [bad]}

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="actions"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_owning_agent_not_in_virtual_mind():
    """owning_agent not matching any Virtual Mind agent raises ValueError."""
    bad = dict(MOCK_AUTOMATION_OUTPUT["automation_opportunities"][0])
    bad["owning_agent"] = "Nonexistent Agent"
    payload = {"executive_summary": "summary", "automation_opportunities": [bad]}

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="owning_agent"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_validates_empty_executive_summary():
    """Empty executive_summary raises ValueError."""
    payload = dict(MOCK_AUTOMATION_OUTPUT)
    payload["executive_summary"] = "   "

    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = _make_bedrock_response(payload)

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError, match="executive_summary"):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)


def test_retries_on_wrong_tool_name():
    """Wrong tool name causes retry, then raises ValueError."""
    bad_response = {
        "output": {
            "message": {
                "content": [{"toolUse": {"name": "wrong_tool", "input": {}}}]
            }
        }
    }
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.converse.return_value = bad_response

        agent = AutomationDiscoveryAgent(model_id="test-model", region="us-east-1", max_retries=1)
        with pytest.raises(ValueError):
            agent.run(knowledge=SAMPLE_KNOWLEDGE, virtual_mind=SAMPLE_VIRTUAL_MIND)

        # initial + 1 retry = 2 calls
        assert mock_client.converse.call_count == 2
