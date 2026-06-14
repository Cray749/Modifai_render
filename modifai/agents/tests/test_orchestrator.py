"""Unit tests for OrchestratorAgent. All AWS calls are mocked."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from modifai.agents.orchestrator import OrchestratorAgent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bedrock_response(tool_input: dict) -> dict:
    """Build a fake Bedrock converse response containing a tool call."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "fake-id-001",
                            "name": "set_pipeline_strategy",
                            "input": tool_input,
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
    }


def _make_bad_response() -> dict:
    """Bedrock response with NO tool call (plain text instead)."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Here is the strategy: QA mode, threshold 0.7"}],
            }
        },
        "stopReason": "end_turn",
    }


VALID_DOC_METADATA = {
    "filename": "support_sop.pdf",
    "page_count": 24,
    "domain": "customer support",
    "estimated_chunk_count": 48,
}

VALID_TOOL_INPUT = {
    "intent": "QA",
    "quality_threshold": 0.72,
    "samples_per_chunk": 5,
    "reasoning": "Customer support FAQ — QA intent, standard threshold.",
}


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("modifai.agents.orchestrator.boto3.client")
def test_happy_path_returns_strategy(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.return_value = _make_bedrock_response(VALID_TOOL_INPUT)

    agent = OrchestratorAgent()
    result = agent.run(goal="Build a support Q&A bot", doc_metadata=VALID_DOC_METADATA)

    assert result["intent"] == "QA"
    assert result["quality_threshold"] == 0.72
    assert result["samples_per_chunk"] == 5
    assert isinstance(result["reasoning"], str) and len(result["reasoning"]) > 0


@patch("modifai.agents.orchestrator.boto3.client")
def test_retries_once_on_bad_output_then_succeeds(mock_boto):
    """First call returns bad response (no tool), second returns valid tool call."""
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.side_effect = [
        _make_bad_response(),
        _make_bedrock_response(VALID_TOOL_INPUT),
    ]

    agent = OrchestratorAgent(max_retries=1)
    result = agent.run(goal="Build a support Q&A bot", doc_metadata=VALID_DOC_METADATA)

    assert result["intent"] == "QA"
    assert mock_client.converse.call_count == 2


@patch("modifai.agents.orchestrator.boto3.client")
def test_raises_after_exhausting_retries(mock_boto):
    """Both attempts fail — should raise ValueError."""
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.return_value = _make_bad_response()

    agent = OrchestratorAgent(max_retries=1)

    with pytest.raises(ValueError, match="failed to produce valid strategy"):
        agent.run(goal="Build a support Q&A bot", doc_metadata=VALID_DOC_METADATA)


@pytest.mark.parametrize("bad_intent", ["qa", "Q&A", "chatbot", "", None])
@patch("modifai.agents.orchestrator.boto3.client")
def test_rejects_invalid_intent(mock_boto, bad_intent):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_tool_input = {**VALID_TOOL_INPUT, "intent": bad_intent}
    mock_client.converse.return_value = _make_bedrock_response(bad_tool_input)

    agent = OrchestratorAgent(max_retries=0)

    with pytest.raises(ValueError):
        agent.run(goal="test", doc_metadata=VALID_DOC_METADATA)


@pytest.mark.parametrize("bad_threshold", [0.3, 0.99, 1.5, -0.1, "high"])
@patch("modifai.agents.orchestrator.boto3.client")
def test_rejects_out_of_range_threshold(mock_boto, bad_threshold):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_tool_input = {**VALID_TOOL_INPUT, "quality_threshold": bad_threshold}
    mock_client.converse.return_value = _make_bedrock_response(bad_tool_input)

    agent = OrchestratorAgent(max_retries=0)

    with pytest.raises(ValueError):
        agent.run(goal="test", doc_metadata=VALID_DOC_METADATA)


@pytest.mark.parametrize("bad_spc", [1, 2, 9, 20, "five"])
@patch("modifai.agents.orchestrator.boto3.client")
def test_rejects_out_of_range_samples_per_chunk(mock_boto, bad_spc):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_tool_input = {**VALID_TOOL_INPUT, "samples_per_chunk": bad_spc}
    mock_client.converse.return_value = _make_bedrock_response(bad_tool_input)

    agent = OrchestratorAgent(max_retries=0)

    with pytest.raises(ValueError):
        agent.run(goal="test", doc_metadata=VALID_DOC_METADATA)


@patch("modifai.agents.orchestrator.boto3.client")
def test_instruction_intent_for_sop_domain(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    tool_input = {**VALID_TOOL_INPUT, "intent": "instruction"}
    mock_client.converse.return_value = _make_bedrock_response(tool_input)

    agent = OrchestratorAgent()
    result = agent.run(
        goal="Create instruction-following data from our engineering runbook",
        doc_metadata={
            "filename": "eng_runbook.pdf",
            "page_count": 60,
            "domain": "engineering runbook",
            "estimated_chunk_count": 120,
        },
    )
    assert result["intent"] == "instruction"
