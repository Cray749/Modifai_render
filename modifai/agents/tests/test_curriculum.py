"""Unit tests for CurriculumAgent. All AWS calls are mocked."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from modifai.agents.curriculum import CurriculumAgent

# ── Helpers ────────────────────────────────────────────────────────────────────

VALID_STRATEGY = {
    "intent": "instruction",
    "quality_threshold": 0.72,
    "samples_per_chunk": 5,
    "reasoning": "SOP doc, instruction intent.",
}

VALID_GAP_CATEGORIES = [
    {
        "name": "lacks_step_by_step_reasoning",
        "description": "Answer skips intermediate steps.",
        "example_bad": "Just restart the server.",
        "example_good": "1. SSH into the server. 2. Run `sudo systemctl restart app`. 3. Check logs.",
    },
    {
        "name": "too_vague_on_entities",
        "description": "Doesn't name specific tools or systems from source.",
        "example_bad": "Use the monitoring tool to check.",
        "example_good": "Open Datadog and navigate to the Infra tab to check CPU usage.",
    },
    {
        "name": "factual_drift",
        "description": "Introduces facts not in the source chunk.",
        "example_bad": "The system uses Redis for caching (not mentioned in source).",
        "example_good": "The system uses the cache layer described in section 3.2.",
    },
]

VALID_TOOL_INPUT = {
    "gap_categories": VALID_GAP_CATEGORIES,
    "targeted_prompt": (
        "Each answer MUST enumerate all numbered steps from the source chunk. "
        "Name every specific tool, system, or person referenced. "
        "Never introduce facts not explicitly stated in the source chunk."
    ),
    "priority_focus": "lacks_step_by_step_reasoning",
}

SAMPLE_REJECTION_REASONS = [
    "Answer is too vague, does not name specific systems",
    "Missing intermediate steps in the procedure",
    "Introduces facts not present in the source chunk",
    "Answer is a single sentence when source has 5 steps",
]


def _make_bedrock_response(tool_input: dict) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "fake-id-002",
                            "name": "analyze_curriculum",
                            "input": tool_input,
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
    }


def _make_bad_response() -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "I found 3 gaps: vague, missing steps, hallucination."}],
            }
        },
        "stopReason": "end_turn",
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("modifai.agents.curriculum.boto3.client")
def test_happy_path_returns_curriculum(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.return_value = _make_bedrock_response(VALID_TOOL_INPUT)

    agent = CurriculumAgent()
    result = agent.run(
        rejection_reasons=SAMPLE_REJECTION_REASONS,
        strategy=VALID_STRATEGY,
        iteration=1,
    )

    assert len(result["gap_categories"]) == 3
    assert result["priority_focus"] == "lacks_step_by_step_reasoning"
    assert len(result["targeted_prompt"]) >= 30


@patch("modifai.agents.curriculum.boto3.client")
def test_retries_once_on_bad_output_then_succeeds(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.side_effect = [
        _make_bad_response(),
        _make_bedrock_response(VALID_TOOL_INPUT),
    ]

    agent = CurriculumAgent(max_retries=1)
    result = agent.run(
        rejection_reasons=SAMPLE_REJECTION_REASONS,
        strategy=VALID_STRATEGY,
        iteration=1,
    )
    assert mock_client.converse.call_count == 2


@patch("modifai.agents.curriculum.boto3.client")
def test_raises_on_fewer_than_3_gap_categories(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_input = {
        **VALID_TOOL_INPUT,
        "gap_categories": VALID_GAP_CATEGORIES[:2],  # only 2
    }
    mock_client.converse.return_value = _make_bedrock_response(bad_input)

    agent = CurriculumAgent(max_retries=0)
    with pytest.raises(ValueError, match="≥3 gap categories"):
        agent.run(
            rejection_reasons=SAMPLE_REJECTION_REASONS,
            strategy=VALID_STRATEGY,
            iteration=1,
        )


@patch("modifai.agents.curriculum.boto3.client")
def test_raises_when_priority_focus_not_in_gap_names(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_input = {**VALID_TOOL_INPUT, "priority_focus": "nonexistent_gap"}
    mock_client.converse.return_value = _make_bedrock_response(bad_input)

    agent = CurriculumAgent(max_retries=0)
    with pytest.raises(ValueError, match="priority_focus"):
        agent.run(
            rejection_reasons=SAMPLE_REJECTION_REASONS,
            strategy=VALID_STRATEGY,
            iteration=1,
        )


def test_raises_on_empty_rejection_reasons():
    agent = CurriculumAgent()
    with pytest.raises(ValueError, match="empty rejection_reasons"):
        agent.run(rejection_reasons=[], strategy=VALID_STRATEGY, iteration=1)


def test_extract_rejection_reasons_filters_accepted():
    batch_output = {
        "verdicts": [
            {"verdict": "accept", "reason": "looks good", "rewritten_output": None},
            {"verdict": "reject", "reason": "too vague", "rewritten_output": None},
            {"verdict": "rewrite", "reason": "missing steps", "rewritten_output": "better answer"},
            {"verdict": "accept", "reason": "fine", "rewritten_output": None},
        ],
        "stats": {"total": 4, "accepted": 2, "rewritten": 1, "rejected": 1, "accept_pct": 50.0},
    }
    reasons = CurriculumAgent.extract_rejection_reasons(batch_output)
    assert reasons == ["too vague", "missing steps"]


def test_extract_rejection_reasons_returns_empty_if_all_accepted():
    batch_output = {
        "verdicts": [
            {"verdict": "accept", "reason": "great", "rewritten_output": None},
        ],
        "stats": {"total": 1, "accepted": 1, "rewritten": 0, "rejected": 0, "accept_pct": 100.0},
    }
    reasons = CurriculumAgent.extract_rejection_reasons(batch_output)
    assert reasons == []
