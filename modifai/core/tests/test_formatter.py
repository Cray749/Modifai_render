"""Unit tests for formatter.py — no AWS calls needed for format tests; S3 upload mocked."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from modifai.core.formatter import (
    format_to_jsonl_string,
    save_to_local_jsonl,
    format_and_upload_to_s3,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

VALID_SAMPLES = [
    {
        "instruction": "What is the refund policy?",
        "input": "Refund policy: customers may request refunds within 30 days.",
        "output": "Customers may request refunds within 30 days of purchase.",
        "chunk_id": 0,
    },
    {
        "instruction": "What is step 1?",
        "input": "Step 1: Open the support portal.",
        "output": "Step 1 is to open the support portal.",
        "chunk_id": 1,
    },
    {
        "instruction": "Summarise the escalation path.",
        "input": "",   # empty input — should be handled
        "output": "Escalate from Tier 1 to Tier 2 then Manager.",
        "chunk_id": 2,
    },
]


# ── format_to_jsonl_string tests ──────────────────────────────────────────────

def test_format_produces_valid_jsonl():
    jsonl = format_to_jsonl_string(VALID_SAMPLES)
    lines = [l for l in jsonl.splitlines() if l.strip()]
    assert len(lines) == 3

    for line in lines:
        record = json.loads(line)
        assert "prompt" in record
        assert "completion" in record
        assert isinstance(record["prompt"], str) and len(record["prompt"]) > 0
        assert isinstance(record["completion"], str) and len(record["completion"]) > 0


def test_format_combines_instruction_and_input():
    """When input is non-empty, prompt should contain both instruction and input."""
    jsonl = format_to_jsonl_string([VALID_SAMPLES[0]])
    record = json.loads(jsonl)
    assert "What is the refund policy?" in record["prompt"]
    assert "customers may request refunds" in record["prompt"]


def test_format_handles_empty_input_field():
    """When input is empty, prompt should be just the instruction."""
    sample = VALID_SAMPLES[2]  # has empty input
    jsonl = format_to_jsonl_string([sample])
    record = json.loads(jsonl)
    assert record["prompt"] == "Summarise the escalation path."
    assert "\n\n" not in record["prompt"]


def test_format_skips_samples_missing_instruction():
    """Samples without instruction should be skipped gracefully."""
    bad_samples = [
        {"instruction": "", "input": "some text", "output": "some output"},
        VALID_SAMPLES[0],  # good sample
    ]
    jsonl = format_to_jsonl_string(bad_samples)
    lines = [l for l in jsonl.splitlines() if l.strip()]
    assert len(lines) == 1  # only the good sample
    record = json.loads(lines[0])
    assert "What is the refund policy?" in record["prompt"]


def test_format_skips_samples_missing_output():
    """Samples without output should be skipped."""
    bad_samples = [
        {"instruction": "What?", "input": "", "output": ""},
        VALID_SAMPLES[1],
    ]
    jsonl = format_to_jsonl_string(bad_samples)
    lines = [l for l in jsonl.splitlines() if l.strip()]
    assert len(lines) == 1


def test_format_completion_is_correct_output():
    """The completion field must be exactly the output field."""
    jsonl = format_to_jsonl_string([VALID_SAMPLES[0]])
    record = json.loads(jsonl)
    assert record["completion"] == VALID_SAMPLES[0]["output"]


# ── save_to_local_jsonl tests ─────────────────────────────────────────────────

def test_save_to_local_creates_file(tmp_path):
    output_path = str(tmp_path / "training_data.jsonl")
    result = save_to_local_jsonl(VALID_SAMPLES, output_path)

    assert result == output_path

    with open(output_path, encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    assert len(lines) == len(VALID_SAMPLES)


def test_save_to_local_content_is_valid_json(tmp_path):
    output_path = str(tmp_path / "training_data.jsonl")
    save_to_local_jsonl(VALID_SAMPLES, output_path)

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                assert "prompt" in record
                assert "completion" in record


# ── format_and_upload_to_s3 tests ─────────────────────────────────────────────

@patch("modifai.core.formatter.boto3.client")
def test_upload_to_s3_returns_correct_uri(mock_boto):
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3
    mock_s3.put_object.return_value = {}

    uri = format_and_upload_to_s3(
        samples=VALID_SAMPLES,
        bucket="my-test-bucket",
        job_id="job123",
    )

    assert uri.startswith("s3://my-test-bucket/modifai-jobs/job123/")
    assert uri.endswith("training_data.jsonl")


@patch("modifai.core.formatter.boto3.client")
def test_upload_to_s3_calls_put_object(mock_boto):
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3

    format_and_upload_to_s3(
        samples=VALID_SAMPLES,
        bucket="my-test-bucket",
        job_id="job456",
    )

    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "my-test-bucket"
    assert "training_data.jsonl" in call_kwargs["Key"]
    assert isinstance(call_kwargs["Body"], bytes)


def test_upload_raises_on_empty_samples():
    with pytest.raises(ValueError, match="empty samples"):
        format_and_upload_to_s3(samples=[], bucket="my-bucket", job_id="job789")


@patch("modifai.core.formatter.boto3.client")
def test_upload_raises_on_s3_error(mock_boto):
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3
    mock_s3.put_object.side_effect = Exception("Access Denied")

    with pytest.raises(RuntimeError, match="S3 upload failed"):
        format_and_upload_to_s3(
            samples=VALID_SAMPLES,
            bucket="forbidden-bucket",
            job_id="job000",
        )
