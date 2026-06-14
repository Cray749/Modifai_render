"""Unit tests for finetuning.py — all AWS/Bedrock calls mocked."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from modifai.core.finetuning import (
    start_finetuning_job,
    wait_for_job,
    get_job_status,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

TRAINING_URI = "s3://my-bucket/modifai-jobs/job123/training_data.jsonl"
OUTPUT_URI = "s3://my-bucket/modifai-jobs/job123/output/"
MODEL_NAME = "modifai-test-v1"
ROLE_ARN = "arn:aws:iam::123456789012:role/ModifaiBedrockRole"
JOB_NAME = "modifai-ft-test001"
MODEL_ARN = "arn:aws:bedrock:us-east-1:123456789012:custom-model/modifai-test-v1/abc123"


# ── start_finetuning_job tests ─────────────────────────────────────────────────

@patch("modifai.core.finetuning.boto3.client")
def test_start_job_calls_create_model_customization(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.create_model_customization_job.return_value = {}

    job_name = start_finetuning_job(
        training_data_s3_uri=TRAINING_URI,
        output_s3_uri=OUTPUT_URI,
        custom_model_name=MODEL_NAME,
        role_arn=ROLE_ARN,
        job_name=JOB_NAME,
    )

    assert job_name == JOB_NAME
    mock_client.create_model_customization_job.assert_called_once()
    call_kwargs = mock_client.create_model_customization_job.call_args[1]
    assert call_kwargs["jobName"] == JOB_NAME
    assert call_kwargs["customModelName"] == MODEL_NAME
    assert call_kwargs["roleArn"] == ROLE_ARN
    assert call_kwargs["trainingDataConfig"]["s3Uri"] == TRAINING_URI
    assert call_kwargs["outputDataConfig"]["s3Uri"] == OUTPUT_URI


@patch("modifai.core.finetuning.boto3.client")
def test_start_job_uses_titan_as_default_base_model(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.create_model_customization_job.return_value = {}

    start_finetuning_job(
        training_data_s3_uri=TRAINING_URI,
        output_s3_uri=OUTPUT_URI,
        custom_model_name=MODEL_NAME,
        role_arn=ROLE_ARN,
        job_name=JOB_NAME,
    )

    call_kwargs = mock_client.create_model_customization_job.call_args[1]
    assert "titan" in call_kwargs["baseModelIdentifier"].lower()


@patch("modifai.core.finetuning.boto3.client")
def test_start_job_auto_generates_name_if_not_provided(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.create_model_customization_job.return_value = {}

    job_name = start_finetuning_job(
        training_data_s3_uri=TRAINING_URI,
        output_s3_uri=OUTPUT_URI,
        custom_model_name=MODEL_NAME,
        role_arn=ROLE_ARN,
        # no job_name provided
    )

    assert job_name.startswith("modifai-ft-")
    assert len(job_name) > len("modifai-ft-")


@patch("modifai.core.finetuning.boto3.client")
def test_start_job_raises_on_api_error(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.create_model_customization_job.side_effect = Exception("AccessDenied")

    with pytest.raises(RuntimeError, match="Failed to start fine-tuning job"):
        start_finetuning_job(
            training_data_s3_uri=TRAINING_URI,
            output_s3_uri=OUTPUT_URI,
            custom_model_name=MODEL_NAME,
            role_arn=ROLE_ARN,
            job_name=JOB_NAME,
        )


# ── wait_for_job tests ─────────────────────────────────────────────────────────

@patch("modifai.core.finetuning.time.sleep")
@patch("modifai.core.finetuning.boto3.client")
def test_wait_for_job_returns_model_arn_on_success(mock_boto, mock_sleep):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.get_model_customization_job.side_effect = [
        {"status": "InProgress"},
        {"status": "Completed", "outputModelArn": MODEL_ARN},
    ]

    result = wait_for_job(JOB_NAME, poll_interval_seconds=1)
    assert result == MODEL_ARN


@patch("modifai.core.finetuning.time.sleep")
@patch("modifai.core.finetuning.boto3.client")
def test_wait_for_job_raises_on_failed_status(mock_boto, mock_sleep):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.get_model_customization_job.return_value = {
        "status": "Failed",
        "failureMessage": "Insufficient training data",
    }

    with pytest.raises(RuntimeError, match="Failed"):
        wait_for_job(JOB_NAME, poll_interval_seconds=1)


@patch("modifai.core.finetuning.time.sleep")
@patch("modifai.core.finetuning.boto3.client")
def test_wait_for_job_raises_on_timeout(mock_boto, mock_sleep):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    # Always returns InProgress — should time out
    mock_client.get_model_customization_job.return_value = {"status": "InProgress"}

    with pytest.raises(RuntimeError, match="did not complete within"):
        wait_for_job(JOB_NAME, poll_interval_seconds=1, max_wait_seconds=3)


# ── get_job_status tests ───────────────────────────────────────────────────────

@patch("modifai.core.finetuning.boto3.client")
def test_get_job_status_returns_correct_fields(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.get_model_customization_job.return_value = {
        "status": "Completed",
        "outputModelArn": MODEL_ARN,
        "jobName": JOB_NAME,
        "baseModelId": "amazon.titan-text-express-v1",
        "failureMessage": None,
    }

    status = get_job_status(JOB_NAME)
    assert status["status"] == "Completed"
    assert status["custom_model_arn"] == MODEL_ARN
    assert status["job_name"] == JOB_NAME
