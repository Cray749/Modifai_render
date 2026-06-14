"""
finetuning.py — Starts and monitors an AWS Bedrock model customization (fine-tuning) job.

Supports any model in SUPPORTED_FINETUNE_MODELS (see below), giving the user
full freedom to choose which base model to fine-tune on their dataset.

Prerequisites:
  - S3 bucket in the same region as Bedrock
  - IAM role with permissions: AmazonBedrockFullAccess + S3 read access
  - Minimum 50 training samples (Bedrock requirement for fine-tuning)

Usage:
    from modifai.core.finetuning import start_finetuning_job, wait_for_job, list_supported_models

    # Show user what models they can choose from
    for m in list_supported_models():
        print(m["label"], "—", m["notes"])

    job_name = start_finetuning_job(
        training_data_s3_uri="s3://my-bucket/modifai-jobs/job123/training_data.jsonl",
        output_s3_uri="s3://my-bucket/modifai-jobs/job123/output/",
        custom_model_name="modifai-hr-policy-v1",
        role_arn="arn:aws:iam::123456789012:role/ModifaiBedrockRole",
        base_model_id="amazon.titan-text-express-v1",   # or any key from SUPPORTED_FINETUNE_MODELS
    )
    model_arn = wait_for_job(job_name)
    print("Fine-tuned model ARN:", model_arn)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

_DEFAULT_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")

# ── Supported base models for fine-tuning ─────────────────────────────────────
# Users can pick any of these when calling run_full_pipeline() or
# start_finetuning_job(). The "model_id" value is passed directly to Bedrock.
# Add new entries here as AWS expands Bedrock fine-tuning support.

SUPPORTED_FINETUNE_MODELS: dict[str, dict] = {
    "titan-text-express": {
        "label":     "Amazon Titan Text Express",
        "model_id":  "amazon.titan-text-express-v1",
        "regions":   ["us-east-1", "us-west-2"],
        "notes":     "Best general-purpose choice; fastest fine-tuning; lowest cost.",
        "min_samples": 50,
    },
    "titan-text-lite": {
        "label":     "Amazon Titan Text Lite",
        "model_id":  "amazon.titan-text-lite-v1",
        "regions":   ["us-east-1", "us-west-2"],
        "notes":     "Smallest & cheapest; great for simple instruction-following tasks.",
        "min_samples": 50,
    },
    "nova-micro": {
        "label":     "Amazon Nova Micro",
        "model_id":  "amazon.nova-micro-v1:0",
        "regions":   ["us-east-1", "ap-southeast-2"],
        "notes":     "Ultra-fast, ultra-low-cost; ideal for high-volume QA datasets.",
        "min_samples": 50,
    },
    "nova-lite": {
        "label":     "Amazon Nova Lite",
        "model_id":  "amazon.nova-lite-v1:0",
        "regions":   ["us-east-1", "ap-southeast-2"],
        "notes":     "Balanced speed & quality; supports multimodal inputs.",
        "min_samples": 50,
    },
    "nova-pro": {
        "label":     "Amazon Nova Pro",
        "model_id":  "amazon.nova-pro-v1:0",
        "regions":   ["us-east-1", "ap-southeast-2"],
        "notes":     "Most capable Nova model; best for complex reasoning tasks.",
        "min_samples": 50,
    },
    "llama3-8b": {
        "label":     "Meta Llama 3 8B Instruct",
        "model_id":  "meta.llama3-8b-instruct-v1:0",
        "regions":   ["us-east-1", "us-west-2"],
        "notes":     "Open-weight; excellent instruction-following; community favourite.",
        "min_samples": 100,
    },
    "llama3-70b": {
        "label":     "Meta Llama 3 70B Instruct",
        "model_id":  "meta.llama3-70b-instruct-v1:0",
        "regions":   ["us-east-1", "us-west-2"],
        "notes":     "High-capacity open-weight model; best quality, higher cost.",
        "min_samples": 100,
    },
}

# Default base model (overridable via env var or function argument)
_BASE_MODEL_ID = os.environ.get(
    "BEDROCK_FINETUNE_MODEL_ID",
    SUPPORTED_FINETUNE_MODELS["titan-text-express"]["model_id"],
)


def list_supported_models() -> list[dict]:
    """
    Return a list of all models the user can choose for fine-tuning.

    Each dict contains:
      - key (str):          Short identifier to pass as base_model_id key
      - label (str):        Human-readable name for display in UI / CLI
      - model_id (str):     Bedrock model ID to pass to the API
      - regions (list[str]): AWS regions where this model can be fine-tuned
      - notes (str):        Brief description for user guidance
      - min_samples (int):  Minimum training samples required by Bedrock

    Usage in a CLI or API endpoint::

        for m in list_supported_models():
            print(f"{m['key']:20s}  {m['label']} — {m['notes']}")
    """
    return [
        {"key": key, **info}
        for key, info in SUPPORTED_FINETUNE_MODELS.items()
    ]

# Bedrock fine-tuning job statuses
_TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}


def start_finetuning_job(
    training_data_s3_uri: str,
    output_s3_uri: str,
    custom_model_name: str,
    role_arn: str,
    job_name: Optional[str] = None,
    base_model_id: str = _BASE_MODEL_ID,
    region: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
) -> str:
    """
    Start a Bedrock model customization (fine-tuning) job.

    Args:
        training_data_s3_uri: S3 URI of the training JSONL file.
                              e.g. "s3://my-bucket/modifai-jobs/job123/training_data.jsonl"
        output_s3_uri:        S3 URI prefix for the fine-tuned model output artifacts.
                              e.g. "s3://my-bucket/modifai-jobs/job123/output/"
        custom_model_name:    Name for the fine-tuned model (alphanumeric + hyphens, max 63 chars).
                              e.g. "modifai-hr-policy-v1"
        role_arn:             IAM role ARN with AmazonBedrockFullAccess + S3 read/write.
                              e.g. "arn:aws:iam::123456789012:role/ModifaiBedrockRole"
        job_name:             Optional unique job name. Auto-generated if not provided.
        base_model_id:        Bedrock base model ID to fine-tune.
                              Pass a model_id string (e.g. "amazon.titan-text-express-v1")
                              OR a short key from SUPPORTED_FINETUNE_MODELS
                              (e.g. "nova-lite"). Call list_supported_models() to see all options.
                              Default: titan-text-express-v1.
        region:               AWS region override (default: ap-southeast-2).
        hyperparameters:      Override default training hyperparameters. Defaults:
                              {"epochCount": "2", "batchSize": "8", "learningRate": "0.00005"}

    Returns:
        job_name (str) — use this to poll status with wait_for_job() or get_job_status().

    Raises:
        RuntimeError: If job creation fails.
    """
    region = region or _DEFAULT_REGION

    # Resolve short key → full model_id if user passed a convenience alias
    if base_model_id in SUPPORTED_FINETUNE_MODELS:
        base_model_id = SUPPORTED_FINETUNE_MODELS[base_model_id]["model_id"]

    client = boto3.client("bedrock", region_name=region)

    import uuid
    job_name = job_name or f"modifai-ft-{str(uuid.uuid4())[:8]}"

    default_hyperparams = {
        "epochCount": "2",
        "batchSize": "8",
        "learningRate": "0.00005",
        "warmupSteps": "50",
    }
    if hyperparameters:
        default_hyperparams.update(hyperparameters)

    logger.info(
        "Starting Bedrock fine-tuning job: %s (base=%s)", job_name, base_model_id
    )
    logger.info("  Training data: %s", training_data_s3_uri)
    logger.info("  Output:        %s", output_s3_uri)
    logger.info("  Hyperparams:   %s", default_hyperparams)

    try:
        client.create_model_customization_job(
            jobName=job_name,
            customModelName=custom_model_name,
            roleArn=role_arn,
            baseModelIdentifier=base_model_id,
            trainingDataConfig={"s3Uri": training_data_s3_uri},
            outputDataConfig={"s3Uri": output_s3_uri},
            hyperParameters=default_hyperparams,
        )
        logger.info("Fine-tuning job created: %s", job_name)
        return job_name

    except Exception as exc:
        raise RuntimeError(f"Failed to start fine-tuning job '{job_name}': {exc}") from exc


def wait_for_job(
    job_name: str,
    region: Optional[str] = None,
    poll_interval_seconds: int = 60,
    max_wait_seconds: int = 7200,  # 2 hours max
) -> str:
    """
    Block until a Bedrock fine-tuning job completes and return the custom model ARN.

    Args:
        job_name:              Job name returned by start_finetuning_job().
        region:                AWS region override.
        poll_interval_seconds: Seconds between status checks (default 60).
        max_wait_seconds:      Max wait time (default 7200s = 2 hours).

    Returns:
        custom_model_arn (str) — ARN of the successfully fine-tuned model.

    Raises:
        RuntimeError: If the job fails, is stopped, or times out.
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock", region_name=region)

    logger.info(
        "Waiting for fine-tuning job '%s' to complete (polling every %ds, max %ds)...",
        job_name, poll_interval_seconds, max_wait_seconds,
    )

    elapsed = 0
    while elapsed < max_wait_seconds:
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

        status_info = get_job_status(job_name, region=region)
        status = status_info["status"]
        logger.info(
            "Job '%s' status: %s (elapsed %ds / %ds)",
            job_name, status, elapsed, max_wait_seconds,
        )

        if status == "Completed":
            model_arn = status_info.get("custom_model_arn")
            if not model_arn:
                raise RuntimeError(
                    f"Job '{job_name}' completed but no custom_model_arn in response."
                )
            logger.info("Fine-tuning complete! Model ARN: %s", model_arn)
            return model_arn

        elif status in ("Failed", "Stopped"):
            failure_message = status_info.get("failure_message", "unknown")
            raise RuntimeError(
                f"Fine-tuning job '{job_name}' ended with status '{status}'. "
                f"Reason: {failure_message}"
            )
        # else: InProgress — keep polling

    raise RuntimeError(
        f"Fine-tuning job '{job_name}' did not complete within {max_wait_seconds}s."
    )


def get_job_status(job_name: str, region: Optional[str] = None) -> dict:
    """
    Get the current status of a fine-tuning job.

    Args:
        job_name: Job name from start_finetuning_job().
        region:   AWS region override.

    Returns:
        Dict with:
          - status (str): "InProgress" | "Completed" | "Failed" | "Stopped"
          - custom_model_arn (str | None): set when status == "Completed"
          - failure_message (str | None): set when status == "Failed"
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock", region_name=region)

    response = client.get_model_customization_job(jobIdentifier=job_name)
    return {
        "status": response.get("status", "Unknown"),
        "custom_model_arn": response.get("outputModelArn"),
        "failure_message": response.get("failureMessage"),
        "job_name": response.get("jobName", job_name),
        "base_model_id": response.get("baseModelId"),
        "creation_time": str(response.get("creationTime", "")),
        "end_time": str(response.get("endTime", "")),
    }
