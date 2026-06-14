"""
deployment.py — Provisions a Bedrock custom model for inference (Provisioned Throughput).

After a fine-tuning job completes, the model artifact exists in S3 but can't be
called directly. You must create a "Provisioned Throughput" unit, which makes
the model callable via the standard Bedrock converse API.

Provisioned Throughput pricing: ~$0.004/minute for 1 Model Unit (billed per minute).
For a hackathon demo, provision → demo → delete to minimise cost.

Usage:
    from modifai.core.deployment import provision_model, delete_provisioned_throughput

    # Provision (takes ~5–15 minutes)
    endpoint_arn = provision_model(
        custom_model_arn="arn:aws:bedrock:us-east-1::foundation-model/...",
        provisioned_model_name="modifai-hr-policy-endpoint",
    )

    # Call it (see inference.py)
    # ...

    # Clean up when done (stop billing)
    delete_provisioned_throughput(endpoint_arn)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

_DEFAULT_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
_TERMINAL_STATUSES = {"InService", "Failed"}


def provision_model(
    custom_model_arn: str,
    provisioned_model_name: str,
    model_units: int = 1,
    region: Optional[str] = None,
    poll_interval_seconds: int = 30,
    max_wait_seconds: int = 1200,  # 20 minutes
) -> str:
    """
    Create provisioned throughput for a Bedrock custom model and wait until InService.

    Args:
        custom_model_arn:        ARN of the fine-tuned model from wait_for_job().
        provisioned_model_name:  Name for the provisioned endpoint (alphanumeric + hyphens).
        model_units:             Number of Model Units to provision (default 1, minimum for demos).
        region:                  AWS region override.
        poll_interval_seconds:   Seconds between status polls (default 30).
        max_wait_seconds:        Max wait before timeout (default 1200s = 20 minutes).

    Returns:
        provisioned_model_arn (str) — use this as the modelId in Bedrock converse calls.

    Raises:
        RuntimeError: If provisioning fails or times out.
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock", region_name=region)

    logger.info(
        "Creating provisioned throughput: name=%s, model=%s, units=%d",
        provisioned_model_name, custom_model_arn, model_units,
    )

    try:
        response = client.create_provisioned_model_throughput(
            provisionedModelName=provisioned_model_name,
            modelId=custom_model_arn,
            modelUnits=model_units,
        )
        provisioned_model_arn = response["provisionedModelArn"]
        logger.info("Provisioned throughput created: %s", provisioned_model_arn)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create provisioned throughput '{provisioned_model_name}': {exc}"
        ) from exc

    # Poll until InService
    elapsed = 0
    while elapsed < max_wait_seconds:
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

        try:
            status_response = client.get_provisioned_model_throughput(
                provisionedModelId=provisioned_model_arn
            )
            status = status_response.get("status", "Unknown")
            logger.info(
                "Provisioned model '%s' status: %s (elapsed %ds)",
                provisioned_model_name, status, elapsed,
            )

            if status == "InService":
                logger.info(
                    "Model endpoint ready! ARN: %s", provisioned_model_arn
                )
                return provisioned_model_arn

            elif status == "Failed":
                failure_msg = status_response.get("failureMessage", "unknown")
                raise RuntimeError(
                    f"Provisioned throughput '{provisioned_model_name}' failed: {failure_msg}"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Status poll error (will retry): %s", exc)

    raise RuntimeError(
        f"Provisioned throughput '{provisioned_model_name}' did not reach InService "
        f"within {max_wait_seconds}s."
    )


def delete_provisioned_throughput(
    provisioned_model_arn: str,
    region: Optional[str] = None,
) -> None:
    """
    Delete a provisioned throughput unit to stop billing.

    IMPORTANT: Call this after your demo/testing is complete.
    Provisioned throughput is billed by the minute.

    Args:
        provisioned_model_arn: ARN returned by provision_model().
        region:                AWS region override.
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock", region_name=region)

    logger.info("Deleting provisioned throughput: %s", provisioned_model_arn)
    try:
        client.delete_provisioned_model_throughput(
            provisionedModelId=provisioned_model_arn
        )
        logger.info("Provisioned throughput deleted successfully.")
    except Exception as exc:
        logger.error(
            "Failed to delete provisioned throughput %s: %s",
            provisioned_model_arn, exc,
        )
        raise


def list_provisioned_models(region: Optional[str] = None) -> list:
    """
    List all provisioned throughput models in the account.
    Useful for checking what's currently running (and costing money).

    Returns:
        List of dicts with: name, arn, status, model_units, creation_time
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock", region_name=region)

    response = client.list_provisioned_model_throughputs()
    models = []
    for item in response.get("provisionedModelSummaries", []):
        models.append({
            "name": item.get("provisionedModelName"),
            "arn": item.get("provisionedModelArn"),
            "status": item.get("status"),
            "model_units": item.get("modelUnits"),
            "creation_time": str(item.get("creationTime", "")),
        })
    return models
