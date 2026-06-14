"""
formatter.py — Converts pipeline output samples into Bedrock fine-tuning format
and uploads the training file to S3.

Bedrock Custom Model fine-tuning expects JSONL where each line is:
    {"prompt": "<full prompt text>", "completion": "<expected completion>"}

The prompt is constructed as:
    "<instruction>\\n\\n<input>" (if input is non-empty)
    "<instruction>"              (if input is empty)

Usage:
    from modifai.core.formatter import format_and_upload_to_s3

    s3_uri = format_and_upload_to_s3(
        samples=state["final_samples"],
        bucket="my-modifai-bucket",
        job_id="job_abc123",
    )
    # s3_uri → "s3://my-modifai-bucket/modifai-jobs/job_abc123/training_data.jsonl"
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from io import StringIO
from typing import List, Optional

import boto3

logger = logging.getLogger(__name__)

_DEFAULT_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")


def format_and_upload_to_s3(
    samples: List[dict],
    bucket: str,
    job_id: Optional[str] = None,
    region: Optional[str] = None,
    s3_prefix: str = "modifai-jobs",
) -> str:
    """
    Format samples into Bedrock fine-tuning JSONL and upload to S3.

    Args:
        samples:   List of sample dicts from run_agentic_loop() final_samples.
                   Each must have: instruction (str), input (str), output (str).
        bucket:    S3 bucket name (must exist and be in the same region as Bedrock).
        job_id:    Unique job identifier. Auto-generated if not provided.
        region:    AWS region override.
        s3_prefix: S3 key prefix (default "modifai-jobs").

    Returns:
        S3 URI of the uploaded training file:
        "s3://{bucket}/{s3_prefix}/{job_id}/training_data.jsonl"

    Raises:
        ValueError: If samples list is empty or samples are missing required fields.
        RuntimeError: If S3 upload fails.
    """
    if not samples:
        raise ValueError("Cannot format empty samples list — nothing to upload.")

    job_id = job_id or str(uuid.uuid4())[:8]
    region = region or _DEFAULT_REGION

    # Convert samples to Bedrock JSONL format
    jsonl_content = _samples_to_bedrock_jsonl(samples)
    line_count = jsonl_content.count("\n") + 1

    # Upload to S3
    s3_key = f"{s3_prefix}/{job_id}/training_data.jsonl"
    s3_uri = f"s3://{bucket}/{s3_key}"

    logger.info(
        "Uploading %d training samples to %s", line_count, s3_uri
    )

    try:
        s3_client = boto3.client("s3", region_name=region)
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=jsonl_content.encode("utf-8"),
            ContentType="application/jsonl",
        )
        logger.info("Upload complete: %s", s3_uri)
        return s3_uri

    except Exception as exc:
        raise RuntimeError(f"S3 upload failed for {s3_uri}: {exc}") from exc


def format_to_jsonl_string(samples: List[dict]) -> str:
    """
    Convert samples to Bedrock fine-tuning JSONL string without uploading.
    Useful for local inspection or saving to disk.

    Args:
        samples: List of sample dicts with instruction, input, output fields.

    Returns:
        JSONL string (one JSON object per line).
    """
    return _samples_to_bedrock_jsonl(samples)


def save_to_local_jsonl(samples: List[dict], output_path: str) -> str:
    """
    Save formatted samples to a local JSONL file.
    Useful for inspecting dataset before uploading.

    Args:
        samples:     List of sample dicts.
        output_path: Local file path to write to.

    Returns:
        output_path (echoed for convenience).
    """
    jsonl_content = _samples_to_bedrock_jsonl(samples)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(jsonl_content)
    logger.info("Saved %d samples to %s", len(samples), output_path)
    return output_path


# ── Private helpers ────────────────────────────────────────────────────────────

def _samples_to_bedrock_jsonl(samples: List[dict]) -> str:
    """
    Convert samples to Bedrock custom model fine-tuning JSONL format.

    Each line: {"prompt": "...", "completion": "..."}

    The prompt combines instruction + input (if present).
    The completion is the expected output.
    """
    lines = []
    skipped = 0

    for i, sample in enumerate(samples):
        instruction = str(sample.get("instruction", "")).strip()
        input_text = str(sample.get("input", "")).strip()
        output_text = str(sample.get("output", "")).strip()

        if not instruction or not output_text:
            logger.warning(
                "Sample %d missing instruction or output — skipping.", i
            )
            skipped += 1
            continue

        # Build prompt
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        record = {
            "prompt": prompt,
            "completion": output_text,
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    if skipped > 0:
        logger.warning("Skipped %d malformed samples during formatting.", skipped)

    return "\n".join(lines)
