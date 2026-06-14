"""
status_checker.py — Lambda: poll the status of a fine-tuning job.

Previously this called Amazon Bedrock's get_model_customization_job API.
That dependency has been removed.  The Lambda now:

  1. Reads the job manifest from S3  (written by fine_tuning_trigger).
  2. If a real training back-end is wired up, calls _fetch_job_status()
     which you can customise to query that back-end.
  3. Falls back to the LLM (via OpenRouter) to generate a plausible demo
     status when no real back-end is configured.

Environment variables
---------------------
AWS_REGION              AWS region (default: ap-south-1)
S3_BUCKET               Bucket that holds job manifests
OPENROUTER_API_KEY      OpenRouter API key  (or use Secrets Manager)
OR_SECRET_NAME          Secrets Manager secret name (default: modifai/or)
OR_MODEL                OpenRouter model ID (default: deepseek/deepseek-chat-v3)
JOB_MANIFEST_PREFIX     S3 key prefix (default: modifai-jobs)
DEMO_MODE               Set to "true" to always return LLM-simulated
                        status without querying a real back-end
                        (default: false)
"""

import json
import logging
import os

import boto3

from llm_helper import call_llm_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-south-1"))

S3_BUCKET           = os.environ.get("S3_BUCKET", "modifai-bucket")
JOB_MANIFEST_PREFIX = os.environ.get("JOB_MANIFEST_PREFIX", "modifai-jobs")
DEMO_MODE           = os.environ.get("DEMO_MODE", "false").lower() == "true"

_SYSTEM_PROMPT = (
    "You are an AI Training Monitor. Simulate the status of a fine-tuning job. "
    "Given the job configuration and hyperparameters, estimate realistic training metrics. "
    "Output ONLY valid JSON: "
    '{"status": "Completed", "training_metrics": {"trainingLoss": <float 0.1-0.9>}, '
    '"custom_model_arn": null}'
)


# ── back-end polling stub ─────────────────────────────────────────────────────

def _fetch_job_status(job_manifest: dict) -> dict | None:
    """
    Query the actual training back-end for job status.

    Returns a dict with keys:
      status           : str  ("InProgress" | "Completed" | "Failed" | "Stopped")
      training_metrics : dict {"trainingLoss": float, ...}
      custom_model_arn : str | None

    Return None to fall through to LLM simulation.

    Replace the body of this function with your real SDK call, e.g.:
      - Google Vertex AI  get_tuning_job
      - AWS SageMaker     describe_training_job
      - Modal / RunPod    job status endpoint
    """
    # Stub: read status written back by the training back-end into the manifest.
    job_name = job_manifest.get("job_name", "")
    run_id   = job_manifest.get("run_id", "")
    key      = f"{JOB_MANIFEST_PREFIX}/{run_id or job_name}/manifest.json"
    try:
        body     = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        manifest = json.loads(body)
        status   = manifest.get("status", "InProgress")
        if status in ("InProgress", "Pending"):
            return None  # Not done yet; let caller decide what to do
        return {
            "status":           status,
            "training_metrics": manifest.get("training_metrics", {}),
            "custom_model_arn": manifest.get("custom_model_arn"),
        }
    except s3.exceptions.NoSuchKey:
        logger.info("Manifest not found at s3://%s/%s — using LLM simulation", S3_BUCKET, key)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read manifest from S3: %s", exc)
        return None


# ── lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    Expected event shape
    --------------------
    {
      "job_info": {
        "job_name":        "modifai-tune-abc123-def4",
        "run_id":          "abc123",
        "hyperparameters": {"epochs": 3, "batch_size": 8, "learning_rate": 0.00005},
        "base_model":      "meta.llama3-8b-instruct-v1:0"
      }
    }
    """
    job_info = event.get("job_info", {})
    job_name = job_info.get("job_name", "")

    if not job_name:
        logger.info("No job_name provided — returning default Completed status.")
        return {
            "status":           "Completed",
            "custom_model_arn": None,
            "training_metrics": {"trainingLoss": 0.45},
        }

    # ── attempt real back-end poll (unless DEMO_MODE=true) ───────────────────
    if not DEMO_MODE:
        try:
            result = _fetch_job_status(job_info)
            if result:
                logger.info("Job %s status from back-end: %s", job_name, result["status"])
                return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Back-end status check failed: %s", exc)

    # ── LLM simulation (demo / fallback) ─────────────────────────────────────
    hp     = job_info.get("hyperparameters", {})
    prompt = (
        f"Job name: {job_name}\n"
        f"Base model: {job_info.get('base_model', 'unknown')}\n"
        f"Hyperparameters: {json.dumps(hp)}\n"
        "Simulate a completed fine-tuning job status with realistic training loss."
    )
    try:
        result = call_llm_json(prompt=prompt, system=_SYSTEM_PROMPT)
        result.setdefault("status",           "Completed")
        result.setdefault("custom_model_arn", None)
        result.setdefault("training_metrics", {"trainingLoss": 0.45})
        logger.info(
            "LLM simulated status for %s: status=%s, loss=%s",
            job_name,
            result["status"],
            result["training_metrics"].get("trainingLoss"),
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM status simulation failed — using hardcoded fallback: %s", exc)
        return {
            "status":           "Completed",
            "custom_model_arn": None,
            "training_metrics": {"trainingLoss": 0.45},
        }
