"""
fine_tuning_trigger.py — Lambda: kick off a fine-tuning job.

Previously this called Amazon Bedrock's create_model_customization_job API.
That dependency has been removed.  Instead the Lambda:

  1. Asks the LLM (via OpenRouter) to validate and confirm the training
     configuration.
  2. Writes a job-manifest JSON to S3 (acts as the job record for downstream
     Lambdas such as status_checker).
  3. Returns the job metadata so the Step Functions pipeline can continue.

To integrate with a real training back-end (Vertex AI, SageMaker, RunPod,
Modal, etc.) replace the _submit_training_job() stub below with the
appropriate SDK call.  Everything else stays the same.

Environment variables
---------------------
AWS_REGION          AWS region (default: ap-south-1)
S3_BUCKET           Bucket used for job manifests and training data
OPENROUTER_API_KEY  OpenRouter API key  (or use Secrets Manager)
OR_SECRET_NAME      Secrets Manager secret name (default: modifai/or)
OR_MODEL            OpenRouter model ID (default: deepseek/deepseek-chat-v3)
JOB_MANIFEST_PREFIX S3 key prefix for job manifest files
                    (default: modifai-jobs)
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import boto3

from llm_helper import call_llm_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-south-1"))

S3_BUCKET           = os.environ.get("S3_BUCKET", "modifai-bucket")
JOB_MANIFEST_PREFIX = os.environ.get("JOB_MANIFEST_PREFIX", "modifai-jobs")

_SYSTEM_PROMPT = (
    "You are an AI Fine-Tuning Orchestrator. "
    "Given a training configuration, validate it and return a concise "
    "confirmation with any corrections or warnings. "
    "Output ONLY valid JSON: "
    '{"validated": true, "warnings": ["<str>", ...], '
    '"final_hyperparameters": {"epochs": <int>, "batch_size": <int>, '
    '"learning_rate": <float>}}'
)


# ── training-backend stub ─────────────────────────────────────────────────────

def _submit_training_job(job_manifest: dict) -> str:
    """
    Submit a training job to the actual training back-end.

    Replace the body of this function with your real SDK call, e.g.:
      - Google Vertex AI  fine-tuning API
      - AWS SageMaker     CreateTrainingJob
      - Modal / RunPod    job submission

    Returns a job_id string that status_checker can poll.
    """
    # Stub: generate a synthetic job ID and persist the manifest to S3.
    job_id = job_manifest["job_name"]
    key    = f"{JOB_MANIFEST_PREFIX}/{job_id}/manifest.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(job_manifest, indent=2),
        ContentType="application/json",
    )
    logger.info("Job manifest written to s3://%s/%s", S3_BUCKET, key)
    return job_id


# ── lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    Expected event shape
    --------------------
    {
      "agent_decision": {                  # from hyperparameter_tuner
        "new_hyperparameters": {...}       # optional override
      },
      "strategy": {                        # from intent_analyzer
        "strategy": {                      # nested OR flat
          "model": "...",
          "hyperparameters": {...}
        }
      },
      "dataset_evaluation": {
        "training_data_uri": "s3://...",
        "run_id": "abc123",
        "bucket": "my-bucket"             # optional override
      }
    }
    """
    if event.get("config", {}).get("openrouter_api_key"):
        os.environ["OPENROUTER_API_KEY"] = event["config"]["openrouter_api_key"]

    agent_decision = event.get("agent_decision", {})

    # Support both nested  {"strategy": {"strategy": {...}}}
    # and flat             {"strategy": {...}}
    raw_strategy = event.get("strategy", {})
    strategy     = raw_strategy.get("strategy", raw_strategy)

    hyperparameters = strategy.get("hyperparameters", {})
    if agent_decision.get("new_hyperparameters"):
        hyperparameters = agent_decision["new_hyperparameters"]
        logger.info("Using tuned hyperparameters from agent_decision: %s", hyperparameters)

    dataset_eval      = event.get("dataset_evaluation", {})
    training_data_uri = dataset_eval.get("training_data_uri", "")
    run_id            = dataset_eval.get("run_id") or str(uuid.uuid4())[:8]
    bucket            = dataset_eval.get("bucket") or S3_BUCKET
    base_model        = strategy.get("model", "meta.llama3-8b-instruct-v1:0")

    job_name = f"modifai-tune-{run_id}-{str(uuid.uuid4())[:4]}"

    # ── LLM: validate config ──────────────────────────────────────────────────
    validation_prompt = (
        f"Base model: {base_model}\n"
        f"Training data URI: {training_data_uri}\n"
        f"Hyperparameters: {json.dumps(hyperparameters)}\n"
        "Validate this fine-tuning configuration."
    )
    try:
        validation = call_llm_json(
            prompt=validation_prompt,
            system=_SYSTEM_PROMPT,
        )
        # Use the LLM's corrections if it returned final_hyperparameters
        if validation.get("final_hyperparameters"):
            hyperparameters = validation["final_hyperparameters"]
        if validation.get("warnings"):
            for w in validation["warnings"]:
                logger.warning("LLM config warning: %s", w)
        logger.info("LLM config validation: validated=%s", validation.get("validated"))
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM validation failed — proceeding with supplied config: %s", exc)

    # ── build and persist job manifest ────────────────────────────────────────
    job_manifest = {
        "job_name":          job_name,
        "run_id":            run_id,
        "base_model":        base_model,
        "training_data_uri": training_data_uri,
        "output_s3_uri":     f"s3://{bucket}/{JOB_MANIFEST_PREFIX}/{run_id}/output/",
        "hyperparameters":   hyperparameters,
        "status":            "InProgress",
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }

    try:
        _submit_training_job(job_manifest)
        logger.info("Fine-tuning job submitted: %s", job_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Job submission failed (pipeline continues): %s", exc)

    return {
        "job_name":          job_name,
        "hyperparameters":   hyperparameters,
        "base_model":        base_model,
        "training_data_uri": training_data_uri,
        "run_id":            run_id,
        "bucket":            bucket,
    }
