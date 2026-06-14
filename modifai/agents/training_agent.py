"""
TrainingAgent — decides SageMaker fine-tuning hyperparameters via LLM,
submits the training job, and monitors it to completion.

WHERE IT SITS IN THE PIPELINE:
    DatasetGeneration → Critic → [TrainingAgent] → Deployment

The agent has two responsibilities:
  1. DECIDE — call Bedrock to pick optimal SageMaker hyperparameters
     (learning rate, epochs, batch size, warmup ratio) based on the dataset
     characteristics produced by the agentic loop.
  2. TRAIN  — submit the SageMaker training job with those hyperparameters,
     upload the dataset to S3, and poll until completion.

It follows the same class pattern as OrchestratorAgent and CriticAgent so it
drops straight into pipeline_loop.py (or full_pipeline.py).

Usage:
    from modifai.agents.training_agent import TrainingAgent

    agent = TrainingAgent(
        role_arn="arn:aws:iam::527371380408:role/ModifaiSageMakerRole",
        s3_bucket="modifai-newbucket",
        base_model_id="meta-llama/Llama-3.1-8B",   # HuggingFace model for SageMaker
    )
    result = agent.run(
        samples=final_samples,
        dataset_stats=final_stats,
        job_id="abc123",
        custom_model_name="modifai-hr-policy-v1",
    )
    print("SageMaker job:", result["job_name"])
    print("Model S3 path:", result["model_s3_uri"])
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
_DEFAULT_BUCKET = os.environ.get("S3_BUCKET_NAME", "modifai-newbucket")

# SageMaker HuggingFace-compatible base models users can choose for fine-tuning.
# These are model IDs that SageMaker's HuggingFace DLC (Deep Learning Container) can pull.
SUPPORTED_SAGEMAKER_MODELS: Dict[str, Dict] = {
    "llama3-8b": {
        "label":      "Meta Llama 3 8B Instruct",
        "model_id":   "meta-llama/Meta-Llama-3-8B-Instruct",
        "instance":   "ml.g5.2xlarge",
        "notes":      "Best quality/cost balance. Recommended for most use-cases.",
    },
    "llama3-70b": {
        "label":      "Meta Llama 3 70B Instruct",
        "model_id":   "meta-llama/Meta-Llama-3-70B-Instruct",
        "instance":   "ml.g5.48xlarge",
        "notes":      "Highest quality. Requires large GPU instance — higher cost.",
    },
    "mistral-7b": {
        "label":      "Mistral 7B Instruct v0.2",
        "model_id":   "mistralai/Mistral-7B-Instruct-v0.2",
        "instance":   "ml.g5.2xlarge",
        "notes":      "Fast, efficient open-weight model. Good for instruction-following.",
    },
    "phi3-mini": {
        "label":      "Microsoft Phi-3 Mini 4K Instruct",
        "model_id":   "microsoft/Phi-3-mini-4k-instruct",
        "instance":   "ml.g5.xlarge",
        "notes":      "Ultra-small & fast. Great for low-latency QA on constrained budgets.",
    },
    "gemma2-9b": {
        "label":      "Google Gemma 2 9B Instruct",
        "model_id":   "google/gemma-2-9b-it",
        "instance":   "ml.g5.4xlarge",
        "notes":      "Strong multilingual support. Good for diverse document languages.",
    },
}

# ── Bedrock tool spec for hyperparameter decision ─────────────────────────────

_HYPERPARAM_TOOL_SPEC = {
    "toolSpec": {
        "name": "set_training_hyperparameters",
        "description": (
            "Set the SageMaker fine-tuning hyperparameters based on dataset "
            "characteristics. Called by the TrainingAgent to choose optimal "
            "training configuration before submitting the job."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "learning_rate": {
                        "type": "number",
                        "description": (
                            "Peak learning rate for the AdamW optimizer. "
                            "Typical range: 1e-5 to 5e-4. "
                            "Use lower values (1e-5 to 5e-5) for larger models or small datasets. "
                            "Use higher values (1e-4 to 3e-4) for small models with large datasets."
                        ),
                    },
                    "num_train_epochs": {
                        "type": "integer",
                        "description": (
                            "Number of full passes over the training dataset. "
                            "Use 1–2 for large datasets (>5000 samples). "
                            "Use 3–5 for medium datasets (500–5000 samples). "
                            "Use 5–10 for small datasets (<500 samples)."
                        ),
                    },
                    "per_device_train_batch_size": {
                        "type": "integer",
                        "description": (
                            "Batch size per GPU. Larger = faster training but more VRAM. "
                            "Typical values: 4, 8, 16. "
                            "Use 4–8 for 7B–8B models. Use 2–4 for 70B models."
                        ),
                    },
                    "warmup_ratio": {
                        "type": "number",
                        "description": (
                            "Fraction of total steps used for linear LR warmup (0.0 – 0.2). "
                            "Use 0.05–0.1 for standard training. "
                            "Use higher values (0.1–0.2) for very small datasets."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "One-sentence explanation of your hyperparameter choices "
                            "based on the dataset statistics provided."
                        ),
                    },
                },
                "required": [
                    "learning_rate",
                    "num_train_epochs",
                    "per_device_train_batch_size",
                    "warmup_ratio",
                    "reasoning",
                ],
            }
        },
    }
}

_HYPERPARAM_SYSTEM_PROMPT = """\
You are the Training Configuration agent for Modifai, an automated LLM fine-tuning platform.

Your ONLY job is to call the `set_training_hyperparameters` tool with the optimal
SageMaker fine-tuning configuration for the given dataset.

DECISION RULES:

learning_rate:
  - Large model (≥70B params)  → 1e-5 to 3e-5
  - Medium model (7B–13B)      → 2e-5 to 1e-4
  - Small model (<7B)          → 5e-5 to 3e-4

num_train_epochs:
  - >5000 samples  → 1–2 epochs (enough data, avoid overfitting)
  - 500–5000       → 2–4 epochs (standard)
  - <500 samples   → 4–8 epochs (need more passes on small data)

per_device_train_batch_size:
  - 70B model    → 1–2 (memory constrained)
  - 7B–13B       → 4–8
  - <7B          → 8–16

warmup_ratio:
  - Always between 0.03 and 0.15
  - Use 0.1 for small datasets (<500 samples), 0.05 for larger

You MUST call the set_training_hyperparameters tool. Output nothing else.
"""


# ── TrainingAgent class ───────────────────────────────────────────────────────


class TrainingAgent:
    """
    Decides SageMaker fine-tuning hyperparameters via LLM, then submits and
    monitors a SageMaker training job to completion.

    This agent bridges the dataset generation phase (Critic/Curriculum loop)
    and the deployment phase, completing the full Modifai pipeline.

    Usage:
        agent = TrainingAgent(
            role_arn="arn:aws:iam::527371380408:role/ModifaiSageMakerRole",
            s3_bucket="modifai-newbucket",
            base_model_id="llama3-8b",   # key from SUPPORTED_SAGEMAKER_MODELS
        )
        result = agent.run(
            samples=final_samples,
            dataset_stats=final_stats,
            job_id="abc123",
            custom_model_name="modifai-hr-policy-v1",
        )
    """

    def __init__(
        self,
        role_arn: str,
        s3_bucket: str,
        base_model_id: str = "llama3-8b",
        model_id: Optional[str] = None,       # Bedrock model for LLM decisions
        region: Optional[str] = None,
        instance_type: Optional[str] = None,  # Override SageMaker instance type
        max_retries: int = 1,
        poll_interval_seconds: int = 60,
        max_wait_seconds: int = 10800,         # 3 hours
    ):
        """
        Args:
            role_arn:               IAM role ARN with SageMaker + S3 + ECR permissions.
            s3_bucket:              S3 bucket for dataset upload and model output.
            base_model_id:          Short key from SUPPORTED_SAGEMAKER_MODELS (e.g. "llama3-8b")
                                    or a full HuggingFace model ID string.
            model_id:               Bedrock model for the hyperparameter decision step.
            region:                 AWS region override.
            instance_type:          Override the default SageMaker instance type.
            max_retries:            Bedrock retries for the hyperparameter tool call.
            poll_interval_seconds:  Seconds between SageMaker job status polls.
            max_wait_seconds:       Max time to wait for job completion (default 3h).
        """
        self.role_arn = role_arn
        self.s3_bucket = s3_bucket
        self.region = region or _DEFAULT_REGION

        # Resolve base_model_id: accept short key OR full HuggingFace model string
        if base_model_id in SUPPORTED_SAGEMAKER_MODELS:
            model_info = SUPPORTED_SAGEMAKER_MODELS[base_model_id]
            self.hf_model_id = model_info["model_id"]
            self.instance_type = instance_type or model_info["instance"]
        else:
            # Treat it as a raw HuggingFace model ID (e.g. "mistralai/Mistral-7B-v0.1")
            self.hf_model_id = base_model_id
            self.instance_type = instance_type or "ml.g5.2xlarge"

        # Bedrock model for the LLM decision step
        self.model_id = model_id or os.environ.get(
            "AWS_MODEL_ID", "amazon.nova-micro-v1:0"
        )
        self.max_retries = max_retries
        self.poll_interval_seconds = poll_interval_seconds
        self.max_wait_seconds = max_wait_seconds

        self._bedrock = boto3.client("bedrock-runtime", region_name=self.region)
        self._sagemaker = boto3.client("sagemaker", region_name=self.region)
        self._s3 = boto3.client("s3", region_name=self.region)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        samples: List[Dict[str, Any]],
        dataset_stats: Dict[str, Any],
        job_id: Optional[str] = None,
        custom_model_name: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Full training run: decide hyperparameters → upload dataset → submit job → poll.

        Args:
            samples:           Final accepted/rewritten samples from the Critic loop.
                               Each dict must have: instruction, input, output (or response).
            dataset_stats:     Stats dict from CriticAgent.run_batch() (total, accept_pct, etc.)
                               Used by the LLM to inform hyperparameter selection.
            job_id:            Unique run identifier. Auto-generated if not provided.
            custom_model_name: Human-readable name for the trained model artifact.
            dry_run:           If True, decide hyperparameters and upload dataset but
                               do NOT submit the SageMaker job. Useful for cost-free testing.

        Returns:
            Dict with:
              - job_id (str)
              - job_name (str | None)      — SageMaker training job name
              - hf_model_id (str)          — HuggingFace base model used
              - instance_type (str)        — SageMaker instance type
              - hyperparameters (dict)     — decided by LLM
              - hp_reasoning (str)         — LLM's reasoning for the choice
              - dataset_s3_uri (str)       — S3 location of the uploaded training JSONL
              - model_s3_uri (str | None)  — S3 prefix for model artifacts (after completion)
              - status (str)               — "Completed" | "dry_run" | "Failed"
        """
        job_id = job_id or str(uuid.uuid4())[:8]
        custom_model_name = custom_model_name or f"modifai-model-{job_id}"
        sagemaker_job_name = f"modifai-ft-{job_id}"

        logger.info("=" * 60)
        logger.info("TrainingAgent — Job: %s", job_id)
        logger.info("  Base model:   %s", self.hf_model_id)
        logger.info("  Instance:     %s", self.instance_type)
        logger.info("  Dataset size: %d samples", len(samples))
        logger.info("=" * 60)

        # ── Step 1: LLM decides hyperparameters ───────────────────────────────
        logger.info("[1/3] Deciding hyperparameters via LLM...")
        hyperparameters, hp_reasoning = self._decide_hyperparameters(
            sample_count=len(samples),
            dataset_stats=dataset_stats,
            hf_model_id=self.hf_model_id,
        )
        logger.info(
            "      LR=%.0e  epochs=%d  batch=%d  warmup=%.2f",
            hyperparameters["learning_rate"],
            hyperparameters["num_train_epochs"],
            hyperparameters["per_device_train_batch_size"],
            hyperparameters["warmup_ratio"],
        )
        logger.info("      Reasoning: %s", hp_reasoning)

        # ── Step 2: Format dataset and upload to S3 ───────────────────────────
        logger.info("[2/3] Uploading training dataset to S3...")
        dataset_s3_uri = self._upload_dataset(samples=samples, job_id=job_id)
        output_s3_uri = f"s3://{self.s3_bucket}/modifai-jobs/{job_id}/sagemaker-output/"
        logger.info("      Dataset:  %s", dataset_s3_uri)
        logger.info("      Output:   %s", output_s3_uri)

        if dry_run:
            logger.info("DRY RUN: skipping SageMaker job submission.")
            return {
                "job_id": job_id,
                "job_name": None,
                "hf_model_id": self.hf_model_id,
                "instance_type": self.instance_type,
                "hyperparameters": hyperparameters,
                "hp_reasoning": hp_reasoning,
                "dataset_s3_uri": dataset_s3_uri,
                "model_s3_uri": None,
                "status": "dry_run",
            }

        # ── Step 3: Submit SageMaker training job ─────────────────────────────
        logger.info("[3/3] Submitting SageMaker training job: %s", sagemaker_job_name)
        self._submit_training_job(
            job_name=sagemaker_job_name,
            dataset_s3_uri=dataset_s3_uri,
            output_s3_uri=output_s3_uri,
            hyperparameters=hyperparameters,
        )
        logger.info("      Job submitted — polling for completion...")

        # ── Poll until done ───────────────────────────────────────────────────
        final_status = self._poll_until_complete(sagemaker_job_name)
        model_s3_uri = f"{output_s3_uri}{sagemaker_job_name}/output/model.tar.gz"

        logger.info("=" * 60)
        logger.info("TrainingAgent complete — status: %s", final_status)
        logger.info("  Model artifacts: %s", model_s3_uri)
        logger.info("=" * 60)

        return {
            "job_id": job_id,
            "job_name": sagemaker_job_name,
            "hf_model_id": self.hf_model_id,
            "instance_type": self.instance_type,
            "hyperparameters": hyperparameters,
            "hp_reasoning": hp_reasoning,
            "dataset_s3_uri": dataset_s3_uri,
            "model_s3_uri": model_s3_uri if final_status == "Completed" else None,
            "status": final_status,
        }

    @staticmethod
    def list_supported_models() -> List[Dict]:
        """
        Return all SageMaker-compatible base models available for fine-tuning.
        Call this from your UI or API to show users what they can choose from.

        Returns:
            List of dicts: key, label, model_id, instance, notes
        """
        return [
            {"key": key, **info}
            for key, info in SUPPORTED_SAGEMAKER_MODELS.items()
        ]

    # ── Internal: LLM hyperparameter decision ──────────────────────────────────

    def _decide_hyperparameters(
        self,
        sample_count: int,
        dataset_stats: Dict[str, Any],
        hf_model_id: str,
    ) -> tuple[Dict[str, Any], str]:
        """
        Call Bedrock to get optimal hyperparameters for this dataset.
        Returns (hyperparameters_dict, reasoning_string).
        """
        # Estimate param count from model name for the LLM's guidance
        param_hint = "7B–8B"
        if "70b" in hf_model_id.lower() or "70B" in hf_model_id:
            param_hint = "70B"
        elif "phi" in hf_model_id.lower() or "mini" in hf_model_id.lower():
            param_hint = "<4B"
        elif "gemma-2-9b" in hf_model_id.lower():
            param_hint = "9B"

        user_message = (
            f"Dataset statistics:\n"
            f"  Total samples:  {sample_count}\n"
            f"  Accept pct:     {dataset_stats.get('accept_pct', 0):.1f}%\n"
            f"  Accepted:       {dataset_stats.get('accepted', sample_count)}\n"
            f"  Rewritten:      {dataset_stats.get('rewritten', 0)}\n\n"
            f"Base model: {hf_model_id} (~{param_hint} parameters)\n"
            f"Instance type: {self.instance_type}\n\n"
            f"Choose the best SageMaker fine-tuning hyperparameters and call "
            f"set_training_hyperparameters."
        )

        attempt = 0
        while attempt <= self.max_retries:
            try:
                response = self._bedrock.converse(
                    modelId=self.model_id,
                    system=[{"text": _HYPERPARAM_SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": [{"text": user_message}]}],
                    toolConfig={
                        "tools": [_HYPERPARAM_TOOL_SPEC],
                        "toolChoice": {"tool": {"name": "set_training_hyperparameters"}},
                    },
                )
                hp = self._parse_hyperparam_tool_output(response)
                self._validate_hyperparameters(hp)
                reasoning = hp.pop("reasoning", "")
                return hp, reasoning

            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error(
                        "TrainingAgent: hyperparameter decision failed after %d attempts: %s. "
                        "Using safe defaults.",
                        self.max_retries + 1,
                        exc,
                    )
                    return self._safe_default_hyperparameters(sample_count), "LLM fallback — safe defaults used."
                logger.warning(
                    "TrainingAgent: malformed hyperparameter response (attempt %d/%d): %s — retrying",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
            except ClientError as exc:
                logger.error("Bedrock API error during hyperparameter decision: %s", exc)
                return self._safe_default_hyperparameters(sample_count), "Bedrock error — safe defaults used."

    def _parse_hyperparam_tool_output(self, response: dict) -> dict:
        content_blocks = response["output"]["message"]["content"]
        for block in content_blocks:
            if block.get("toolUse", {}).get("name") == "set_training_hyperparameters":
                return block["toolUse"]["input"]
        raise ValueError(
            f"Model did not call set_training_hyperparameters tool. "
            f"Response content: {content_blocks}"
        )

    def _validate_hyperparameters(self, hp: dict) -> None:
        lr = hp.get("learning_rate")
        if not isinstance(lr, (int, float)) or not (1e-6 <= lr <= 1e-2):
            raise ValueError(f"learning_rate out of range: {lr}")

        epochs = hp.get("num_train_epochs")
        if not isinstance(epochs, int) or not (1 <= epochs <= 20):
            raise ValueError(f"num_train_epochs out of range: {epochs}")

        batch = hp.get("per_device_train_batch_size")
        if not isinstance(batch, int) or not (1 <= batch <= 64):
            raise ValueError(f"per_device_train_batch_size out of range: {batch}")

        warmup = hp.get("warmup_ratio")
        if not isinstance(warmup, (int, float)) or not (0.0 <= warmup <= 0.5):
            raise ValueError(f"warmup_ratio out of range: {warmup}")

    @staticmethod
    def _safe_default_hyperparameters(sample_count: int) -> Dict[str, Any]:
        """Conservative defaults that work for most models and dataset sizes."""
        epochs = 3 if sample_count >= 500 else 5
        return {
            "learning_rate": 2e-5,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": 4,
            "warmup_ratio": 0.1,
        }

    # ── Internal: dataset upload ──────────────────────────────────────────────

    def _upload_dataset(self, samples: List[Dict[str, Any]], job_id: str) -> str:
        """
        Format samples to SageMaker-compatible JSONL (prompt/completion format)
        and upload to S3. Returns the S3 URI.
        """
        lines = []
        for sample in samples:
            instruction = str(sample.get("instruction", "")).strip()
            input_text = str(sample.get("input", "")).strip()
            output_text = (
                str(sample.get("output", "")).strip()
                or str(sample.get("response", "")).strip()
            )

            if not instruction or not output_text:
                continue

            prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
            lines.append(json.dumps({"prompt": prompt, "completion": output_text}, ensure_ascii=False))

        if not lines:
            raise ValueError(
                "No valid samples to upload — all samples are missing instruction or output."
            )

        jsonl_content = "\n".join(lines)
        s3_key = f"modifai-jobs/{job_id}/training_data.jsonl"
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"

        self._s3.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=jsonl_content.encode("utf-8"),
            ContentType="application/jsonl",
        )
        logger.info("Uploaded %d training samples to %s", len(lines), s3_uri)
        return s3_uri

    # ── Internal: SageMaker job submission ────────────────────────────────────

    def _submit_training_job(
        self,
        job_name: str,
        dataset_s3_uri: str,
        output_s3_uri: str,
        hyperparameters: Dict[str, Any],
    ) -> None:
        """
        Submit a SageMaker training job using the HuggingFace DLC estimator config.
        All hyperparameters are passed as strings (SageMaker requirement).
        """
        # SageMaker requires all hyperparameter values as strings
        str_hyperparams = {
            "model_id":                     self.hf_model_id,
            "learning_rate":                str(hyperparameters["learning_rate"]),
            "num_train_epochs":             str(hyperparameters["num_train_epochs"]),
            "per_device_train_batch_size":  str(hyperparameters["per_device_train_batch_size"]),
            "warmup_ratio":                 str(hyperparameters["warmup_ratio"]),
            "gradient_checkpointing":       "true",
            "bf16":                         "true",      # Use bfloat16 mixed precision
            "merge_adapters":               "true",      # Merge LoRA weights into base model
        }

        try:
            self._sagemaker.create_training_job(
                TrainingJobName=job_name,
                HyperParameters=str_hyperparams,
                AlgorithmSpecification={
                    # SageMaker HuggingFace DLC image (transformers + PEFT + LoRA)
                    "TrainingImage": (
                        f"763104351884.dkr.ecr.{self.region}.amazonaws.com/"
                        "huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04"
                    ),
                    "TrainingInputMode": "File",
                },
                RoleArn=self.role_arn,
                InputDataConfig=[
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": dataset_s3_uri.rsplit("/", 1)[0] + "/",
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "ContentType": "application/jsonl",
                    }
                ],
                OutputDataConfig={"S3OutputPath": output_s3_uri},
                ResourceConfig={
                    "InstanceType": self.instance_type,
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 200,
                },
                StoppingCondition={
                    "MaxRuntimeInSeconds": self.max_wait_seconds,
                },
            )
            logger.info("SageMaker training job created: %s", job_name)

        except ClientError as exc:
            raise RuntimeError(
                f"Failed to create SageMaker training job '{job_name}': {exc}"
            ) from exc

    # ── Internal: polling ─────────────────────────────────────────────────────

    def _poll_until_complete(self, job_name: str) -> str:
        """
        Poll SageMaker until the training job reaches a terminal status.
        Returns the final status string: "Completed" | "Failed" | "Stopped".
        Raises RuntimeError on timeout or failure.
        """
        terminal_statuses = {"Completed", "Failed", "Stopped"}
        elapsed = 0

        while elapsed < self.max_wait_seconds:
            time.sleep(self.poll_interval_seconds)
            elapsed += self.poll_interval_seconds

            try:
                response = self._sagemaker.describe_training_job(
                    TrainingJobName=job_name
                )
                status = response["TrainingJobStatus"]
                secondary = response.get("SecondaryStatus", "")

                logger.info(
                    "Job '%s' status: %s (%s) — elapsed %ds / %ds",
                    job_name, status, secondary, elapsed, self.max_wait_seconds,
                )

                if status in terminal_statuses:
                    if status == "Failed":
                        failure_reason = response.get("FailureReason", "unknown")
                        raise RuntimeError(
                            f"SageMaker training job '{job_name}' failed: {failure_reason}"
                        )
                    return status

            except RuntimeError:
                raise
            except ClientError as exc:
                logger.warning("Status poll error (will retry): %s", exc)

        raise RuntimeError(
            f"SageMaker training job '{job_name}' did not complete within "
            f"{self.max_wait_seconds}s ({self.max_wait_seconds // 3600}h)."
        )
