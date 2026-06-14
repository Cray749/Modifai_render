"""
full_pipeline.py — End-to-end Modifai pipeline: PDF → fine-tuned deployed model.

This is the top-level entry point that wires all layers together:
  1. OCR (Textract)           → raw text
  2. Chunking                 → text chunks
  3. Agentic dataset gen      → clean training samples (EXISTING pipeline)
  4. Formatting + S3 upload   → training JSONL on S3
  5. Bedrock fine-tuning      → custom model ARN
  6. Deployment (provisioning)→ live inference endpoint ARN
  7. (Optional) Test query    → validates the deployed model works

Usage:
    from modifai.core.full_pipeline import run_full_pipeline

    result = run_full_pipeline(
        pdf_path="hr_policy.pdf",
        goal="Build a Q&A bot for our HR policy",
        s3_bucket="my-modifai-bucket",
        role_arn="arn:aws:iam::123456789012:role/ModifaiBedrockRole",
        custom_model_name="modifai-hr-policy-v1",
        test_questions=["What is the leave policy?", "How do I submit expenses?"],
    )
    print("Endpoint ARN:", result["endpoint_arn"])
    print("Test answers:", result["test_answers"])
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Dict, List, Optional, Any

from modifai.core.text_extraction import extract_text_from_file
from modifai.core.chunking import chunk_text
from modifai.agents.pipeline_loop import run_agentic_loop
from modifai.agents.training_agent import TrainingAgent
from modifai.core.formatter import format_and_upload_to_s3
from modifai.core.finetuning import start_finetuning_job, wait_for_job, list_supported_models
from modifai.core.deployment import provision_model
from modifai.core.inference import batch_query

logger = logging.getLogger(__name__)

_DEFAULT_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
_DEFAULT_BUCKET = os.environ.get("S3_BUCKET_NAME", "modifai-newbucket")


def run_full_pipeline(
    pdf_path: str,
    goal: str,
    s3_bucket: str,
    role_arn: str,
    custom_model_name: str,
    base_model_id: str = "titan-text-express",
    use_sagemaker: bool = False,
    doc_domain: str = "general",
    max_iterations: int = 3,
    job_id: Optional[str] = None,
    test_questions: Optional[List[str]] = None,
    region: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run the full end-to-end Modifai pipeline.

    Args:
        pdf_path:           Path to the input PDF file.
        goal:               Fine-tuning goal (e.g. "Build a Q&A bot for our HR policy").
        s3_bucket:          S3 bucket for training data and model output (must be in us-east-1).
        role_arn:           IAM role ARN with AmazonBedrockFullAccess and S3 access.
        custom_model_name:  Name for the fine-tuned model (alphanumeric + hyphens).
        base_model_id:      Base model to fine-tune on.
                            - If use_sagemaker=True: short key from TrainingAgent.list_supported_models()
                              (e.g. "llama3-8b", "mistral-7b") or a full HuggingFace model ID.
                            - If use_sagemaker=False: short key from finetuning.list_supported_models()
                              (e.g. "nova-lite", "titan-text-express") or a full Bedrock model ID.
        use_sagemaker:      If True, fine-tunes using SageMaker + HuggingFace DLC (supports
                            open-weight models like Llama 3, Mistral, Phi-3).
                            If False (default), uses Bedrock model customisation API
                            (faster, managed, supports Titan/Nova models).
        doc_domain:         Document domain (e.g. "HR policy", "engineering runbook").
        max_iterations:     Max Critic→Curriculum loop iterations (default 3).
        job_id:             Unique run ID for S3 paths. Auto-generated if not provided.
        test_questions:     Optional list of test questions to send to the deployed model.
        region:             AWS region override (default: us-east-1).
        dry_run:            If True, stops after dataset generation (no fine-tuning/deployment).
                            Use for testing the dataset quality without spending on fine-tuning.

    Returns:
        Dict with:
          - job_id (str)
          - chunks_count (int)
          - samples_count (int)
          - accept_pct (float)
          - dataset_s3_uri (str)
          - finetuning_job_name (str) — None if dry_run=True
          - custom_model_arn (str)    — None if dry_run=True
          - endpoint_arn (str)        — None if dry_run=True
          - test_answers (list[dict]) — None if no test_questions provided
          - exit_reason (str)         — from the agentic loop
          - events (list[dict])       — all agent events (for P3 dashboard)
    """
    region = region or _DEFAULT_REGION
    job_id = job_id or str(uuid.uuid4())[:8]
    event_log_path = f"agent_events_{job_id}.jsonl"

    logger.info("=" * 60)
    logger.info("Modifai Full Pipeline — Job: %s", job_id)
    logger.info("=" * 60)

    logger.info("[1/6] Extracting text from PDF: %s", pdf_path)
    raw_text = extract_text_from_file(pdf_path, region=region)
    logger.info("      Extracted %d characters.", len(raw_text))

    # ── Step 2: Chunking ──────────────────────────────────────────────────────
    logger.info("[2/6] Chunking text...")
    chunks = chunk_text(raw_text, target_tokens=512, overlap_tokens=64)
    logger.info("      Created %d chunks.", len(chunks))

    if not chunks:
        raise ValueError(
            "No text chunks produced from PDF. "
            "Check that the PDF has readable text (not just images)."
        )

    # ── Step 3: Agentic Dataset Generation ───────────────────────────────────
    logger.info("[3/6] Running agentic dataset generation (max %d iterations)...", max_iterations)
    from pathlib import Path
    pdf_filename = Path(pdf_path).name
    page_count_estimate = max(1, len(raw_text) // 3000)  # rough: ~3000 chars/page

    loop_state = run_agentic_loop(
        goal=goal,
        doc_metadata={
            "filename": pdf_filename,
            "page_count": page_count_estimate,
            "domain": doc_domain,
            "estimated_chunk_count": len(chunks),
        },
        chunks=chunks,
        max_iterations=max_iterations,
        event_log_path=event_log_path,
        region=region,
    )

    final_samples = loop_state["final_samples"]
    accept_pct = loop_state["final_stats"]["accept_pct"]

    logger.info(
        "      Dataset generated: %d samples (accept_pct=%.1f%%, exit=%s)",
        len(final_samples), accept_pct, loop_state["exit_reason"],
    )

    if dry_run:
        logger.info("DRY RUN: stopping after dataset generation (no fine-tuning).")
        return {
            "job_id": job_id,
            "chunks_count": len(chunks),
            "samples_count": len(final_samples),
            "accept_pct": accept_pct,
            "dataset_s3_uri": None,
            "finetuning_job_name": None,
            "custom_model_arn": None,
            "endpoint_arn": None,
            "test_answers": None,
            "exit_reason": loop_state["exit_reason"],
            "events": loop_state["events"],
            "knowledge_analysis": loop_state.get("knowledge_analysis"),
            "virtual_mind": loop_state.get("virtual_mind"),
            "automation_discovery_output": loop_state.get("automation_discovery_output"),
        }

    # ── Step 4: Format + Upload to S3 ────────────────────────────────────────
    logger.info("[4/6] Formatting dataset and uploading to S3...")
    training_s3_uri = format_and_upload_to_s3(
        samples=final_samples,
        bucket=s3_bucket,
        job_id=job_id,
        region=region,
    )
    output_s3_uri = f"s3://{s3_bucket}/modifai-jobs/{job_id}/output/"
    logger.info("      Training data: %s", training_s3_uri)

    # ── Step 5: Fine-Tuning ───────────────────────────────────────────────────
    if use_sagemaker:
        # ── SageMaker path (open-weight models: Llama, Mistral, Phi-3, etc.) ──
        logger.info("[5/6] Starting SageMaker fine-tuning job (TrainingAgent)...")
        training_agent = TrainingAgent(
            role_arn=role_arn,
            s3_bucket=s3_bucket,
            base_model_id=base_model_id,
            region=region,
        )
        train_result = training_agent.run(
            samples=final_samples,
            dataset_stats=loop_state["final_stats"],
            job_id=job_id,
            custom_model_name=custom_model_name,
        )
        ft_job_name = train_result["job_name"]
        training_s3_uri = train_result["dataset_s3_uri"]
        # SageMaker produces model artifacts in S3, not a Bedrock model ARN
        custom_model_arn = train_result.get("model_s3_uri")  # S3 URI of model.tar.gz
        logger.info(
            "      SageMaker training complete! Status: %s, artifacts: %s",
            train_result["status"], custom_model_arn,
        )
    else:
        # ── Bedrock Customisation path (Titan, Nova models) ───────────────────
        logger.info("[5/6] Starting Bedrock fine-tuning job...")
        ft_job_name = start_finetuning_job(
            training_data_s3_uri=training_s3_uri,
            output_s3_uri=output_s3_uri,
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            base_model_id=base_model_id,
            region=region,
        )
        logger.info("      Fine-tuning job: %s (this takes 30–90 minutes...)", ft_job_name)
        custom_model_arn = wait_for_job(ft_job_name, region=region)
        logger.info("      Fine-tuning complete! Model ARN: %s", custom_model_arn)

    # ── Step 6: Deployment ────────────────────────────────────────────────────
    logger.info("[6/6] Provisioning model endpoint...")
    provisioned_model_name = f"{custom_model_name}-ep"[:63]
    endpoint_arn = provision_model(
        custom_model_arn=custom_model_arn,
        provisioned_model_name=provisioned_model_name,
        region=region,
    )
    logger.info("      Endpoint ready: %s", endpoint_arn)

    # ── Step 7 (optional): Test queries ──────────────────────────────────────
    test_answers = None
    if test_questions:
        logger.info("Running %d test queries against the deployed model...", len(test_questions))
        test_answers = batch_query(
            model_arn=endpoint_arn,
            questions=test_questions,
            region=region,
        )
        for qa in test_answers:
            logger.info("  Q: %s", qa.get("question", "?"))
            logger.info("  A: %s", qa["response"][:200])

    logger.info("=" * 60)
    logger.info("Pipeline complete! Job: %s", job_id)
    logger.info("  Samples used:  %d", len(final_samples))
    logger.info("  Accept pct:    %.1f%%", accept_pct)
    logger.info("  Endpoint ARN:  %s", endpoint_arn)
    logger.info("=" * 60)

    return {
        "job_id": job_id,
        "chunks_count": len(chunks),
        "samples_count": len(final_samples),
        "accept_pct": accept_pct,
        "dataset_s3_uri": training_s3_uri,
        "finetuning_job_name": ft_job_name,
        "custom_model_arn": custom_model_arn,
        "endpoint_arn": endpoint_arn,
        "test_answers": test_answers,
        "exit_reason": loop_state["exit_reason"],
        "events": loop_state["events"],
        "knowledge_analysis": loop_state.get("knowledge_analysis"),
        "virtual_mind": loop_state.get("virtual_mind"),
        "automation_discovery_output": loop_state.get("automation_discovery_output"),
    }
