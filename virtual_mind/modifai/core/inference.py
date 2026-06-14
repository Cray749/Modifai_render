"""
inference.py — Calls a deployed Bedrock custom model endpoint.

After provisioning (deployment.py), the model is callable via the standard
Bedrock converse API using the provisioned_model_arn as the modelId.

Usage:
    from modifai.core.inference import query_model

    result = query_model(
        model_arn="arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123",
        question="What is the refund policy?",
        context="",  # optional additional context
    )
    print(result["response"])
    print(f"Latency: {result['latency_ms']}ms")
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

_DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")

_INFERENCE_SYSTEM_PROMPT = (
    "You are a helpful assistant trained on a specific document. "
    "Answer questions accurately and concisely based on your training. "
    "If you don't know the answer, say so clearly."
)


def query_model(
    model_arn: str,
    question: str,
    context: str = "",
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
    region: Optional[str] = None,
) -> dict:
    """
    Send a question to a deployed Bedrock custom model and return the response.

    Args:
        model_arn:     Provisioned model ARN from deployment.provision_model().
        question:      The user's question or instruction.
        context:       Optional additional context to include in the prompt.
        system_prompt: Override the default system prompt.
        max_tokens:    Maximum tokens in the response (default 512).
        temperature:   Response randomness 0.0–1.0 (default 0.3 for factual Q&A).
        region:        AWS region override.

    Returns:
        Dict with:
          - response (str): The model's answer.
          - model_arn (str): The model ARN used.
          - latency_ms (int): Round-trip time in milliseconds.
          - input_tokens (int): Approximate input token count.
          - output_tokens (int): Approximate output token count.

    Raises:
        RuntimeError: If the Bedrock call fails.
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock-runtime", region_name=region)

    # Build user message
    if context:
        user_text = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_text = question

    system = system_prompt or _INFERENCE_SYSTEM_PROMPT

    logger.debug("Querying model %s: %s", model_arn[:60], question[:80])
    start_time = time.time()

    try:
        response = client.converse(
            modelId=model_arn,
            system=[{"text": system}],
            messages=[
                {"role": "user", "content": [{"text": user_text}]}
            ],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )

        latency_ms = int((time.time() - start_time) * 1000)
        answer = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})

        logger.info(
            "Inference complete: %d chars in %dms", len(answer), latency_ms
        )

        return {
            "response": answer,
            "model_arn": model_arn,
            "latency_ms": latency_ms,
            "input_tokens": usage.get("inputTokens", -1),
            "output_tokens": usage.get("outputTokens", -1),
        }

    except Exception as exc:
        raise RuntimeError(
            f"Inference call failed for model {model_arn}: {exc}"
        ) from exc


def batch_query(
    model_arn: str,
    questions: list[str],
    region: Optional[str] = None,
) -> list[dict]:
    """
    Query the model with multiple questions sequentially.

    Args:
        model_arn: Provisioned model ARN.
        questions: List of question strings.
        region:    AWS region override.

    Returns:
        List of result dicts (same shape as query_model() output).
    """
    results = []
    for i, question in enumerate(questions):
        logger.info("Batch query %d/%d: %s", i + 1, len(questions), question[:60])
        try:
            result = query_model(model_arn=model_arn, question=question, region=region)
            results.append(result)
        except Exception as exc:
            logger.error("Query %d failed: %s", i, exc)
            results.append({
                "response": f"ERROR: {exc}",
                "model_arn": model_arn,
                "latency_ms": -1,
                "input_tokens": -1,
                "output_tokens": -1,
            })
    return results
