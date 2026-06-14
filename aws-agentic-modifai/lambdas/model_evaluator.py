"""
model_evaluator.py — Lambda: evaluate a completed fine-tuning run.

Uses an LLM via OpenRouter as a critic agent to estimate test performance
from training metrics.  No Amazon Bedrock dependency.

Scoring
-------
metric_score   = max(0, 1 - training_loss)      — derived from training loss
test_score     = LLM estimate                   — critic-agent judgement
weighted_score = (metric_score * 0.4) + (test_score * 0.6)

Environment variables
---------------------
OPENROUTER_API_KEY  OpenRouter API key  (or use Secrets Manager)
OR_SECRET_NAME      Secrets Manager secret (default: modifai/or)
OR_MODEL            OpenRouter model ID (default: deepseek/deepseek-chat-v3)
"""

import logging
import os

from llm_helper import call_llm_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_SYSTEM_PROMPT = (
    "You are an AI Model Evaluator. "
    "Given a fine-tuning run's training loss and hyperparameters, estimate the "
    "model's generalisation ability as a 'test_score' between 0.0 and 1.0 "
    "(lower training loss = higher score, but also consider over-fitting risk). "
    "Output ONLY valid JSON: "
    '{"test_score": <float>, "confidence": <float 0-1>, "reasoning": "<one sentence>"}'
)


def lambda_handler(event: dict, context) -> dict:
    """
    Expected event shape
    --------------------
    {
      "job_status": {
        "training_metrics": {"trainingLoss": 0.35},
        "status": "Completed"
      },
      "job_info": {                          # optional — enriches LLM prompt
        "hyperparameters": {"epochs": 3, "batch_size": 8, "learning_rate": 0.00005},
        "base_model": "meta.llama3-8b-instruct-v1:0"
      }
    }
    """
    job_status       = event.get("job_status", {})
    training_metrics = job_status.get("training_metrics", {})
    training_loss    = float(training_metrics.get("trainingLoss", 0.5))

    # Derive metric score from training loss (lower loss → higher score)
    metric_score = max(0.0, min(1.0, 1.0 - training_loss))

    # Optional context to help LLM give a more grounded estimate
    job_info = event.get("job_info", {})
    hp       = job_info.get("hyperparameters", {})

    prompt = (
        f"Training Loss: {training_loss}\n"
        f"Metric Score (1 - loss): {metric_score:.4f}\n"
        f"Hyperparameters: epochs={hp.get('epochs', 'n/a')}, "
        f"batch_size={hp.get('batch_size', 'n/a')}, "
        f"learning_rate={hp.get('learning_rate', 'n/a')}\n"
        f"Base model: {job_info.get('base_model', 'unknown')}\n"
        "Estimate the test_score for this fine-tuned model."
    )

    try:
        eval_data  = call_llm_json(prompt=prompt, system=_SYSTEM_PROMPT)
        test_score = float(eval_data.get("test_score", 0.5))
        confidence = float(eval_data.get("confidence", 0.5))
        reasoning  = eval_data.get("reasoning", "")
        logger.info(
            "LLM evaluation: test_score=%.4f, confidence=%.4f, reasoning=%s",
            test_score, confidence, reasoning,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM evaluation failed — using fallback test_score=0.75: %s", exc)
        test_score = 0.75
        confidence = 0.5
        reasoning  = "Fallback score used due to LLM evaluation error."

    return {
        "training_metrics_score": round(metric_score, 4),
        "test_prompts_score":     round(test_score,   4),
        "confidence":             round(confidence,   4),
        "details":                reasoning,
    }
