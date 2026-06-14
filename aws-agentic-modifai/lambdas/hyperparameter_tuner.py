"""
hyperparameter_tuner.py — Lambda: decide whether to deploy or tune further.

Uses an LLM via OpenRouter to suggest improved hyperparameters when the
model doesn't yet meet the quality threshold.  No Amazon Bedrock dependency.

Weighted quality score
----------------------
weighted_score = (training_metrics_score * METRIC_WEIGHT)
               + (test_prompts_score      * TEST_WEIGHT)

Actions
-------
"deploy"               weighted_score >= QUALITY_THRESHOLD
"tune"                 below threshold and attempts remaining
"max_attempts_reached" below threshold and MAX_TUNING_ATTEMPTS exhausted

Environment variables
---------------------
QUALITY_THRESHOLD   Minimum weighted score to approve deployment (default: 0.85)
MAX_TUNING_ATTEMPTS Maximum tune iterations before giving up     (default: 3)
METRIC_WEIGHT       Weight for training-metrics score            (default: 0.4)
TEST_WEIGHT         Weight for test-prompts score                (default: 0.6)
OPENROUTER_API_KEY  OpenRouter API key  (or use Secrets Manager)
OR_SECRET_NAME      Secrets Manager secret (default: modifai/or)
OR_MODEL            OpenRouter model ID (default: deepseek/deepseek-chat-v3)
"""

import json
import logging
import os

from llm_helper import call_llm_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

QUALITY_THRESHOLD   = float(os.environ.get("QUALITY_THRESHOLD",   "0.85"))
MAX_TUNING_ATTEMPTS = int(os.environ.get("MAX_TUNING_ATTEMPTS",   "3"))
METRIC_WEIGHT       = float(os.environ.get("METRIC_WEIGHT",       "0.4"))
TEST_WEIGHT         = float(os.environ.get("TEST_WEIGHT",         "0.6"))

_SYSTEM_PROMPT = (
    "You are an AI Hyperparameter Tuning Agent for LLM fine-tuning. "
    "Given the current hyperparameters and quality scores, suggest improved "
    "hyperparameters that are likely to push the weighted score above the target. "
    "Make incremental, well-reasoned adjustments — do not make extreme changes. "
    "Output ONLY valid JSON: "
    '{"epochs": <int>, "batch_size": <int>, "learning_rate": <float>, '
    '"rationale": "<one sentence>"}'
)


def lambda_handler(event: dict, context) -> dict:
    """
    Expected event shape
    --------------------
    {
      "agent_decision": {          # populated on retry iterations
        "tuning_attempt": 1
      },
      "tuning_attempt": 0,         # alternative location (first call)
      "model_evaluation": {
        "training_metrics_score": 0.65,
        "test_prompts_score":     0.70
      },
      "job_info": {
        "hyperparameters": {"epochs": 2, "batch_size": 8, "learning_rate": 0.00005}
      }
    }
    """
    if event.get("config", {}).get("openrouter_api_key"):
        os.environ["OPENROUTER_API_KEY"] = event["config"]["openrouter_api_key"]

    previous_decision = event.get("agent_decision", {})
    attempt_count     = previous_decision.get(
        "tuning_attempt", event.get("tuning_attempt", 0)
    )

    evaluation   = event.get("model_evaluation", {})
    metric_score = float(evaluation.get("training_metrics_score", 0.8))
    test_score   = float(evaluation.get("test_prompts_score",     0.7))

    weighted_score = (metric_score * METRIC_WEIGHT) + (test_score * TEST_WEIGHT)
    logger.info(
        "Weighted score: %.4f (metric=%.4f×%.1f + test=%.4f×%.1f) | "
        "threshold=%.2f | attempt=%d/%d",
        weighted_score,
        metric_score, METRIC_WEIGHT,
        test_score,   TEST_WEIGHT,
        QUALITY_THRESHOLD,
        attempt_count, MAX_TUNING_ATTEMPTS,
    )

    # ── deploy? ───────────────────────────────────────────────────────────────
    if weighted_score >= QUALITY_THRESHOLD:
        return {
            "action": "deploy",
            "reason": (
                f"Model achieved weighted score {weighted_score:.4f} "
                f">= threshold {QUALITY_THRESHOLD}"
            ),
        }

    # ── max attempts reached? ─────────────────────────────────────────────────
    if attempt_count >= MAX_TUNING_ATTEMPTS:
        logger.warning("Max attempts reached. Proceeding to deploy best model.")
        return {
            "action": "deploy",
            "reason": (
                f"Failed to reach threshold {QUALITY_THRESHOLD} after "
                f"{MAX_TUNING_ATTEMPTS} attempt(s). Deploying best effort. "
                f"Best weighted score: {weighted_score:.4f}"
            ),
        }

    # ── ask LLM for better hyperparameters ─────────────────────────────────
    current_hp = event.get("job_info", {}).get("hyperparameters", {})
    prompt = (
        f"Current hyperparameters: {json.dumps(current_hp)}\n"
        f"Training metrics score : {metric_score:.4f}  (weight {METRIC_WEIGHT})\n"
        f"Test prompts score     : {test_score:.4f}  (weight {TEST_WEIGHT})\n"
        f"Weighted score         : {weighted_score:.4f}  (target >= {QUALITY_THRESHOLD})\n"
        f"Tuning attempt         : {attempt_count + 1}/{MAX_TUNING_ATTEMPTS}\n"
        "Suggest hyperparameter changes to improve the weighted score."
    )
    try:
        suggestion      = call_llm_json(prompt=prompt, system=_SYSTEM_PROMPT)
        new_hp          = {
            "epochs":        int(suggestion.get("epochs",        current_hp.get("epochs", 2))),
            "batch_size":    int(suggestion.get("batch_size",    current_hp.get("batch_size", 8))),
            "learning_rate": float(suggestion.get("learning_rate", current_hp.get("learning_rate", 0.00005))),
        }
        rationale = suggestion.get("rationale", "")
        logger.info("LLM suggested hyperparameters: %s | rationale: %s", new_hp, rationale)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM HP suggestion failed — incrementing epochs: %s", exc)
        new_hp    = {
            "epochs":        current_hp.get("epochs", 2) + 1,
            "batch_size":    current_hp.get("batch_size", 8),
            "learning_rate": current_hp.get("learning_rate", 0.00005),
        }
        rationale = "Fallback: incremented epochs due to LLM error."

    return {
        "action":              "tune",
        "new_hyperparameters": new_hp,
        "tuning_attempt":      attempt_count + 1,
        "reason": (
            f"Weighted score {weighted_score:.4f} < threshold {QUALITY_THRESHOLD}. "
            f"Tuning hyperparameters (attempt {attempt_count + 1}/{MAX_TUNING_ATTEMPTS}). "
            f"{rationale}"
        ),
    }
