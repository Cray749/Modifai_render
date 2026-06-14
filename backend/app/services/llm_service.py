"""
LLM service — data quality evaluation and model comparison.

Uses OpenRouter (reusing the llm_helper.py pattern from the existing Lambdas)
with a fallback chain: deepseek → qwen → gemini-flash-lite.
"""

import json
import logging
import time
from typing import Any

import requests

from app.config import settings

logger = logging.getLogger(__name__)


# ── OpenRouter Client ───────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Get OpenRouter API key from settings or AWS Secrets Manager."""
    if settings.OPENROUTER_API_KEY:
        return settings.OPENROUTER_API_KEY

    # Fallback: try AWS Secrets Manager
    try:
        import boto3
        client = boto3.client("secretsmanager", region_name=settings.AWS_REGION)
        response = client.get_secret_value(SecretId=settings.OR_SECRET_NAME)
        secret = json.loads(response["SecretString"])
        return secret.get("OPENROUTER_API_KEY", "")
    except Exception as e:
        logger.warning("Could not retrieve API key from Secrets Manager: %s", e)
        return ""


def _call_openrouter(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    Call OpenRouter with automatic model fallback.

    Returns the assistant's response text.
    Raises RuntimeError if all models fail.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("No OpenRouter API key configured")

    models = [
        model or settings.OR_PRIMARY_MODEL,
        settings.OR_FALLBACK_1,
        settings.OR_FALLBACK_2,
    ]

    last_error = None
    for m in models:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": m,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.info("LLM response from %s (%d chars)", m, len(content))
            return content
        except Exception as e:
            last_error = e
            logger.warning("Model %s failed: %s — trying next", m, e)
            continue

    raise RuntimeError(f"All LLM models failed. Last error: {last_error}")


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response text (handles markdown code blocks)."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    import re
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } or [ ... ]
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


# ── Data Quality Evaluation ─────────────────────────────────────────────────────

def evaluate_data_quality(text_sample: str, intent: str) -> dict:
    """
    Evaluate a text sample for suitability as fine-tuning data.

    Args:
        text_sample: The raw text to evaluate.
        intent: The user's intended use case (e.g., "question-answering").

    Returns:
        { "score": float (0.0-1.0), "explanation": str }
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert data quality evaluator for LLM fine-tuning. "
                "Evaluate the given text sample for its suitability as training data. "
                "Consider: text clarity, information density, domain relevance, "
                "formatting quality, and noise level.\n\n"
                "You MUST respond with valid JSON only, no other text:\n"
                '{"score": <float 0.0-1.0>, "explanation": "<brief explanation>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Intent: {intent}\n\n"
                f"Text sample:\n{text_sample[:3000]}\n\n"
                "Rate this text sample from 0.0 to 1.0 for suitability as fine-tuning data. "
                "Return JSON only."
            ),
        },
    ]

    try:
        response_text = _call_openrouter(messages, temperature=0.2)
        result = _extract_json(response_text)

        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))  # Clamp

        return {
            "score": round(score, 2),
            "explanation": str(result.get("explanation", "Evaluation complete.")),
        }
    except Exception as e:
        logger.error("Data quality evaluation failed: %s", e)
        return {
            "score": 0.5,
            "explanation": f"Evaluation could not be completed: {str(e)}",
        }


# ── Model Comparison ────────────────────────────────────────────────────────────

def compare_models(project_id: str, prompt: str, system_prompt: str | None = None) -> dict:
    """
    Run inference on both base model and fine-tuned model, measuring latency.

    For now, uses OpenRouter for the base model and simulates a fine-tuned response.
    When a real fine-tuned endpoint is available, this will call both.

    Returns:
        {
            "base_model": { "response": str, "latency_ms": int, "model_id": str },
            "fine_tuned": { "response": str, "latency_ms": int, "model_id": str }
        }
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # ── Base Model ──
    base_start = time.time()
    try:
        base_response = _call_openrouter(messages, model=settings.OR_PRIMARY_MODEL)
        base_latency = int((time.time() - base_start) * 1000)
        base_result = {
            "response": base_response,
            "latency_ms": base_latency,
            "model_id": settings.OR_PRIMARY_MODEL,
        }
    except Exception as e:
        base_result = {
            "response": f"Error: {str(e)}",
            "latency_ms": int((time.time() - base_start) * 1000),
            "model_id": settings.OR_PRIMARY_MODEL,
            "error": str(e),
        }

    # ── Fine-Tuned Model ──
    # TODO: Replace with actual fine-tuned endpoint call when available.
    # For now, call a different model to simulate the comparison experience.
    ft_messages = list(messages)
    if system_prompt:
        # Enhance system prompt to simulate fine-tuning effect
        ft_messages[0] = {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                "You are a fine-tuned specialist model. Provide concise, "
                "domain-specific responses with higher accuracy and less filler."
            ),
        }
    else:
        ft_messages.insert(0, {
            "role": "system",
            "content": (
                "You are a fine-tuned specialist model. Provide concise, "
                "domain-specific responses with higher accuracy and less filler."
            ),
        })

    ft_start = time.time()
    try:
        ft_response = _call_openrouter(ft_messages, model=settings.OR_FALLBACK_1)
        ft_latency = int((time.time() - ft_start) * 1000)
        ft_result = {
            "response": ft_response,
            "latency_ms": ft_latency,
            "model_id": "fine-tuned-model",
        }
    except Exception as e:
        ft_result = {
            "response": f"Error: {str(e)}",
            "latency_ms": int((time.time() - ft_start) * 1000),
            "model_id": "fine-tuned-model",
            "error": str(e),
        }

    return {
        "base_model": base_result,
        "fine_tuned": ft_result,
    }
