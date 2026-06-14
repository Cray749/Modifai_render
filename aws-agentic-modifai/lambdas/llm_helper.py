"""
llm_helper.py — Shared OpenRouter LLM client for all pipeline Lambdas.
Drop-in replacement for the old gemini_helper.py.

API-key resolution order
------------------------
1. OPENROUTER_API_KEY  environment variable  (local / CI fast-path)
2. AWS Secrets Manager  →  secret name from OR_SECRET_NAME env var
   (default: "modifai/or")  →  JSON payload: {"api_key": "..."}

Model fallback chain
--------------------
On 429 (rate-limit) or 5xx the caller automatically retries the next
model in the fallback list before raising.

Primary  : OR_MODEL env var  (default: deepseek/deepseek-chat-v3)
Fallback : deepseek/deepseek-chat-v3
           qwen/qwen3-235b-a22b
           google/gemini-2.5-flash-lite

Public surface
--------------
call_llm(prompt, system, model, temperature, max_output_tokens)      → str
call_llm_json(prompt, system, model, temperature, max_output_tokens)  → dict
"""

import json
import logging
import os
import re
import time

import boto3
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── tuneable defaults (all overridable via env vars) ─────────────────────────
OPENROUTER_API_BASE       = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL             = os.environ.get("OR_MODEL",            "deepseek/deepseek-chat-v3")
DEFAULT_TEMPERATURE       = float(os.environ.get("OR_TEMPERATURE", "0.3"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("OR_MAX_TOKENS",    "2048"))
MAX_RETRIES               = int(os.environ.get("OR_MAX_RETRIES",   "3"))
RETRY_BASE_DELAY_S        = float(os.environ.get("OR_RETRY_DELAY", "1.5"))

# Models tried in order when the primary fails with 429 / 5xx
_FALLBACK_MODELS = [
    "deepseek/deepseek-chat-v3",
    "qwen/qwen3-235b-a22b",
    "google/gemini-2.5-flash-lite",
]


# ── key resolution ────────────────────────────────────────────────────────────

def _resolve_api_key() -> str:
    """Return the OpenRouter API key, or raise RuntimeError if unavailable."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return api_key

    secret_name = os.environ.get("OR_SECRET_NAME", "modifai/or")
    region      = os.environ.get("AWS_REGION", "ap-south-1")
    session     = boto3.session.Session()
    sm          = session.client("secretsmanager", region_name=region)
    try:
        payload = sm.get_secret_value(SecretId=secret_name)
        secret  = json.loads(payload["SecretString"])
        api_key = (secret.get("api_key") or secret.get("OPENROUTER_API_KEY", "")).strip()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot retrieve OpenRouter API key from Secrets Manager "
            f"(secret='{secret_name}', region='{region}'): {exc}"
        ) from exc

    if not api_key:
        raise RuntimeError(
            "OpenRouter API key resolved to an empty string.  "
            "Set OPENROUTER_API_KEY env var or store it in Secrets Manager "
            f"under '{secret_name}' as {{\"api_key\": \"<key>\"}}."
        )
    return api_key


# ── low-level HTTP call ───────────────────────────────────────────────────────

def _call_openrouter(
    api_key:           str,
    model:             str,
    messages:          list,
    temperature:       float,
    max_output_tokens: int,
) -> str:
    """
    POST to OpenRouter /chat/completions and return the assistant message text.
    Raises requests.HTTPError on non-2xx responses.
    """
    headers = {
        "Authorization":  f"Bearer {api_key}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   "https://github.com/modifai",   # recommended by OpenRouter
        "X-Title":        "ModifAI Pipeline",
    }
    body = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_output_tokens,
    }
    response = requests.post(
        OPENROUTER_API_BASE,
        headers=headers,
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # OpenAI-compatible response shape
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Unexpected OpenRouter response shape: {json.dumps(data)[:500]}"
        ) from exc


# ── core call with retry + model fallback ────────────────────────────────────

def call_llm(
    prompt:            str,
    system:            str   = "",
    model:             str   = None,
    temperature:       float = DEFAULT_TEMPERATURE,
    max_output_tokens: int   = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """
    Send *prompt* to OpenRouter and return the response text.

    Retries up to MAX_RETRIES times on transient errors with exponential
    back-off.  On 429 or 5xx, automatically falls back to the next model
    in the fallback chain.  Raises RuntimeError if all attempts fail.
    """
    api_key = _resolve_api_key()

    # Build the ordered list of models to try
    primary = model or DEFAULT_MODEL
    model_queue = [primary] + [m for m in _FALLBACK_MODELS if m != primary]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None

    for model_id in model_queue:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                text = _call_openrouter(
                    api_key=api_key,
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                logger.info(
                    "LLM call succeeded — model=%s attempt=%d/%d",
                    model_id, attempt, MAX_RETRIES,
                )
                return text

            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                last_exc = exc
                # 429 / 5xx → try next model immediately (after short delay)
                if status_code == 429 or status_code >= 500:
                    delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM HTTP %d on model=%s (attempt %d/%d) — retrying in %.1fs",
                        status_code, model_id, attempt, MAX_RETRIES, delay,
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "Exhausted retries for model=%s — trying next fallback.", model_id
                        )
                        break   # move to next model
                else:
                    # 4xx client error other than 429 — not transient, bail immediately
                    raise RuntimeError(
                        f"OpenRouter client error {status_code} on model={model_id}: {exc}"
                    ) from exc

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    "LLM call failed on model=%s (attempt %d/%d): %s — retrying in %.1fs",
                    model_id, attempt, MAX_RETRIES, exc, delay,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                else:
                    logger.warning(
                        "Exhausted retries for model=%s — trying next fallback.", model_id
                    )
                    break   # move to next model

    raise RuntimeError(
        f"LLM call failed across all models {model_queue} after {MAX_RETRIES} "
        f"attempts each: {last_exc}"
    ) from last_exc


def call_llm_json(
    prompt:            str,
    system:            str   = "",
    model:             str   = None,
    temperature:       float = DEFAULT_TEMPERATURE,
    max_output_tokens: int   = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict:
    """
    Like call_llm(), but parses the response as JSON and returns a dict.

    Strips leading/trailing whitespace, backtick fences, and the 'json'
    language tag before parsing so that models that ignore "output ONLY JSON"
    instructions still work.

    Raises ValueError if the response cannot be parsed as JSON.
    """
    raw = call_llm(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    cleaned = _strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON content.\n"
            f"Raw response (first 500 chars): {raw[:500]!r}\n"
            f"Parse error: {exc}"
        ) from exc


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$",          "", text)
    return text.strip()
