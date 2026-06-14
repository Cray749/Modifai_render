"""
intent_analyzer.py — Lambda: analyse uploaded documents and produce a
fine-tuning strategy (intent, chunking, base model, hyperparameters).

All AI inference is routed through llm_helper.call_llm_json().
No Amazon Bedrock dependency.

Environment variables
---------------------
AWS_REGION          AWS region for S3 (default: ap-south-1)
OPENROUTER_API_KEY  OpenRouter API key  (or use Secrets Manager)
OR_SECRET_NAME      Secrets Manager secret name (default: modifai/or)
OR_MODEL            OpenRouter model ID (default: deepseek/deepseek-chat-v3)
BASE_MODEL          Base model identifier forwarded to fine_tuning_trigger
                    (default: meta.llama3-8b-instruct-v1:0)
SNIPPET_CHARS       Max chars extracted per document for context
                    (default: 1500)
MAX_DOCS_SAMPLED    Number of documents sampled for analysis (default: 3)
"""

import io
import logging
import os

import boto3
import PyPDF2

from llm_helper import call_llm_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-south-1"))

SNIPPET_CHARS    = int(os.environ.get("SNIPPET_CHARS",    "1500"))
MAX_DOCS_SAMPLED = int(os.environ.get("MAX_DOCS_SAMPLED", "3"))
BASE_MODEL       = os.environ.get(
    "BASE_MODEL", "meta.llama3-8b-instruct-v1:0"
)

_DEFAULT_STRATEGY = {
    "intent":   "QA",
    "chunking": {"strategy": "semantic", "max_tokens": 512, "overlap": 64},
    "model":    BASE_MODEL,
    "hyperparameters": {
        "epochs": 2, "batch_size": 8, "learning_rate": 0.00005,
    },
}

_SYSTEM_PROMPT = (
    "You are an AI Architect Agent. Analyse the provided document samples. "
    "Identify the intent (QA, summarization, or instruction). "
    "Recommend a chunking strategy and initial fine-tuning hyperparameters. "
    f"Use '{BASE_MODEL}' as the base model. "
    "Output ONLY valid JSON matching this exact schema — no markdown, no explanation:\n"
    "{\n"
    '  "intent": "QA",\n'
    '  "chunking": {"strategy": "semantic", "max_tokens": 512, "overlap": 64},\n'
    f'  "model": "{BASE_MODEL}",\n'
    '  "hyperparameters": {"epochs": 2, "batch_size": 8, "learning_rate": 0.00005}\n'
    "}"
)


# ── text extraction helpers ───────────────────────────────────────────────────

def _extract_snippet(bucket: str, key: str) -> str:
    """Download a file from S3 and return up to SNIPPET_CHARS of text."""
    ext = key.lower().rsplit(".", 1)[-1] if "." in key else ""
    file_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    if ext == "pdf":
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages[:2]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text[:SNIPPET_CHARS]
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF extraction failed for s3://%s/%s: %s", bucket, key, exc)
            return ""

    # Plain text / JSON / CSV / JSONL / etc.
    return file_bytes[:SNIPPET_CHARS].decode("utf-8", errors="ignore")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) from an s3://bucket/key URI."""
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


# ── lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    Expected event shape
    --------------------
    {
      "document_s3_uris": ["s3://bucket/path/doc1.pdf", ...],
      "document_s3_uri":  "s3://bucket/path/single.pdf"   # optional
    }
    """
    document_uris: list[str] = list(event.get("document_s3_uris", []))
    single_uri = event.get("document_s3_uri")
    if single_uri and single_uri not in document_uris:
        document_uris.append(single_uri)

    if not document_uris:
        logger.warning("No document URIs provided — returning default strategy.")
        return {
            "statusCode":       200,
            "strategy":         _DEFAULT_STRATEGY,
            "document_s3_uris": [],
        }

    # ── sample text from documents ────────────────────────────────────────────
    sample_texts: list[str] = []
    for doc_uri in document_uris[:MAX_DOCS_SAMPLED]:
        try:
            bucket, key = _parse_s3_uri(doc_uri)
            snippet = _extract_snippet(bucket, key)
            sample_texts.append(f"Snippet from {doc_uri}:\n{snippet}")
            logger.info("Extracted snippet from %s (%d chars)", doc_uri, len(snippet))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch snippet from %s: %s", doc_uri, exc)

    combined_sample = (
        "\n---\n".join(sample_texts) if sample_texts
        else "No text could be extracted from the supplied documents."
    )

    # ── LLM intent analysis ────────────────────────────────────────────────
    prompt = f"Document Samples:\n{combined_sample}"
    try:
        strategy = call_llm_json(prompt=prompt, system=_SYSTEM_PROMPT)
        # Ensure base model is not accidentally overridden to something unsupported
        strategy.setdefault("model", BASE_MODEL)
        logger.info("Intent analysis complete: intent=%s", strategy.get("intent"))
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM intent analysis failed — using default strategy: %s", exc)
        strategy = _DEFAULT_STRATEGY

    return {
        "statusCode":       200,
        "strategy":         strategy,
        "document_s3_uris": document_uris,
    }
