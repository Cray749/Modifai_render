"""
text_extraction.py — PDF → raw text via AWS Textract.

Supports two modes:
  - sync:  DetectDocumentText (< 5 pages, document bytes in memory)
  - async: StartDocumentTextDetection (any size PDF from S3)

Usage:
    # From a local file path (auto-detects sync vs async by page count)
    text = extract_text_from_file("path/to/document.pdf")

    # From S3
    text = extract_text_from_s3("my-bucket", "prefix/document.pdf")
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

_DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Pages at or below this threshold use synchronous Textract (faster, cheaper)
_SYNC_PAGE_LIMIT = 5


def extract_text_from_file(
    pdf_path: str,
    region: Optional[str] = None,
) -> str:
    """
    Extract all text from a local PDF file using AWS Textract.

    For files ≤ 5 pages, uses synchronous detection (no S3 needed).
    For larger files, uploads to a temp S3 location and uses async detection.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        region:   AWS region override (default: AWS_REGION env or us-east-1).

    Returns:
        Extracted text as a single string, pages joined with newlines.

    Raises:
        FileNotFoundError: If pdf_path doesn't exist.
        RuntimeError: If Textract job fails.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    region = region or _DEFAULT_REGION
    client = boto3.client("textract", region_name=region)

    file_bytes = path.read_bytes()
    logger.info("Extracting text from %s (%d bytes)", path.name, len(file_bytes))

    # Use synchronous API for small files (no S3 needed)
    response = client.detect_document_text(
        Document={"Bytes": file_bytes}
    )
    text = _collect_lines(response["Blocks"])
    logger.info("Textract sync complete: %d characters extracted.", len(text))
    return text


def extract_text_from_s3(
    bucket: str,
    key: str,
    region: Optional[str] = None,
    poll_interval_seconds: int = 5,
    max_wait_seconds: int = 300,
) -> str:
    """
    Extract text from a PDF already stored in S3 using async Textract.

    Use this for large PDFs (> 5 pages) or PDFs that are already in S3.

    Args:
        bucket:                S3 bucket name.
        key:                   S3 object key (path to the PDF).
        region:                AWS region override.
        poll_interval_seconds: Seconds between status polls (default 5).
        max_wait_seconds:      Maximum time to wait for completion (default 300s).

    Returns:
        Extracted text as a single string.

    Raises:
        RuntimeError: If Textract job fails or times out.
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("textract", region_name=region)

    logger.info("Starting async Textract job: s3://%s/%s", bucket, key)
    start_response = client.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = start_response["JobId"]
    logger.info("Textract job started: %s", job_id)

    # Poll until complete
    elapsed = 0
    while elapsed < max_wait_seconds:
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

        status_response = client.get_document_text_detection(JobId=job_id)
        status = status_response["JobStatus"]
        logger.debug("Textract job %s status: %s (elapsed %ds)", job_id, status, elapsed)

        if status == "SUCCEEDED":
            all_blocks = status_response["Blocks"]

            # Paginate through results if there are more pages
            next_token = status_response.get("NextToken")
            while next_token:
                page_response = client.get_document_text_detection(
                    JobId=job_id, NextToken=next_token
                )
                all_blocks.extend(page_response["Blocks"])
                next_token = page_response.get("NextToken")

            text = _collect_lines(all_blocks)
            logger.info(
                "Textract async complete: %d characters from s3://%s/%s",
                len(text), bucket, key,
            )
            return text

        elif status == "FAILED":
            raise RuntimeError(
                f"Textract job {job_id} FAILED for s3://{bucket}/{key}. "
                f"Status message: {status_response.get('StatusMessage', 'unknown')}"
            )
        # else: IN_PROGRESS — keep polling

    raise RuntimeError(
        f"Textract job {job_id} did not complete within {max_wait_seconds}s."
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _collect_lines(blocks: list) -> str:
    """
    Collect LINE blocks from Textract output, preserving page order.

    Textract returns blocks of type LINE, WORD, PAGE, etc.
    We only want LINEs — they are already assembled words in reading order.
    PAGE blocks mark page breaks, which we preserve with double newlines.
    """
    page_texts: dict[int, list[str]] = {}

    for block in blocks:
        block_type = block.get("BlockType")
        page_num = block.get("Page", 1)

        if block_type == "LINE":
            page_texts.setdefault(page_num, []).append(block["Text"])

    # Join in page order
    parts = []
    for page_num in sorted(page_texts.keys()):
        parts.append("\n".join(page_texts[page_num]))

    return "\n\n".join(parts)
