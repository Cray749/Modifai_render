"""Unit tests for text_extraction.py — all AWS calls mocked."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_textract_blocks(lines_per_page: dict) -> list:
    """Build fake Textract blocks: {page_number: [line_texts]}."""
    blocks = []
    for page_num, lines in lines_per_page.items():
        blocks.append({"BlockType": "PAGE", "Page": page_num})
        for line in lines:
            blocks.append({"BlockType": "LINE", "Page": page_num, "Text": line})
    return blocks


def _make_sync_response(lines_per_page: dict) -> dict:
    return {"Blocks": _make_textract_blocks(lines_per_page)}


def _make_async_status_response(status: str, blocks: list = None, next_token=None) -> dict:
    r = {"JobStatus": status, "Blocks": blocks or []}
    if next_token:
        r["NextToken"] = next_token
    return r


# ── text_extraction tests ──────────────────────────────────────────────────────

@patch("modifai.core.text_extraction.boto3.client")
def test_extract_from_file_returns_text(mock_boto, tmp_path):
    """Sync Textract path: local PDF → text string."""
    # Create a dummy PDF file
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")

    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.detect_document_text.return_value = _make_sync_response({
        1: ["Hello world.", "This is page one."],
        2: ["Page two content.", "More text here."],
    })

    from modifai.core.text_extraction import extract_text_from_file
    text = extract_text_from_file(str(pdf_file))

    assert "Hello world." in text
    assert "This is page one." in text
    assert "Page two content." in text
    # Pages should be separated
    assert text.index("Hello world.") < text.index("Page two content.")


@patch("modifai.core.text_extraction.boto3.client")
def test_extract_from_file_raises_if_not_found(mock_boto):
    """Should raise FileNotFoundError for non-existent files."""
    from modifai.core.text_extraction import extract_text_from_file
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract_text_from_file("/nonexistent/path/doc.pdf")


@patch("modifai.core.text_extraction.time.sleep")
@patch("modifai.core.text_extraction.boto3.client")
def test_extract_from_s3_success(mock_boto, mock_sleep):
    """Async Textract path: S3 PDF → polls → returns text."""
    mock_client = MagicMock()
    mock_boto.return_value = mock_client

    mock_client.start_document_text_detection.return_value = {"JobId": "job-abc"}
    mock_client.get_document_text_detection.side_effect = [
        _make_async_status_response("IN_PROGRESS"),
        {
            "JobStatus": "SUCCEEDED",
            "Blocks": _make_textract_blocks({1: ["S3 document line 1.", "Line 2."]}),
        },
    ]

    from modifai.core.text_extraction import extract_text_from_s3
    text = extract_text_from_s3("my-bucket", "docs/test.pdf")

    assert "S3 document line 1." in text
    assert "Line 2." in text
    assert mock_client.start_document_text_detection.call_count == 1
    assert mock_client.get_document_text_detection.call_count == 2


@patch("modifai.core.text_extraction.time.sleep")
@patch("modifai.core.text_extraction.boto3.client")
def test_extract_from_s3_raises_on_failure(mock_boto, mock_sleep):
    """Failed Textract job should raise RuntimeError."""
    mock_client = MagicMock()
    mock_boto.return_value = mock_client

    mock_client.start_document_text_detection.return_value = {"JobId": "job-fail"}
    mock_client.get_document_text_detection.return_value = {
        "JobStatus": "FAILED",
        "StatusMessage": "Unsupported document format",
    }

    from modifai.core.text_extraction import extract_text_from_s3
    with pytest.raises(RuntimeError, match="FAILED"):
        extract_text_from_s3("my-bucket", "docs/bad.pdf", poll_interval_seconds=1)


@patch("modifai.core.text_extraction.time.sleep")
@patch("modifai.core.text_extraction.boto3.client")
def test_extract_from_s3_handles_pagination(mock_boto, mock_sleep):
    """Multi-page Textract results with NextToken pagination."""
    mock_client = MagicMock()
    mock_boto.return_value = mock_client

    mock_client.start_document_text_detection.return_value = {"JobId": "job-pages"}

    page1_blocks = _make_textract_blocks({1: ["Page 1 content."]})
    page2_blocks = _make_textract_blocks({2: ["Page 2 content."]})

    mock_client.get_document_text_detection.side_effect = [
        {
            "JobStatus": "SUCCEEDED",
            "Blocks": page1_blocks,
            "NextToken": "token-for-page2",
        },
        {
            "JobStatus": "SUCCEEDED",
            "Blocks": page2_blocks,
        },
    ]

    from modifai.core.text_extraction import extract_text_from_s3
    text = extract_text_from_s3("my-bucket", "docs/long.pdf", poll_interval_seconds=1)

    assert "Page 1 content." in text
    assert "Page 2 content." in text
