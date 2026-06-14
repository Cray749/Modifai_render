"""
Dataset service — CRUD operations on the generated training dataset.

The dataset lives as a JSONL file in S3 at:
  s3://{bucket}/projects/{project_id}/dataset/clean_dataset.jsonl
"""

import logging

from app.services import s3_service

logger = logging.getLogger(__name__)


def get_dataset(project_id: str) -> dict:
    """
    Read the full dataset from S3.

    Returns:
        { "dataset": [{ "instruction": str, "response": str, "confidence": float|None }], "total": int }
    """
    examples = s3_service.get_dataset_jsonl(project_id)

    dataset = []
    for ex in examples:
        dataset.append({
            "instruction": ex.get("instruction", ex.get("input", "")),
            "response": ex.get("response", ex.get("output", "")),
            "confidence": ex.get("confidence", ex.get("score")),
        })

    return {"dataset": dataset, "total": len(dataset)}


def search_dataset(project_id: str, query: str) -> dict:
    """
    Search within dataset examples (case-insensitive substring match).

    Returns:
        { "results": [...], "total": int }
    """
    data = get_dataset(project_id)
    q = query.lower()

    results = [
        ex for ex in data["dataset"]
        if q in ex["instruction"].lower() or q in ex["response"].lower()
    ]

    return {"results": results, "total": len(results)}


def update_example(project_id: str, index: int, instruction: str, response: str) -> None:
    """
    Update a single example by index. Rewrites the full JSONL to S3.
    """
    examples = s3_service.get_dataset_jsonl(project_id)

    if index < 0 or index >= len(examples):
        raise IndexError(f"Example index {index} out of range (0-{len(examples)-1})")

    examples[index]["instruction"] = instruction
    if "input" in examples[index]:
        examples[index]["input"] = instruction
    examples[index]["response"] = response
    if "output" in examples[index]:
        examples[index]["output"] = response

    s3_service.put_dataset_jsonl(project_id, examples)
    logger.info("Updated example %d in project %s", index, project_id)


def delete_example(project_id: str, index: int) -> None:
    """
    Delete a single example by index. Rewrites the full JSONL to S3.
    """
    examples = s3_service.get_dataset_jsonl(project_id)

    if index < 0 or index >= len(examples):
        raise IndexError(f"Example index {index} out of range (0-{len(examples)-1})")

    examples.pop(index)
    s3_service.put_dataset_jsonl(project_id, examples)
    logger.info("Deleted example %d from project %s", index, project_id)


def export_dataset(project_id: str) -> dict:
    """
    Generate a presigned download URL for the dataset JSONL file.

    Returns:
        { "download_url": str }
    """
    s3_key = f"projects/{project_id}/dataset/clean_dataset.jsonl"
    url = s3_service.generate_presigned_download_url(s3_key)
    return {"download_url": url}
