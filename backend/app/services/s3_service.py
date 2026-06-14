"""
S3 service — presigned URLs, file read/write, cleanup.
"""

import json
import logging

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=settings.AWS_REGION)
    return _client


# ── Presigned URLs ──────────────────────────────────────────────────────────────

def generate_presigned_upload_url(project_id: str, filename: str, content_type: str = "application/octet-stream") -> dict:
    """
    Generate a presigned PUT URL for uploading a file to:
      s3://{bucket}/projects/{project_id}/raw/{filename}

    Returns: { "presigned_url": str, "s3_key": str }
    """
    s3_key = f"projects/{project_id}/raw/{filename}"
    try:
        params = {
            "Bucket": settings.S3_BUCKET,
            "Key": s3_key,
            "ContentType": content_type,
        }
        url = _get_client().generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=3600,  # 1 hour
        )
        return {"presigned_url": url, "s3_key": s3_key}
    except ClientError as e:
        logger.error("Failed to generate presigned upload URL: %s", e)
        raise


def generate_presigned_download_url(s3_key: str) -> str:
    """Generate a presigned GET URL for downloading a file."""
    try:
        url = _get_client().generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": settings.S3_BUCKET,
                "Key": s3_key,
            },
            ExpiresIn=3600,
        )
        return url
    except ClientError as e:
        logger.error("Failed to generate presigned download URL: %s", e)
        raise


# ── File Operations ─────────────────────────────────────────────────────────────

def delete_project_files(project_id: str) -> int:
    """
    Delete all objects under s3://{bucket}/projects/{project_id}/.
    Returns the number of objects deleted.
    """
    prefix = f"projects/{project_id}/"
    client = _get_client()
    deleted = 0

    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
            objects = page.get("Contents", [])
            if not objects:
                continue

            delete_keys = [{"Key": obj["Key"]} for obj in objects]
            client.delete_objects(
                Bucket=settings.S3_BUCKET,
                Delete={"Objects": delete_keys},
            )
            deleted += len(delete_keys)
    except ClientError as e:
        logger.error("Failed to delete project files: %s", e)
        raise

    logger.info("Deleted %d objects for project %s", deleted, project_id)
    return deleted


def get_dataset_jsonl(project_id: str) -> list[dict]:
    """
    Read the clean_dataset.jsonl from S3.
    Returns a list of parsed JSON objects (one per line).
    """
    s3_key = f"projects/{project_id}/dataset/clean_dataset.jsonl"
    try:
        response = _get_client().get_object(
            Bucket=settings.S3_BUCKET,
            Key=s3_key,
        )
        content = response["Body"].read().decode("utf-8")
        examples = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                examples.append(json.loads(line))
        return examples
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("No dataset file found for project %s", project_id)
            return []
        logger.error("Failed to read dataset: %s", e)
        raise


def put_dataset_jsonl(project_id: str, dataset: list[dict]) -> str:
    """
    Write the dataset list back to S3 as JSONL.
    Returns the S3 key.
    """
    s3_key = f"projects/{project_id}/dataset/clean_dataset.jsonl"
    content = "\n".join(json.dumps(example, ensure_ascii=False) for example in dataset)

    try:
        _get_client().put_object(
            Bucket=settings.S3_BUCKET,
            Key=s3_key,
            Body=content.encode("utf-8"),
            ContentType="application/jsonl",
        )
        return s3_key
    except ClientError as e:
        logger.error("Failed to write dataset: %s", e)
        raise


def check_file_exists(s3_key: str) -> bool:
    """Check if a file exists in S3."""
    try:
        _get_client().head_object(Bucket=settings.S3_BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False
