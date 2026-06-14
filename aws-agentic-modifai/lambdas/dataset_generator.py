import json
import boto3
import os
import time
import re
from botocore.exceptions import ClientError

from llm_helper import call_llm

s3 = boto3.client('s3')

SYSTEM_PROMPT = (
    "You are a training data generator for LLM fine-tuning. "
    "Given a source text chunk, generate question-answer pairs grounded entirely in that chunk. "
    "Return ONLY a valid JSON array of objects with fields: "
    "\"instruction\" (the question/task), \"input\" (empty string), \"output\" (the answer). "
    "No markdown, no preamble, no trailing text."
)


def lambda_handler(event, context):
    chunk_uri = event
    bucket = chunk_uri.split("/")[2]
    key    = "/".join(chunk_uri.split("/")[3:])

    # ── 1. Download chunk ─────────────────────────────────────────────────────
    chunk_data = json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
    chunk_text = chunk_data.get("text", "")
    chunk_id   = chunk_data.get("chunk_id", 0)

    samples_per_chunk = int(os.environ.get("SAMPLES_PER_CHUNK", "4"))

    # ── 2. LLM generation with retry ─────────────────────────────────────────
    prompt = (
        f"SOURCE CHUNK (chunk_id={chunk_id}):\n{chunk_text}\n\n"
        f"Generate exactly {samples_per_chunk} training samples from this chunk. "
        "Return a JSON array only."
    )

    samples = []
    for attempt in range(5):
        try:
            raw = call_llm(prompt=prompt, system=SYSTEM_PROMPT)
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
            bracket = raw.find("[")
            if bracket != -1:
                raw = raw[bracket:]
            samples = json.loads(raw)
            if not isinstance(samples, list):
                raise ValueError("Response was not a JSON array")
            print(f"Generated {len(samples)} samples for chunk {chunk_id} (attempt {attempt+1})")
            break
        except Exception as e:
            wait = (2 ** attempt) * 2   # 2, 4, 8, 16, 32 seconds
            print(f"Generation attempt {attempt+1}/5 failed: {e} — retrying in {wait}s")
            time.sleep(wait)
    else:
        print(f"All 5 attempts exhausted for chunk {chunk_id}.")
        samples = []

    # ── 3. Normalise and stamp chunk_id ──────────────────────────────────────
    valid = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        s.setdefault("instruction", "")
        s.setdefault("input", "")
        s.setdefault("output", "")
        s["chunk_id"] = chunk_id
        valid.append(s)

    # ── 4. Upload to S3 ───────────────────────────────────────────────────────
    sample_key = key.replace("chunks", "samples")
    s3.put_object(
        Bucket=bucket,
        Key=sample_key,
        Body=json.dumps(valid)
    )

    return {
        "sample_uri":   f"s3://{bucket}/{sample_key}",
        "sample_count": len(valid)
    }
