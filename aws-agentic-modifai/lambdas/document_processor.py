import json
import boto3
import os
import uuid
import io
import PyPDF2

from llm_helper import call_llm

s3 = boto3.client('s3')


# ── Text extraction helpers ───────────────────────────────────────────────────

def extract_pdf_text(bucket: str, key: str) -> str:
    file_bytes = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_plain_text(bucket: str, key: str) -> str:
    file_bytes = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    return file_bytes.decode('utf-8', errors='ignore')


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, target_words: int = 384, overlap_words: int = 48) -> list:
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + target_words])
        if len(chunk.split()) >= 20:   # skip tiny trailing fragments
            chunks.append(chunk)
        i += (target_words - overlap_words)
    return chunks


# ── LLM-powered semantic chunking strategy ────────────────────────────────────

def get_chunking_strategy(strategy_event: dict) -> dict:
    """
    Use the strategy passed from IntentAnalyzer.
    Falls back to the LLM if chunking params are missing.
    """
    chunking = strategy_event.get("chunking", {})
    max_tokens = chunking.get("max_tokens")
    overlap    = chunking.get("overlap")

    if max_tokens and overlap:
        return {"max_tokens": int(max_tokens), "overlap": int(overlap)}

    # Fallback: ask LLM for a sensible chunking strategy
    try:
        intent = strategy_event.get("intent", "QA")
        prompt = (
            f"For a '{intent}' fine-tuning dataset, recommend chunking parameters. "
            "Return ONLY JSON: {\"max_tokens\": <int>, \"overlap\": <int>}. "
            "max_tokens should be 256-1024, overlap should be 32-128."
        )
        raw = call_llm(prompt=prompt)
        raw = raw.strip().strip("`").replace("json\n", "").strip()
        result = json.loads(raw)
        return {
            "max_tokens": int(result.get("max_tokens", 512)),
            "overlap":    int(result.get("overlap", 64))
        }
    except Exception as e:
        print(f"LLM chunking strategy failed, using defaults: {e}")
        return {"max_tokens": 512, "overlap": 64}


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    strategy_wrapper = event.get('strategy', {})
    # IntentAnalyzer wraps its output inside $.strategy.strategy
    strategy = strategy_wrapper.get('strategy', strategy_wrapper)
    document_uris = event.get('document_s3_uris', [])

    if not document_uris:
        raise ValueError("Missing document_s3_uris in event")

    bucket = document_uris[0].split("/")[2]

    # ── 1. Extract text from all documents ────────────────────────────────────
    all_text = []
    for uri in document_uris:
        key = "/".join(uri.split("/")[3:])
        ext = key.lower().rsplit('.', 1)[-1]
        try:
            if ext == 'pdf':
                text = extract_pdf_text(bucket, key)
            elif ext in ('txt', 'md', 'csv'):
                text = extract_plain_text(bucket, key)
            else:
                print(f"Skipping unsupported file type: {ext} ({uri})")
                continue
            all_text.append(text)
        except Exception as e:
            print(f"Failed to process {uri}: {e}")

    combined_text = "\n\n---\n\n".join(all_text)
    if not combined_text.strip():
        raise ValueError("No text could be extracted from any document")

    # ── 2. Determine chunking parameters ──────────────────────────────────────
    params       = get_chunking_strategy(strategy)
    max_tokens   = params["max_tokens"]
    overlap      = params["overlap"]
    target_words = int(max_tokens * 0.75)
    overlap_words = int(overlap * 0.75)

    chunks = chunk_text(combined_text, target_words, overlap_words)
    print(f"Created {len(chunks)} chunks (max_tokens={max_tokens}, overlap={overlap})")

    # ── 3. Upload chunks to S3 for Map State ──────────────────────────────────
    run_id     = str(uuid.uuid4())[:8]
    chunk_uris = []

    for i, chunk_data in enumerate(chunks):
        chunk_key = f"modifai-jobs/{run_id}/chunks/chunk_{i}.json"
        s3.put_object(
            Bucket=bucket,
            Key=chunk_key,
            Body=json.dumps({"chunk_id": i, "text": chunk_data})
        )
        chunk_uris.append(f"s3://{bucket}/{chunk_key}")

    return {
        "chunk_uris":   chunk_uris,
        "total_chunks": len(chunk_uris),
        "run_id":       run_id,
        "bucket":       bucket
    }
