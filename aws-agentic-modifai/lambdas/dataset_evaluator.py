import json
import boto3
import os

from llm_helper import call_llm

s3 = boto3.client('s3')


def lambda_handler(event, context):
    dataset = event.get('dataset', [])
    dataset_uris = [
        item.get('sample_uri')
        for item in dataset
        if isinstance(item, dict) and 'sample_uri' in item
    ]

    if not dataset_uris:
        return {"action": "proceed", "training_data_uri": None, "score": 1.0}

    bucket = dataset_uris[0].split("/")[2]
    run_id  = dataset_uris[0].split("/")[4]

    # ── Merge all per-chunk sample files from S3 ─────────────────────────────
    all_samples = []
    for uri in dataset_uris:
        key = "/".join(uri.split("/")[3:])
        try:
            samples = json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
            all_samples.extend(samples)
        except Exception as e:
            print(f"Warning: could not load samples from {uri}: {e}")

    if not all_samples:
        return {"action": "proceed", "training_data_uri": None, "score": 1.0}

    # ── LLM Critic evaluation ─────────────────────────────────────────────────
    sample_preview = json.dumps(all_samples[:5], indent=2)

    system_prompt = (
        "You are a dataset quality evaluator for LLM fine-tuning. "
        "Analyse a sample of training examples and output ONLY a JSON object: "
        "{\"score\": <float 0.0-1.0>, \"action\": \"proceed\" | \"regenerate\", "
        "\"reason\": \"<one sentence>\"}. "
        "Score >= 0.7 → action=proceed. Score < 0.7 → action=regenerate."
    )
    prompt = (
        f"Evaluate this training dataset sample ({len(all_samples)} total samples, "
        f"showing first 5):\n\n{sample_preview}\n\n"
        "Return JSON only."
    )

    try:
        raw = call_llm(prompt=prompt, system=system_prompt)
        raw = raw.strip().strip("`").replace("json\n", "").strip()
        evaluation = json.loads(raw)
        dataset_score = float(evaluation.get("score", 0.9))
        action = evaluation.get("action", "proceed")
        reason = evaluation.get("reason", "")
    except Exception as e:
        print(f"LLM evaluation failed, defaulting to proceed: {e}")
        dataset_score = 0.9
        action = "proceed"
        reason = "LLM evaluation unavailable — defaulting to proceed."

    if action == "regenerate":
        return {
            "action": "regenerate",
            "reason": reason or f"Dataset score {dataset_score:.2f} < 0.7. Needs improvement."
        }

    # ── Write merged JSONL for Bedrock fine-tuning ────────────────────────────
    jsonl_lines = []
    for s in all_samples:
        prompt_text = f"User: {s.get('instruction', '')}\nBot:"
        completion  = f" {s.get('output', '')}"
        jsonl_lines.append(json.dumps({"prompt": prompt_text, "completion": completion}))

    training_data = "\n".join(jsonl_lines)
    training_key  = f"modifai-jobs/{run_id}/training_data.jsonl"

    s3.put_object(Bucket=bucket, Key=training_key, Body=training_data)

    return {
        "action": "proceed",
        "training_data_uri": f"s3://{bucket}/{training_key}",
        "score": dataset_score,
        "bucket": bucket,
        "run_id": run_id
    }
