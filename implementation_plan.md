# Modifai — Full MLOps Platform Implementation Plan
### Extending from Dataset Generator → Production Fine-Tuned Models

---

## The Honest Gap Analysis

**What's built today (P1 — done ✅):**
```
PDF text (manual) → Agentic Dataset Generator → clean_dataset.jsonl
```
The 3-agent pipeline (Orchestrator → Critic → Curriculum) generates high-quality
**synthetic training data** and saves it as a JSONL file. That's it.

**What the full Modifai vision requires:**
```
PDF upload → OCR → Chunking → Agentic Dataset Gen → Quality Control
         → Fine-Tune on SageMaker → Deploy Endpoint → Live Inference API
```

Everything from "Fine-Tune on SageMaker" onwards is **not built yet.**

---

## What Needs to Be Built

### Missing Layers (in order of pipeline flow)

| # | Layer | Status | What it does |
|---|---|---|---|
| 1 | **PDF Ingestion + OCR** | ❌ Not built | AWS Textract extracts text from uploaded PDFs |
| 2 | **Chunking** | ❌ Not built | Splits extracted text into ~512-token chunks |
| 3 | **Agentic Dataset Gen** | ✅ **Done** | Generates + filters training samples |
| 4 | **Dataset Formatting** | ❌ Not built | Converts JSONL → SageMaker-compatible format |
| 5 | **S3 Upload** | ❌ Not built | Pushes dataset to S3 bucket for SageMaker |
| 6 | **SageMaker Fine-Tuning** | ❌ Not built | Runs fine-tuning job on base LLM |
| 7 | **Model Deployment** | ❌ Not built | Deploys fine-tuned model to inference endpoint |
| 8 | **Inference API** | ❌ Not built | REST endpoint teammates can call to test the model |
| 9 | **Orchestration (Step Functions)** | ❌ Not built | Ties all steps into a serverless workflow |
| 10 | **Frontend / P3 Dashboard** | ⚠️ Partial | Shows agent events; needs upload UI + status polling |

---

## User Review Required

> [!IMPORTANT]
> **SageMaker fine-tuning costs real money.** A single fine-tuning job on a small model
> (e.g., Llama 3 8B) costs ~$15–40 on SageMaker `ml.g5.2xlarge` instances. We need to
> decide: do we demo with a real fine-tune, or use a cheaper alternative?

> [!WARNING]
> **The deadline is 14 June 2026.** Fine-tuning a model end-to-end (including dataset prep,
> SageMaker job, and deployment) takes ~2–6 hours for a small model. We need to start
> this immediately if we want a live demo.

> [!CAUTION]
> **Hackathon alternative to full fine-tuning:** If cost/time is a concern, we can use
> AWS Bedrock's **Custom Model Import** or **Model Customisation (fine-tune via Bedrock)**
> instead of SageMaker. Bedrock fine-tuning is simpler to set up and directly compatible
> with our existing Bedrock stack. Recommend confirming this decision before implementation.

---

## Open Questions

> [!IMPORTANT]
> 1. **Base model choice:** Which model do you want to fine-tune? Options:
>    - `amazon.titan-text-express-v1` — fine-tunable via Bedrock, cheapest
>    - `meta.llama3-8b-instruct-v1` — via SageMaker JumpStart, better quality
>    - `mistral.mistral-7b-instruct-v0:2` — via SageMaker, very popular for fine-tuning
>
> 2. **SageMaker vs Bedrock fine-tuning?**
>    - SageMaker = more control, more cost, more setup time
>    - Bedrock Custom Model = simpler, integrates directly with existing code, but limited model choice
>
> 3. **Who owns which new layer?** (P1 = you, P2 = infra teammate, P3 = frontend teammate)
>
> 4. **Do you have a SageMaker execution role ARN?** Fine-tuning jobs require one.

---

## Proposed Architecture

```
User uploads PDF
       │
       ▼
[Layer 1] S3 Bucket
   PDF stored at s3://modifai-datasets/{job_id}/input.pdf
       │
       ▼
[Layer 2] AWS Textract (OCR)
   Extracts raw text + structure from PDF
       │
       ▼
[Layer 3] Chunker (modifai/core/chunking.py — new)
   Splits text into ~512-token chunks with overlap
       │
       ▼
[Layer 4] ✅ EXISTING: Agentic Pipeline (run_agentic_loop)
   Orchestrator → Generator → Critic → Curriculum → clean dataset
       │
       ▼
[Layer 5] Dataset Formatter (modifai/core/formatter.py — new)
   Converts JSONL → SageMaker/Bedrock fine-tuning format
   Uploads to s3://modifai-datasets/{job_id}/training_data.jsonl
       │
       ▼
[Layer 6] Fine-Tuning Job (modifai/core/finetuning.py — new)
   Starts SageMaker Training Job (or Bedrock Custom Model job)
   Monitors status via polling
       │
       ▼
[Layer 7] Endpoint Deployment (modifai/core/deployment.py — new)
   Creates SageMaker endpoint from trained model artifact
   Returns inference endpoint URL
       │
       ▼
[Layer 8] Inference API (modifai/core/inference.py — new)
   Wrapper to call the deployed endpoint
   Returns: {response: str, model_id: str, latency_ms: int}
       │
       ▼
[Layer 9] Step Functions Orchestration (infra/ — P2's layer)
   Ties all Lambda functions into a serverless state machine
       │
       ▼
[Layer 10] P3 Dashboard
   Shows live status: OCR → Dataset Gen → Fine-tuning → Deployed
```

---

## Proposed Changes

### Layer 2 — OCR (new file)

#### [NEW] modifai/core/text_extraction.py
- Uses `boto3` Textract `start_document_text_detection` (async) for PDFs
- Waits for job completion, concatenates page text in order
- Returns raw text string + page metadata

---

### Layer 3 — Chunking (new file)

#### [NEW] modifai/core/chunking.py
- Simple sliding window chunker: 512 tokens per chunk, 64 token overlap
- No new dependencies — uses word count approximation (1 word ≈ 1.3 tokens)
- Returns `List[str]`

---

### Layer 5 — Dataset Formatting + S3 Upload (new file)

#### [NEW] modifai/core/formatter.py
- Converts `List[dict]` from `run_agentic_loop` into fine-tuning JSONL format
- Two target formats:
  - **Bedrock fine-tuning format:** `{"prompt": "...", "completion": "..."}`
  - **SageMaker/HuggingFace format:** `{"instruction": "...", "input": "...", "output": "..."}`
- Uploads formatted file to S3

---

### Layer 6 — Fine-Tuning (new file)

#### [NEW] modifai/core/finetuning.py

**Option A: Bedrock Custom Model (recommended for hackathon)**
```python
bedrock.create_model_customization_job(
    jobName=job_name,
    baseModelIdentifier="amazon.titan-text-express-v1",
    customModelName=custom_model_name,
    trainingDataConfig={"s3Uri": s3_training_data_uri},
    outputDataConfig={"s3Uri": s3_output_uri},
    hyperParameters={"epochCount": "3", "batchSize": "8", "learningRate": "0.00005"},
)
```

**Option B: SageMaker JumpStart (better quality, more cost)**
```python
from sagemaker.jumpstart.estimator import JumpStartEstimator
estimator = JumpStartEstimator(model_id="meta-textgeneration-llama-3-8b")
estimator.set_hyperparameters(instruction_tuned="True", epoch="3")
estimator.fit({"training": training_data_uri})
```

---

### Layer 7 — Deployment (new file)

#### [NEW] modifai/core/deployment.py
- For Bedrock: provisions the custom model, returns model ARN
- For SageMaker: calls `estimator.deploy()`, returns endpoint name
- Polls status until deployment is complete (with timeout)

---

### Layer 8 — Inference Wrapper (new file)

#### [NEW] modifai/core/inference.py
- Calls the deployed endpoint with a user query
- Works with both Bedrock and SageMaker endpoints
- Returns `{response: str, tokens_used: int, latency_ms: int}`

---

### Layer 9 — Full Pipeline Orchestrator (new file)

#### [NEW] modifai/core/full_pipeline.py
```python
def run_full_pipeline(pdf_path: str, goal: str, base_model: str) -> dict:
    # 1. OCR
    text = extract_text(pdf_path)
    # 2. Chunk
    chunks = chunk_text(text)
    # 3. Agentic dataset gen (EXISTING)
    state = run_agentic_loop(goal=goal, chunks=chunks, ...)
    # 4. Format + upload to S3
    s3_uri = format_and_upload(state["final_samples"])
    # 5. Fine-tune
    job_name = start_finetuning_job(base_model, s3_uri)
    # 6. Deploy
    endpoint = deploy_model(job_name)
    # 7. Return
    return {"endpoint": endpoint, "samples_used": len(state["final_samples"]), ...}
```

---

### Tests

#### [NEW] modifai/core/tests/test_finetuning.py
- Mock SageMaker/Bedrock calls
- Verify job creation params, status polling, error handling

#### [NEW] modifai/core/tests/test_full_pipeline.py
- Full E2E mock test: PDF path → endpoint name
- Verify each layer is called in correct order

---

## Verification Plan

### Automated Tests
```bash
python -m pytest modifai/ -v
# All existing 31 tests must still pass
# New fine-tuning + pipeline tests added (target: ~45 total)
```

### Manual Verification
1. Run `smoke_test.py` — confirms existing dataset gen still works (no regression)
2. Run a real fine-tuning job with a 3-chunk test doc (costs ~$5–15)
3. Call the deployed endpoint with a test question, verify it answers from the doc

### Demo Script (for judges)
```bash
python demo.py \
  --pdf examples/sample_policy.pdf \
  --goal "Build a Q&A bot for this HR policy" \
  --model amazon.titan-text-express-v1
```
Expected: Shows live progress → prints endpoint URL → answers a test question
using the fine-tuned model.

---

## Recommended Build Order (given deadline: 14 June)

| Priority | Task | Time estimate | Owner |
|---|---|---|---|
| HIGH | `text_extraction.py` (OCR) | 2 hours | P1 (you) |
| HIGH | `chunking.py` | 1 hour | P1 (you) |
| HIGH | `formatter.py` + S3 upload | 2 hours | P1 (you) |
| HIGH | `finetuning.py` — Bedrock option | 3 hours | P1 (you) |
| MED | `deployment.py` | 2 hours | P2 (infra teammate) |
| MED | `inference.py` | 1 hour | P2 (infra teammate) |
| MED | `full_pipeline.py` | 2 hours | P1 (you) |
| LOW | Step Functions wiring | 4 hours | P2 (infra teammate) |
| LOW | Dashboard upload UI + status | 3 hours | P3 (frontend teammate) |

**Total estimated time: ~20 hours of parallel work**

If the team works in parallel (P1 on OCR+dataset, P2 on deployment, P3 on dashboard),
this is achievable before the 14 June deadline.
