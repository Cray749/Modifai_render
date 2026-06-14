# Modifai — Automated LLM Fine-Tuning Pipeline

> **Hackathon project** · Automatically generates high-quality fine-tuning datasets from your documents using a self-improving multi-agent loop on AWS — powered by **OpenRouter**.

---

## What does Modifai do?

You upload documents to S3. The pipeline automatically:
1. **Analyses your documents** — detects intent (Q&A, summarisation, instruction-following)
2. **Chunks and processes** them into optimal segments
3. **Generates synthetic training samples** (question-answer pairs, instruction-output pairs, etc.)
4. **Evaluates dataset quality** using an LLM critic
5. **Validates and triggers** a fine-tuning job, writing a manifest to S3
6. **Monitors job status**, evaluates the trained model, and tunes hyperparameters if needed
7. **Deploys** the final model via Bedrock provisioned throughput

No manual labelling. No prompt engineering. The agents do it themselves.

---

## Architecture — The 9 Lambda Functions

All AI inference is routed through **`llm_helper.py`** using the [OpenRouter](https://openrouter.ai) API.

```
Document Upload (S3)
        │
        ▼
┌────────────────────┐
│  intent_analyzer   │  ← Analyses docs, recommends chunking + hyperparameters
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ document_processor │  ← Extracts text, chunks docs, uploads chunk JSONs to S3
└────────┬───────────┘
         │ (Map State — parallel per chunk)
         ▼
┌────────────────────┐
│ dataset_generator  │  ← Generates Q&A / instruction samples from each chunk
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ dataset_evaluator  │  ← LLM critic scores dataset quality; proceed or regenerate
└────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│ fine_tuning_trigger │  ← Validates config via LLM; writes job manifest to S3
└────────┬────────────┘
         │
         ▼
┌────────────────────┐
│  status_checker    │  ← Polls job manifest / simulates status in DEMO_MODE
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  model_evaluator   │  ← LLM critic estimates generalisation score from loss
└────────┬───────────┘
         │
         ▼
┌──────────────────────┐
│ hyperparameter_tuner │  ← deploy / tune / max_attempts_reached decision
└────────┬─────────────┘
         │ (if score < threshold → loop back to fine_tuning_trigger)
         ▼
┌────────────────────┐
│     deployer       │  ← Provisions model via Bedrock (no LLM calls)
└────────────────────┘
```

---

## LLM Provider — OpenRouter

All Lambda functions call `llm_helper.py` which POSTs to the **OpenRouter** `/chat/completions` endpoint (OpenAI-compatible).

| Setting | Value |
|---|---|
| Primary model | `deepseek/deepseek-chat-v3` |
| Fallback 1 | `qwen/qwen3-235b-a22b` |
| Fallback 2 | `google/gemini-2.5-flash-lite` |
| Retries | 3 per model, exponential back-off |
| Secret | `modifai/or` in AWS Secrets Manager (`{"api_key": "sk-or-v1-..."}`) |

Models are tried in order on 429 (rate limit) or 5xx errors — the pipeline **never crashes** due to a single model being unavailable.

---

## Project Structure

```
modifai/
├── 5 lambda/
│   ├── llm_helper.py           # Shared OpenRouter client (all LLM calls go here)
│   ├── intent_analyzer.py      # Lambda 1 — document intent + strategy
│   ├── document_processor.py   # Lambda 2 — text extraction + chunking
│   ├── dataset_generator.py    # Lambda 3 — sample generation (Map State)
│   ├── dataset_evaluator.py    # Lambda 4 — dataset quality critic
│   ├── fine_tuning_trigger.py  # Lambda 5 — config validation + job manifest
│   ├── status_checker.py       # Lambda 6 — job status polling / simulation
│   ├── model_evaluator.py      # Lambda 7 — model quality estimation
│   ├── hyperparameter_tuner.py # Lambda 8 — deploy / tune decision
│   ├── deployer.py             # Lambda 9 — Bedrock provisioned throughput (no LLM)
│   ├── requirements.txt        # requests, boto3, PyPDF2
│   └── template.yaml           # AWS SAM template
├── core/
│   ├── critic_agent.py         # Critic LLM evaluation functions
│   ├── dataset_generation.py   # Sample generator
│   └── utils.py                # Logger helper
└── agents/
    ├── schemas.py              # All shared TypedDicts (locked — don't rename keys)
    ├── orchestrator.py         # OrchestratorAgent class
    ├── critic.py               # CriticAgent adapter class
    ├── curriculum.py           # CurriculumAgent class
    ├── logging_utils.py        # AgentEventLogger (writes JSONL for dashboard)
    ├── pipeline_loop.py        # run_agentic_loop() — wires everything together
    └── tests/
        ├── test_orchestrator.py
        ├── test_curriculum.py
        └── test_pipeline_e2e.py
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- AWS credentials with access to S3, Secrets Manager, and Bedrock (`ap-south-1`)
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- OpenRouter API key stored in Secrets Manager as `modifai/or`

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd <repo>
pip install boto3 requests PyPDF2 pytest
```

### 2. Configure AWS credentials

**On Windows (PowerShell):**
```powershell
$env:AWS_ACCESS_KEY_ID="your_access_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key"
$env:AWS_REGION="ap-south-1"
```

**On Mac/Linux:**
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="ap-south-1"
```

### 3. Store the OpenRouter API key in Secrets Manager

The secret must already exist at `modifai/or` with this exact JSON structure:
```json
{"api_key": "sk-or-v1-..."}
```

For local testing only, you can bypass Secrets Manager with:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 4. Deploy with SAM

```bash
cd "5 lambda"
sam build
sam deploy --guided
# Environment: dev | staging | prod
```

### 5. Run the unit tests (no AWS needed)

```bash
python -m pytest modifai/agents/tests/ -v
```

---

## Environment Variables

All Lambda functions inherit these globals from `template.yaml`:

| Variable | Default | Description |
|---|---|---|
| `OR_SECRET_NAME` | `modifai/or` | Secrets Manager secret for OpenRouter key |
| `OR_MODEL` | `deepseek/deepseek-chat-v3` | Primary LLM model |
| `OR_MAX_RETRIES` | `3` | Retry attempts per model before falling back |
| `OR_RETRY_DELAY` | `1.5` | Base retry delay in seconds (exponential back-off) |
| `S3_BUCKET` | _(auto)_ | Pipeline bucket for chunks, samples, manifests |
| `JOB_MANIFEST_PREFIX` | `modifai-jobs` | S3 prefix for job manifests |
| `DEMO_MODE` | `false` | `true` = simulate job status via LLM (no real training backend) |
| `QUALITY_THRESHOLD` | `0.85` | Minimum weighted score to approve deployment |
| `MAX_TUNING_ATTEMPTS` | `3` | Max hyperparameter tuning iterations |
| `BASE_MODEL` | `meta.llama3-8b-instruct-v1:0` | Bedrock base model for fine-tuning |

---

## For Teammates — How to Contribute

### Branch naming convention
```
feature/<your-name>/<what-youre-adding>
# Examples:
feature/riya/pdf-chunking
feature/arjun/dashboard
feature/lakshya/openrouter-migration
```

### Before pushing
```bash
# Always run this first — must be 31/31
python -m pytest modifai/agents/tests/ -v
```

### What NOT to change without a team sync
- Field names in `modifai/agents/schemas.py` — dashboard depends on these exact keys
- `modifai/core/critic_agent.py` — the critic implementation is locked (owned by teammate)
- The `run_agentic_loop()` function signature — other services call this
- The `llm_helper.py` public API (`call_llm`, `call_llm_json`) — all 8 Lambdas depend on it

---

## Key Design Decisions

| Decision | Why |
|---|---|
| OpenRouter instead of Gemini | Gemini free tier hit 429 rate limits; OpenRouter gives multi-model fallback flexibility |
| Model fallback chain | Ensures pipeline never crashes from a single model being rate-limited |
| `llm_helper.py` as single LLM entry point | One place to swap providers, add logging, or change retry logic |
| `requests` instead of Gemini SDK | OpenRouter is OpenAI-compatible — no heavyweight SDK needed |
| Secrets Manager for API key | Key never touches env vars in production; IAM controls access |
| `DEMO_MODE` | Lets the pipeline run end-to-end without a real training backend |

---

## Team

| Person | Component |
|---|---|
| Lakshya | Orchestrator Agent, Pipeline Loop, Lambda Integration, OpenRouter Migration |
| [Teammate] | Critic Agent (core evaluation logic) |
| [Teammate] | Curriculum Agent |
| [Teammate] | Dashboard / Frontend |

---

*Built for the FAR AWAY Hackathon · June 2026*
