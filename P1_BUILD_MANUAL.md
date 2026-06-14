# P1 Build Manual — Modifai Agentic Pipeline
### FAR AWAY Hackathon · Deadline: 14 June 2026
### Prepared for P1 (Agent Core). Critic agent is already done.

---

## ⚡ HOW TO USE THIS DOCUMENT

Three sessions run in **parallel** — give each one to a separate Claude account (or tab):

| Session | What it builds | Can start | Blocks |
|---------|---------------|-----------|--------|
| **Session A** | Orchestrator Agent + schemas | NOW | Session C |
| **Session B** | Curriculum Agent | NOW (needs critic schema only) | Session C |
| **Session C** | Pipeline wiring, logging, E2E tests | After A + B done | Submission |

Copy the **"CONTEXT BLOCK"** at the start of every session prompt. It contains the
project summary and all schema contracts so each Claude doesn't need to ask questions.

**Paste order per Claude account:**
1. Context Block (below)
2. The session section (A, B, or C)
3. "Here is the existing critic code: [paste critic code]" ← only needed for Session C

---

## CONTEXT BLOCK — paste this first in EVERY session

```
PROJECT: Modifai — serverless LLM fine-tuning platform on AWS.
A user uploads a PDF → Modifai extracts text, chunks it, generates synthetic
training pairs via Amazon Bedrock, filters them for quality, then fine-tunes
a model on SageMaker.

TECH STACK:
- Python 3.9+
- AWS Bedrock (model: amazon.nova-micro-v1:0, region: ap-south-1)
- boto3 Bedrock converse API (NOT invoke_model — use converse for tool-use)
- Existing code lives under modifai/core/ (text_extraction, chunking, dataset_generation, quality_control, scoring)
- modifai/config/settings.py has PipelineConfig dataclass
- All agent code goes in a NEW folder: modifai/agents/

EXISTING PIPELINE FLOW (before agentic pivot):
PDF → OCR (Textract) → Chunking → DatasetGeneration (Bedrock) → QualityControl → JSONL

NEW AGENTIC FLOW WE ARE BUILDING:
goal + doc_metadata
    → OrchestratorAgent (decides intent/threshold/samples_per_chunk)
    → DatasetGeneration (generates samples using strategy)
    → CriticAgent (scores samples, returns accept/rewrite/reject per sample)
    → if accept% < threshold AND iteration < 3:
          CurriculumAgent (analyses rejections → targeted prompt)
          → DatasetGeneration (regenerates with targeted prompt)
          → CriticAgent again
          → loop
    → final accepted samples → clean_dataset.jsonl

CRITIC AGENT IS ALREADY DONE. Its Python class interface is:
    class CriticAgent:
        def run_single(self, sample: dict) -> dict
            # sample = {"instruction": str, "input": str, "output": str}
            # returns = {"verdict": "accept"|"rewrite"|"reject", "reason": str, "rewritten_output": str|None}
        
        def run_batch(self, samples: list[dict]) -> dict
            # returns = {
            #   "verdicts": list[dict],   # one per sample, same shape as run_single
            #   "stats": {
            #     "total": int, "accepted": int, "rewritten": int,
            #     "rejected": int, "accept_pct": float
            #   }
            # }

LOCKED SHARED SCHEMAS (do NOT change these — P2 and P3 depend on them):

# 1. OrchestratorInput
{
    "goal": str,            # e.g. "Generate a fine-tuning dataset for customer support Q&A"
    "doc_metadata": {
        "filename": str,
        "page_count": int,
        "domain": str,                  # e.g. "customer support", "HR policy"
        "estimated_chunk_count": int
    }
}

# 2. OrchestratorOutput  ← strategy JSON — P2 wires this into Step Functions
{
    "intent": str,              # "QA" | "instruction" | "tutor"
    "quality_threshold": float, # 0.5–0.95
    "samples_per_chunk": int,   # 3–8
    "reasoning": str            # brief explanation for logs/UI
}

# 3. CurriculumInput
{
    "rejection_reasons": list[str],  # reason strings from Critic run_batch verdicts
    "strategy": dict,                # the OrchestratorOutput dict
    "iteration": int                 # 1-based loop iteration number
}

# 4. CurriculumOutput
{
    "gap_categories": [              # ≥3 items required
        {
            "name": str,             # e.g. "lacks_step_by_step_reasoning"
            "description": str,
            "example_bad": str,
            "example_good": str
        }
    ],
    "targeted_prompt": str,          # injected into DatasetGeneration as custom_prompt
    "priority_focus": str            # the single most critical gap name
}

# 5. AgentEvent  ← written to JSONL stream; P3 dashboard reads this
{
    "event_id": str,        # uuid4 string
    "timestamp": str,       # ISO 8601 UTC e.g. "2026-06-09T14:32:01Z"
    "agent": str,           # "orchestrator" | "critic" | "curriculum"
    "iteration": int,       # 0 for orchestrator (pre-loop), 1–3 for loop agents
    "decision": str,        # human-readable one-liner of what was decided
    "reason": str | None,   # why (can be null)
    "data": dict            # full agent output payload (OrchestratorOutput, CriticBatchOutput, or CurriculumOutput)
}

# 6. PipelineLoopState  ← return type of run_agentic_loop()
{
    "iteration": int,
    "strategy": dict,               # OrchestratorOutput
    "final_samples": list[dict],    # accepted (+ rewritten) samples
    "final_stats": dict,            # last CriticBatchOutput["stats"]
    "curriculum_outputs": list[dict], # one CurriculumOutput per loop iteration (may be empty)
    "events": list[dict],           # all AgentEvents in order
    "exit_reason": str              # "threshold_met" | "max_iterations" | "all_accepted_first_pass"
}

FILE STRUCTURE TO CREATE:
modifai/
└── agents/
    ├── __init__.py
    ├── schemas.py              ← Session A creates this
    ├── orchestrator.py         ← Session A creates this
    ├── curriculum.py           ← Session B creates this
    ├── logging_utils.py        ← Session C creates this
    ├── pipeline_loop.py        ← Session C creates this
    └── tests/
        ├── __init__.py
        ├── test_orchestrator.py    ← Session A creates this
        ├── test_curriculum.py      ← Session B creates this
        └── test_pipeline_e2e.py    ← Session C creates this
```

---

---

# SESSION A — Orchestrator Agent

## Your mission
Build the `OrchestratorAgent` class. It uses the **Bedrock converse API with tool-use**
to force structured JSON output. No free-form text — the model MUST call the tool.
Also build `schemas.py` with all shared TypedDicts.

## Files to create
1. `modifai/agents/__init__.py` (empty + exports)
2. `modifai/agents/schemas.py`
3. `modifai/agents/orchestrator.py`
4. `modifai/agents/tests/__init__.py` (empty)
5. `modifai/agents/tests/test_orchestrator.py`

---

## File 1: `modifai/agents/__init__.py`

```python
"""Modifai agentic pipeline — Orchestrator, Critic, Curriculum."""

from modifai.agents.orchestrator import OrchestratorAgent

__all__ = ["OrchestratorAgent"]
```

---

## File 2: `modifai/agents/schemas.py`

Implement ALL of these TypedDicts and dataclasses exactly. Other sessions import from here.

```python
"""
Shared schema definitions for the Modifai agentic pipeline.
DO NOT MODIFY field names — P2 (infra) and P3 (frontend) depend on these exact keys.
"""
from __future__ import annotations
from typing import TypedDict, Optional, List, Literal


# ── Orchestrator ──────────────────────────────────────────────────────────────

class DocMetadata(TypedDict):
    filename: str
    page_count: int
    domain: str                   # e.g. "customer support", "HR policy", "engineering runbook"
    estimated_chunk_count: int


class OrchestratorInput(TypedDict):
    goal: str
    doc_metadata: DocMetadata


class OrchestratorOutput(TypedDict):
    intent: Literal["QA", "instruction", "tutor"]
    quality_threshold: float      # 0.5 – 0.95
    samples_per_chunk: int        # 3 – 8
    reasoning: str


# ── Critic (defined here for reference; implementation is in critic.py) ───────

class CriticVerdict(TypedDict):
    verdict: Literal["accept", "rewrite", "reject"]
    reason: str
    rewritten_output: Optional[str]


class CriticStats(TypedDict):
    total: int
    accepted: int
    rewritten: int
    rejected: int
    accept_pct: float


class CriticBatchOutput(TypedDict):
    verdicts: List[CriticVerdict]
    stats: CriticStats


# ── Curriculum ────────────────────────────────────────────────────────────────

class GapCategory(TypedDict):
    name: str           # snake_case identifier
    description: str
    example_bad: str
    example_good: str


class CurriculumInput(TypedDict):
    rejection_reasons: List[str]
    strategy: OrchestratorOutput
    iteration: int


class CurriculumOutput(TypedDict):
    gap_categories: List[GapCategory]   # len >= 3
    targeted_prompt: str
    priority_focus: str                 # name of the most critical gap


# ── Logging ───────────────────────────────────────────────────────────────────

class AgentEvent(TypedDict):
    event_id: str           # uuid4
    timestamp: str          # ISO 8601 UTC
    agent: Literal["orchestrator", "critic", "curriculum"]
    iteration: int          # 0 = pre-loop (orchestrator), 1–3 = loop
    decision: str           # human-readable one-liner
    reason: Optional[str]
    data: dict              # full payload


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineLoopState(TypedDict):
    iteration: int
    strategy: OrchestratorOutput
    final_samples: List[dict]
    final_stats: CriticStats
    curriculum_outputs: List[CurriculumOutput]
    events: List[AgentEvent]
    exit_reason: Literal["threshold_met", "max_iterations", "all_accepted_first_pass"]
```

---

## File 3: `modifai/agents/orchestrator.py`

Implement this exactly. Key points:
- Use `client.converse()` — NOT `invoke_model`
- Use `toolChoice={"tool": {"name": "set_pipeline_strategy"}}` to FORCE tool use
- Retry once on malformed/missing tool output before raising
- Validate output ranges before returning

```python
"""
OrchestratorAgent — decides pipeline strategy from goal + doc metadata.

Uses Bedrock converse API with forced tool-use to guarantee structured JSON output.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from modifai.agents.schemas import (
    DocMetadata,
    OrchestratorInput,
    OrchestratorOutput,
)

logger = logging.getLogger(__name__)

# ── Bedrock tool definition ────────────────────────────────────────────────────

_TOOL_SPEC = {
    "toolSpec": {
        "name": "set_pipeline_strategy",
        "description": (
            "Set the strategy for the Modifai fine-tuning pipeline based on "
            "the user's goal and document metadata."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["QA", "instruction", "tutor"],
                        "description": (
                            "Generation intent. Use 'QA' for support docs/FAQs. "
                            "Use 'instruction' for SOPs/how-to guides. "
                            "Use 'tutor' for educational or training material."
                        ),
                    },
                    "quality_threshold": {
                        "type": "number",
                        "description": (
                            "Critic quality threshold between 0.5 and 0.95. "
                            "Use 0.75–0.9 for precise domains (legal, medical). "
                            "Use 0.55–0.7 for narrative domains. Default 0.7."
                        ),
                    },
                    "samples_per_chunk": {
                        "type": "integer",
                        "description": (
                            "Synthetic samples to generate per text chunk. "
                            "Use 3 for dense technical docs. "
                            "Use 5–6 for rich narrative docs. Max 8."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Brief one-sentence explanation of your strategy choice. "
                            "This is shown in the UI and logs."
                        ),
                    },
                },
                "required": ["intent", "quality_threshold", "samples_per_chunk", "reasoning"],
            }
        },
    }
}

_SYSTEM_PROMPT = """\
You are the Orchestrator agent for Modifai, an automated LLM fine-tuning platform.

Your ONLY job is to call the `set_pipeline_strategy` tool with the correct strategy
for the given document and goal. Never respond with plain text.

DECISION RULES:

intent:
  - "QA"          → support docs, FAQs, Q&A pairs, manuals users query
  - "instruction" → SOPs, how-to guides, procedural docs, step-by-step content
  - "tutor"       → educational material, textbooks, training curricula

quality_threshold:
  - 0.80–0.90 → short, precise, high-stakes docs (legal, compliance, medical, financial)
  - 0.65–0.75 → standard business docs (SOPs, HR policy, runbooks) ← DEFAULT range
  - 0.55–0.65 → narrative, creative, or loosely structured content
  Never set below 0.5 or above 0.95.

samples_per_chunk:
  - 3 → dense technical content (many facts per chunk, risk of hallucination is high)
  - 4–5 → standard docs ← DEFAULT
  - 6–8 → rich narrative docs with many paraphraseable angles
  Never exceed 8.

You MUST call the set_pipeline_strategy tool. Output nothing else.
"""


class OrchestratorAgent:
    """
    Decides pipeline strategy from a goal string and document metadata.

    Usage:
        agent = OrchestratorAgent()
        strategy = agent.run(
            goal="Fine-tune a Q&A bot on our customer support runbook",
            doc_metadata={
                "filename": "support_sop.pdf",
                "page_count": 24,
                "domain": "customer support",
                "estimated_chunk_count": 48,
            }
        )
        # strategy is an OrchestratorOutput TypedDict
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_retries: int = 1,
    ):
        self.model_id = model_id or os.environ.get(
            "AWS_MODEL_ID", "amazon.nova-micro-v1:0"
        )
        self.region = region or os.environ.get("AWS_REGION", "ap-south-1")
        self.max_retries = max_retries  # retry once on bad tool output
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, goal: str, doc_metadata: DocMetadata) -> OrchestratorOutput:
        """
        Run the Orchestrator agent.

        Args:
            goal: Natural-language description of what the user wants.
            doc_metadata: DocMetadata TypedDict with filename, page_count, domain,
                          estimated_chunk_count.

        Returns:
            OrchestratorOutput with intent, quality_threshold, samples_per_chunk,
            reasoning.

        Raises:
            ValueError: If the model fails to produce valid tool output after retries.
            ClientError: On AWS API errors.
        """
        user_message = self._build_user_message(goal, doc_metadata)
        attempt = 0

        while attempt <= self.max_retries:
            try:
                raw = self._call_bedrock(user_message)
                strategy = self._parse_tool_output(raw)
                self._validate(strategy)
                logger.info(
                    "Orchestrator strategy: intent=%s threshold=%.2f spc=%d",
                    strategy["intent"],
                    strategy["quality_threshold"],
                    strategy["samples_per_chunk"],
                )
                return strategy
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise ValueError(
                        f"Orchestrator failed to produce valid strategy after "
                        f"{self.max_retries + 1} attempt(s): {exc}"
                    ) from exc
                logger.warning(
                    "Orchestrator output malformed (attempt %d/%d): %s — retrying",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _build_user_message(self, goal: str, doc_metadata: DocMetadata) -> str:
        return (
            f"Goal: {goal}\n\n"
            f"Document metadata:\n"
            f"  filename: {doc_metadata['filename']}\n"
            f"  pages: {doc_metadata['page_count']}\n"
            f"  domain: {doc_metadata['domain']}\n"
            f"  estimated chunks: {doc_metadata['estimated_chunk_count']}\n\n"
            f"Choose the best pipeline strategy and call set_pipeline_strategy."
        )

    def _call_bedrock(self, user_message: str) -> dict:
        response = self._client.converse(
            modelId=self.model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_message}],
                }
            ],
            toolConfig={
                "tools": [_TOOL_SPEC],
                "toolChoice": {"tool": {"name": "set_pipeline_strategy"}},
            },
        )
        return response

    def _parse_tool_output(self, response: dict) -> dict:
        """Extract tool input dict from converse response."""
        content_blocks = response["output"]["message"]["content"]
        for block in content_blocks:
            if block.get("toolUse", {}).get("name") == "set_pipeline_strategy":
                return block["toolUse"]["input"]
        raise ValueError(
            "Model did not call set_pipeline_strategy tool. "
            f"Response content: {content_blocks}"
        )

    def _validate(self, strategy: dict) -> None:
        """Range-check the strategy fields and raise ValueError if invalid."""
        if strategy.get("intent") not in ("QA", "instruction", "tutor"):
            raise ValueError(f"Invalid intent: {strategy.get('intent')}")

        threshold = strategy.get("quality_threshold")
        if not isinstance(threshold, (int, float)) or not (0.5 <= threshold <= 0.95):
            raise ValueError(f"quality_threshold out of range: {threshold}")

        spc = strategy.get("samples_per_chunk")
        if not isinstance(spc, int) or not (3 <= spc <= 8):
            raise ValueError(f"samples_per_chunk out of range: {spc}")

        if not strategy.get("reasoning"):
            raise ValueError("reasoning must be a non-empty string")
```

---

## File 4: `modifai/agents/tests/test_orchestrator.py`

All tests mock boto3 — no real AWS calls. Run with `pytest modifai/agents/tests/`.

```python
"""Unit tests for OrchestratorAgent. All AWS calls are mocked."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from modifai.agents.orchestrator import OrchestratorAgent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bedrock_response(tool_input: dict) -> dict:
    """Build a fake Bedrock converse response containing a tool call."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "fake-id-001",
                            "name": "set_pipeline_strategy",
                            "input": tool_input,
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
    }


def _make_bad_response() -> dict:
    """Bedrock response with NO tool call (plain text instead)."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Here is the strategy: QA mode, threshold 0.7"}],
            }
        },
        "stopReason": "end_turn",
    }


VALID_DOC_METADATA = {
    "filename": "support_sop.pdf",
    "page_count": 24,
    "domain": "customer support",
    "estimated_chunk_count": 48,
}

VALID_TOOL_INPUT = {
    "intent": "QA",
    "quality_threshold": 0.72,
    "samples_per_chunk": 5,
    "reasoning": "Customer support FAQ — QA intent, standard threshold.",
}


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("modifai.agents.orchestrator.boto3.client")
def test_happy_path_returns_strategy(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.return_value = _make_bedrock_response(VALID_TOOL_INPUT)

    agent = OrchestratorAgent()
    result = agent.run(goal="Build a support Q&A bot", doc_metadata=VALID_DOC_METADATA)

    assert result["intent"] == "QA"
    assert result["quality_threshold"] == 0.72
    assert result["samples_per_chunk"] == 5
    assert isinstance(result["reasoning"], str) and len(result["reasoning"]) > 0


@patch("modifai.agents.orchestrator.boto3.client")
def test_retries_once_on_bad_output_then_succeeds(mock_boto):
    """First call returns bad response (no tool), second returns valid tool call."""
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.side_effect = [
        _make_bad_response(),
        _make_bedrock_response(VALID_TOOL_INPUT),
    ]

    agent = OrchestratorAgent(max_retries=1)
    result = agent.run(goal="Build a support Q&A bot", doc_metadata=VALID_DOC_METADATA)

    assert result["intent"] == "QA"
    assert mock_client.converse.call_count == 2


@patch("modifai.agents.orchestrator.boto3.client")
def test_raises_after_exhausting_retries(mock_boto):
    """Both attempts fail — should raise ValueError."""
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.return_value = _make_bad_response()

    agent = OrchestratorAgent(max_retries=1)

    with pytest.raises(ValueError, match="failed to produce valid strategy"):
        agent.run(goal="Build a support Q&A bot", doc_metadata=VALID_DOC_METADATA)


@pytest.mark.parametrize("bad_intent", ["qa", "Q&A", "chatbot", "", None])
@patch("modifai.agents.orchestrator.boto3.client")
def test_rejects_invalid_intent(mock_boto, bad_intent):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_tool_input = {**VALID_TOOL_INPUT, "intent": bad_intent}
    mock_client.converse.return_value = _make_bedrock_response(bad_tool_input)

    agent = OrchestratorAgent(max_retries=0)

    with pytest.raises(ValueError):
        agent.run(goal="test", doc_metadata=VALID_DOC_METADATA)


@pytest.mark.parametrize("bad_threshold", [0.3, 0.99, 1.5, -0.1, "high"])
@patch("modifai.agents.orchestrator.boto3.client")
def test_rejects_out_of_range_threshold(mock_boto, bad_threshold):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_tool_input = {**VALID_TOOL_INPUT, "quality_threshold": bad_threshold}
    mock_client.converse.return_value = _make_bedrock_response(bad_tool_input)

    agent = OrchestratorAgent(max_retries=0)

    with pytest.raises(ValueError):
        agent.run(goal="test", doc_metadata=VALID_DOC_METADATA)


@pytest.mark.parametrize("bad_spc", [1, 2, 9, 20, "five"])
@patch("modifai.agents.orchestrator.boto3.client")
def test_rejects_out_of_range_samples_per_chunk(mock_boto, bad_spc):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_tool_input = {**VALID_TOOL_INPUT, "samples_per_chunk": bad_spc}
    mock_client.converse.return_value = _make_bedrock_response(bad_tool_input)

    agent = OrchestratorAgent(max_retries=0)

    with pytest.raises(ValueError):
        agent.run(goal="test", doc_metadata=VALID_DOC_METADATA)


@patch("modifai.agents.orchestrator.boto3.client")
def test_instruction_intent_for_sop_domain(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    tool_input = {**VALID_TOOL_INPUT, "intent": "instruction"}
    mock_client.converse.return_value = _make_bedrock_response(tool_input)

    agent = OrchestratorAgent()
    result = agent.run(
        goal="Create instruction-following data from our engineering runbook",
        doc_metadata={
            "filename": "eng_runbook.pdf",
            "page_count": 60,
            "domain": "engineering runbook",
            "estimated_chunk_count": 120,
        },
    )
    assert result["intent"] == "instruction"
```

---

## Session A — Definition of Done

- [ ] `schemas.py` exists with all TypedDicts — importable with no errors
- [ ] `orchestrator.py` runs `from modifai.agents.orchestrator import OrchestratorAgent` cleanly
- [ ] `OrchestratorAgent().run(goal, doc_metadata)` returns a valid `OrchestratorOutput` dict
- [ ] All 7 unit tests pass: `pytest modifai/agents/tests/test_orchestrator.py -v`
- [ ] No hardcoded AWS credentials — all via env vars or boto3 default chain
- [ ] Paste the final `OrchestratorOutput` schema JSON into the team chat for P2

---

---

# SESSION B — Curriculum Agent

## Your mission
Build the `CurriculumAgent` class. It reads Critic rejection reasons, clusters them
into ≥3 gap categories using Bedrock, and outputs a `targeted_prompt` for the dataset
generator. **Critic code is already done** — you only depend on its output schema
(defined in the Context Block).

## Files to create
1. `modifai/agents/curriculum.py`
2. `modifai/agents/tests/test_curriculum.py`

> **Note:** `modifai/agents/__init__.py` and `modifai/agents/schemas.py` are created
> by Session A. If running in parallel, create stub versions locally and reconcile later.
> Session C does the final wiring.

---

## File 1: `modifai/agents/curriculum.py`

Key design points:
- Use Bedrock converse API with a forced `analyze_curriculum` tool
- Must output **at least 3** gap categories — validate and raise if fewer
- The `targeted_prompt` field becomes a direct instruction injected into
  `modifai/core/dataset_generation.py` — make it concrete and actionable

```python
"""
CurriculumAgent — analyses Critic rejection reasons and generates a targeted
data generation prompt to patch identified weaknesses.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import boto3

from modifai.agents.schemas import (
    CurriculumInput,
    CurriculumOutput,
    GapCategory,
    OrchestratorOutput,
)

logger = logging.getLogger(__name__)

# ── Bedrock tool definition ────────────────────────────────────────────────────

_TOOL_SPEC = {
    "toolSpec": {
        "name": "analyze_curriculum",
        "description": (
            "Analyse Critic rejection reasons, identify gap categories, "
            "and produce a targeted data generation prompt."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "gap_categories": {
                        "type": "array",
                        "minItems": 3,
                        "description": "At least 3 distinct weakness categories found in rejections.",
                        "items": {
                            "type": "object",
                            "required": ["name", "description", "example_bad", "example_good"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "snake_case identifier, e.g. 'lacks_step_by_step_reasoning'",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "One sentence describing the failure pattern.",
                                },
                                "example_bad": {
                                    "type": "string",
                                    "description": "Short example of a bad output exhibiting this gap.",
                                },
                                "example_good": {
                                    "type": "string",
                                    "description": "Short example of a good output that fixes this gap.",
                                },
                            },
                        },
                    },
                    "targeted_prompt": {
                        "type": "string",
                        "description": (
                            "A concrete, specific generation instruction (2–5 sentences) "
                            "that tells the dataset generator exactly how to fix the identified gaps. "
                            "Must reference specific gap names. "
                            "This will be appended to the existing generation system prompt."
                        ),
                    },
                    "priority_focus": {
                        "type": "string",
                        "description": "The snake_case name of the single most critical gap to fix first.",
                    },
                },
                "required": ["gap_categories", "targeted_prompt", "priority_focus"],
            }
        },
    }
}

_SYSTEM_PROMPT = """\
You are the Curriculum agent for Modifai, an automated LLM fine-tuning platform.

Your job: analyze why training samples were rejected by the Critic agent, identify
at least 3 distinct weakness patterns, and output a targeted generation prompt
that will make the next round of samples significantly better.

INPUTS YOU RECEIVE:
- A list of rejection reasons from the Critic
- The current pipeline strategy (intent, threshold, samples_per_chunk)
- The loop iteration number (1 = first retry, higher = subsequent retries)

YOUR RESPONSIBILITIES:
1. Cluster rejection reasons into SPECIFIC, DISTINCT gap categories (min 3)
2. Name each gap in snake_case (e.g. "lacks_step_by_step_reasoning")
3. Write a targeted_prompt that is CONCRETE and ACTIONABLE — not generic advice
4. Set priority_focus to the most impactful gap to fix

GAP CATEGORY EXAMPLES (use as reference, not a fixed list):
- "lacks_step_by_step_reasoning" — answer skips intermediate steps
- "too_vague_on_entities" — doesn't name specific items from source
- "format_mismatch" — paragraph where list is needed, or vice versa
- "factual_drift" — introduces facts not in the source chunk
- "truncated_answer" — answer is cut short or incomplete
- "hallucinated_procedure" — describes steps that aren't in the document
- "passive_language" — uses vague "it may be" instead of assertive statements
- "missing_preconditions" — omits required context before steps
- "no_grounding_in_source" — answer could be from any generic knowledge

TARGETED PROMPT QUALITY BAR:
BAD:  "Generate better answers that are more specific."
GOOD: "Each answer MUST enumerate all numbered steps found in the source chunk.
       Name every specific tool, system, or person referenced in the source.
       Never introduce facts not explicitly stated in the chunk.
       If the answer would be a list, format it as a numbered list."

You MUST call the analyze_curriculum tool. Output nothing else.
"""


class CurriculumAgent:
    """
    Analyses Critic rejection patterns and generates a targeted data generation prompt.

    Usage:
        agent = CurriculumAgent()
        output = agent.run(
            rejection_reasons=["too vague", "missing steps", ...],
            strategy=orchestrator_output,
            iteration=1,
        )
        # output["targeted_prompt"] → inject into dataset generation
        # output["gap_categories"] → log for P3 dashboard
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_retries: int = 1,
    ):
        self.model_id = model_id or os.environ.get(
            "AWS_MODEL_ID", "amazon.nova-micro-v1:0"
        )
        self.region = region or os.environ.get("AWS_REGION", "ap-south-1")
        self.max_retries = max_retries
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        rejection_reasons: List[str],
        strategy: OrchestratorOutput,
        iteration: int,
    ) -> CurriculumOutput:
        """
        Analyse rejection reasons and return a targeted curriculum.

        Args:
            rejection_reasons: List of reason strings from CriticAgent.run_batch()
                               (only rejected/rewritten verdicts' reasons).
            strategy: OrchestratorOutput dict from the Orchestrator agent.
            iteration: Current loop iteration number (1-based).

        Returns:
            CurriculumOutput with gap_categories (≥3), targeted_prompt, priority_focus.

        Raises:
            ValueError: If model fails to produce ≥3 gap categories after retries.
        """
        if not rejection_reasons:
            raise ValueError(
                "CurriculumAgent.run() called with empty rejection_reasons. "
                "Only call Curriculum when there are actual rejections."
            )

        user_message = self._build_user_message(rejection_reasons, strategy, iteration)
        attempt = 0

        while attempt <= self.max_retries:
            try:
                raw = self._call_bedrock(user_message)
                output = self._parse_tool_output(raw)
                self._validate(output)
                logger.info(
                    "Curriculum iter=%d gaps=%d priority=%s",
                    iteration,
                    len(output["gap_categories"]),
                    output["priority_focus"],
                )
                return output
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise ValueError(
                        f"CurriculumAgent failed to produce valid output after "
                        f"{self.max_retries + 1} attempt(s): {exc}"
                    ) from exc
                logger.warning(
                    "Curriculum output malformed (attempt %d/%d): %s — retrying",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _build_user_message(
        self,
        rejection_reasons: List[str],
        strategy: OrchestratorOutput,
        iteration: int,
    ) -> str:
        numbered = "\n".join(
            f"  {i + 1}. {reason}" for i, reason in enumerate(rejection_reasons)
        )
        return (
            f"Loop iteration: {iteration}\n\n"
            f"Pipeline strategy:\n"
            f"  intent: {strategy['intent']}\n"
            f"  quality_threshold: {strategy['quality_threshold']}\n"
            f"  samples_per_chunk: {strategy['samples_per_chunk']}\n\n"
            f"Critic rejection reasons ({len(rejection_reasons)} total):\n"
            f"{numbered}\n\n"
            f"Identify at least 3 gap categories and produce a targeted generation prompt. "
            f"Call analyze_curriculum."
        )

    def _call_bedrock(self, user_message: str) -> dict:
        return self._client.converse(
            modelId=self.model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_message}],
                }
            ],
            toolConfig={
                "tools": [_TOOL_SPEC],
                "toolChoice": {"tool": {"name": "analyze_curriculum"}},
            },
        )

    def _parse_tool_output(self, response: dict) -> dict:
        content_blocks = response["output"]["message"]["content"]
        for block in content_blocks:
            if block.get("toolUse", {}).get("name") == "analyze_curriculum":
                return block["toolUse"]["input"]
        raise ValueError(
            f"Model did not call analyze_curriculum tool. Content: {content_blocks}"
        )

    def _validate(self, output: dict) -> None:
        gaps = output.get("gap_categories", [])
        if len(gaps) < 3:
            raise ValueError(
                f"Curriculum must produce ≥3 gap categories, got {len(gaps)}."
            )

        for i, gap in enumerate(gaps):
            for field in ("name", "description", "example_bad", "example_good"):
                if not gap.get(field):
                    raise ValueError(f"gap_categories[{i}].{field} is empty.")

        if not output.get("targeted_prompt") or len(output["targeted_prompt"]) < 30:
            raise ValueError("targeted_prompt is missing or too short (must be ≥30 chars).")

        priority = output.get("priority_focus")
        gap_names = {g["name"] for g in gaps}
        if priority not in gap_names:
            raise ValueError(
                f"priority_focus '{priority}' does not match any gap category name. "
                f"Valid names: {gap_names}"
            )

    # ── Utility: extract rejection reasons from batch output ──────────────────

    @staticmethod
    def extract_rejection_reasons(batch_output: dict) -> List[str]:
        """
        Convenience method: pull reason strings from CriticBatchOutput
        for all rejected or rewritten verdicts.

        Args:
            batch_output: CriticBatchOutput dict from CriticAgent.run_batch()

        Returns:
            List of reason strings (may be empty if all accepted).
        """
        reasons = []
        for verdict in batch_output.get("verdicts", []):
            if verdict.get("verdict") in ("reject", "rewrite"):
                reason = verdict.get("reason", "").strip()
                if reason:
                    reasons.append(reason)
        return reasons
```

---

## File 2: `modifai/agents/tests/test_curriculum.py`

```python
"""Unit tests for CurriculumAgent. All AWS calls are mocked."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from modifai.agents.curriculum import CurriculumAgent

# ── Helpers ────────────────────────────────────────────────────────────────────

VALID_STRATEGY = {
    "intent": "instruction",
    "quality_threshold": 0.72,
    "samples_per_chunk": 5,
    "reasoning": "SOP doc, instruction intent.",
}

VALID_GAP_CATEGORIES = [
    {
        "name": "lacks_step_by_step_reasoning",
        "description": "Answer skips intermediate steps.",
        "example_bad": "Just restart the server.",
        "example_good": "1. SSH into the server. 2. Run `sudo systemctl restart app`. 3. Check logs.",
    },
    {
        "name": "too_vague_on_entities",
        "description": "Doesn't name specific tools or systems from source.",
        "example_bad": "Use the monitoring tool to check.",
        "example_good": "Open Datadog and navigate to the Infra tab to check CPU usage.",
    },
    {
        "name": "factual_drift",
        "description": "Introduces facts not in the source chunk.",
        "example_bad": "The system uses Redis for caching (not mentioned in source).",
        "example_good": "The system uses the cache layer described in section 3.2.",
    },
]

VALID_TOOL_INPUT = {
    "gap_categories": VALID_GAP_CATEGORIES,
    "targeted_prompt": (
        "Each answer MUST enumerate all numbered steps from the source chunk. "
        "Name every specific tool, system, or person referenced. "
        "Never introduce facts not explicitly stated in the source chunk."
    ),
    "priority_focus": "lacks_step_by_step_reasoning",
}

SAMPLE_REJECTION_REASONS = [
    "Answer is too vague, does not name specific systems",
    "Missing intermediate steps in the procedure",
    "Introduces facts not present in the source chunk",
    "Answer is a single sentence when source has 5 steps",
]


def _make_bedrock_response(tool_input: dict) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "fake-id-002",
                            "name": "analyze_curriculum",
                            "input": tool_input,
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
    }


def _make_bad_response() -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "I found 3 gaps: vague, missing steps, hallucination."}],
            }
        },
        "stopReason": "end_turn",
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("modifai.agents.curriculum.boto3.client")
def test_happy_path_returns_curriculum(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.return_value = _make_bedrock_response(VALID_TOOL_INPUT)

    agent = CurriculumAgent()
    result = agent.run(
        rejection_reasons=SAMPLE_REJECTION_REASONS,
        strategy=VALID_STRATEGY,
        iteration=1,
    )

    assert len(result["gap_categories"]) == 3
    assert result["priority_focus"] == "lacks_step_by_step_reasoning"
    assert len(result["targeted_prompt"]) >= 30


@patch("modifai.agents.curriculum.boto3.client")
def test_retries_once_on_bad_output_then_succeeds(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.converse.side_effect = [
        _make_bad_response(),
        _make_bedrock_response(VALID_TOOL_INPUT),
    ]

    agent = CurriculumAgent(max_retries=1)
    result = agent.run(
        rejection_reasons=SAMPLE_REJECTION_REASONS,
        strategy=VALID_STRATEGY,
        iteration=1,
    )
    assert mock_client.converse.call_count == 2


@patch("modifai.agents.curriculum.boto3.client")
def test_raises_on_fewer_than_3_gap_categories(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_input = {
        **VALID_TOOL_INPUT,
        "gap_categories": VALID_GAP_CATEGORIES[:2],  # only 2
    }
    mock_client.converse.return_value = _make_bedrock_response(bad_input)

    agent = CurriculumAgent(max_retries=0)
    with pytest.raises(ValueError, match="≥3 gap categories"):
        agent.run(
            rejection_reasons=SAMPLE_REJECTION_REASONS,
            strategy=VALID_STRATEGY,
            iteration=1,
        )


@patch("modifai.agents.curriculum.boto3.client")
def test_raises_when_priority_focus_not_in_gap_names(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    bad_input = {**VALID_TOOL_INPUT, "priority_focus": "nonexistent_gap"}
    mock_client.converse.return_value = _make_bedrock_response(bad_input)

    agent = CurriculumAgent(max_retries=0)
    with pytest.raises(ValueError, match="priority_focus"):
        agent.run(
            rejection_reasons=SAMPLE_REJECTION_REASONS,
            strategy=VALID_STRATEGY,
            iteration=1,
        )


def test_raises_on_empty_rejection_reasons():
    agent = CurriculumAgent()
    with pytest.raises(ValueError, match="empty rejection_reasons"):
        agent.run(rejection_reasons=[], strategy=VALID_STRATEGY, iteration=1)


def test_extract_rejection_reasons_filters_accepted():
    batch_output = {
        "verdicts": [
            {"verdict": "accept", "reason": "looks good", "rewritten_output": None},
            {"verdict": "reject", "reason": "too vague", "rewritten_output": None},
            {"verdict": "rewrite", "reason": "missing steps", "rewritten_output": "better answer"},
            {"verdict": "accept", "reason": "fine", "rewritten_output": None},
        ],
        "stats": {"total": 4, "accepted": 2, "rewritten": 1, "rejected": 1, "accept_pct": 50.0},
    }
    reasons = CurriculumAgent.extract_rejection_reasons(batch_output)
    assert reasons == ["too vague", "missing steps"]


def test_extract_rejection_reasons_returns_empty_if_all_accepted():
    batch_output = {
        "verdicts": [
            {"verdict": "accept", "reason": "great", "rewritten_output": None},
        ],
        "stats": {"total": 1, "accepted": 1, "rewritten": 0, "rejected": 0, "accept_pct": 100.0},
    }
    reasons = CurriculumAgent.extract_rejection_reasons(batch_output)
    assert reasons == []
```

---

## Session B — Definition of Done

- [ ] `curriculum.py` runs `from modifai.agents.curriculum import CurriculumAgent` cleanly
- [ ] `CurriculumAgent().run(rejection_reasons, strategy, iteration)` returns valid `CurriculumOutput`
- [ ] `CurriculumAgent.extract_rejection_reasons(batch_output)` correctly filters non-accepted verdicts
- [ ] All 6 unit tests pass: `pytest modifai/agents/tests/test_curriculum.py -v`
- [ ] Validated: output always has ≥3 gap categories and `priority_focus` matches a category name
- [ ] No hardcoded AWS credentials

---

---

# SESSION C — Pipeline Wiring, Logging & E2E Tests

## Your mission
1. Build `AgentEventLogger` — writes every agent decision to a JSONL event stream
2. Build `run_agentic_loop()` — wires Orchestrator → DatasetGeneration → Critic → Curriculum
3. Write comprehensive E2E tests (mocked AWS) that prove the loop works
4. Handle the edge case: all samples accepted on first pass (skip Curriculum entirely)

## Prerequisites
- Session A complete: `OrchestratorAgent` importable
- Session B complete: `CurriculumAgent` importable
- Critic code received: paste it in, verify its class interface matches the Context Block schema
- Existing `modifai/core/dataset_generation.py` — understand its generate function signature

## IMPORTANT: Interface check with existing dataset_generation.py

Before writing pipeline_loop.py, read `modifai/core/dataset_generation.py` and
confirm there is a callable like one of these (pick the one that matches):

```python
# Option A: if the function is standalone
generate_dataset(chunks, mode="QA", samples_per_chunk=3, custom_prompt=None)

# Option B: if it's class-based
gen = DatasetGenerator(config)
gen.generate(chunks, custom_prompt=None)
```

The `targeted_prompt` from the Curriculum agent goes in as `custom_prompt` (or equivalent).
If the existing function has no `custom_prompt` parameter, ADD ONE — it should be appended
to whatever system prompt the generation step currently uses.

---

## File 1: `modifai/agents/logging_utils.py`

```python
"""
AgentEventLogger — writes structured agent decision events to a JSONL file.
P3 dashboard polls for these events to build the live feed.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from modifai.agents.schemas import AgentEvent


class AgentEventLogger:
    """
    Thread-unsafe (single-process) structured event logger.
    Each call to log() appends one AgentEvent as a JSON line.

    Usage:
        logger = AgentEventLogger(path="agent_events.jsonl")
        logger.log(
            agent="orchestrator",
            iteration=0,
            decision="intent=QA, threshold=0.72, spc=5",
            reason="Customer support FAQ domain",
            data=strategy_dict,
        )
    """

    def __init__(self, path: str = "agent_events.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on init so each pipeline run starts fresh
        self.path.write_text("")

    def log(
        self,
        agent: str,
        iteration: int,
        decision: str,
        data: dict,
        reason: Optional[str] = None,
    ) -> AgentEvent:
        """
        Append one event to the JSONL file and return it.

        Args:
            agent: "orchestrator" | "critic" | "curriculum"
            iteration: 0 for orchestrator (pre-loop), 1–3 for loop agents
            decision: Human-readable one-liner (shown in P3 dashboard)
            data: Full agent output payload dict
            reason: Optional explanation (shown in P3 dashboard)

        Returns:
            The AgentEvent dict that was written.
        """
        event: AgentEvent = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "agent": agent,
            "iteration": iteration,
            "decision": decision,
            "reason": reason,
            "data": data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        return event

    def read_all(self) -> List[AgentEvent]:
        """Read and return all events from the log file."""
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events
```

---

## File 2: `modifai/agents/pipeline_loop.py`

This is the most important file — the full 3-agent orchestration.

```python
"""
run_agentic_loop() — wires Orchestrator → DatasetGeneration → Critic → Curriculum.

The loop runs at most max_iterations times. Early exit if:
  - All samples accepted on first Critic pass (exit_reason = "all_accepted_first_pass")
  - accept_pct >= strategy["quality_threshold"] (exit_reason = "threshold_met")
  - max_iterations reached (exit_reason = "max_iterations")
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from modifai.agents.orchestrator import OrchestratorAgent
from modifai.agents.curriculum import CurriculumAgent
from modifai.agents.logging_utils import AgentEventLogger
from modifai.agents.schemas import (
    DocMetadata,
    OrchestratorOutput,
    PipelineLoopState,
)

# Import the existing Critic agent — adjust import path to match actual file
# from modifai.agents.critic import CriticAgent   ← update this path

# Import existing dataset generation — adjust to match actual function signature
# from modifai.core.dataset_generation import generate_dataset

logger = logging.getLogger(__name__)


def run_agentic_loop(
    goal: str,
    doc_metadata: DocMetadata,
    chunks: List[str],
    max_iterations: int = 3,
    event_log_path: str = "agent_events.jsonl",
    model_id: Optional[str] = None,
    region: Optional[str] = None,
) -> PipelineLoopState:
    """
    Run the full Modifai agentic loop.

    Args:
        goal: User's goal string (e.g. "Build a Q&A bot for our HR policy docs")
        doc_metadata: DocMetadata TypedDict
        chunks: List of text chunks from the chunking step
        max_iterations: Maximum Critic→Curriculum loop iterations (default 3)
        event_log_path: Path to write JSONL event stream (read by P3 dashboard)
        model_id: Bedrock model ID override (default from env or nova-micro)
        region: AWS region override (default from env or ap-south-1)

    Returns:
        PipelineLoopState with final_samples, final_stats, events, exit_reason
    """

    event_logger = AgentEventLogger(path=event_log_path)
    curriculum_outputs = []

    # ── Step 1: Orchestrator ───────────────────────────────────────────────────
    logger.info("Running OrchestratorAgent...")
    orchestrator = OrchestratorAgent(model_id=model_id, region=region)
    strategy: OrchestratorOutput = orchestrator.run(
        goal=goal, doc_metadata=doc_metadata
    )

    event_logger.log(
        agent="orchestrator",
        iteration=0,
        decision=(
            f"intent={strategy['intent']}, "
            f"threshold={strategy['quality_threshold']}, "
            f"spc={strategy['samples_per_chunk']}"
        ),
        reason=strategy["reasoning"],
        data=dict(strategy),
    )

    # ── Step 2: Initial dataset generation ────────────────────────────────────
    # NOTE: Adjust this call to match the actual generate_dataset signature.
    # The key parameter is custom_prompt=None initially (no targeted prompt yet).
    logger.info("Generating initial dataset with strategy: %s", strategy)
    samples = _generate_samples(
        chunks=chunks,
        intent=strategy["intent"],
        samples_per_chunk=strategy["samples_per_chunk"],
        custom_prompt=None,
    )
    logger.info("Generated %d samples", len(samples))

    # ── Step 3: Agentic loop ───────────────────────────────────────────────────
    # Import Critic here so import errors surface clearly
    from modifai.agents.critic import CriticAgent  # ← adjust path if needed
    critic = CriticAgent(model_id=model_id, region=region)
    curriculum_agent = CurriculumAgent(model_id=model_id, region=region)

    iteration = 0
    exit_reason = "max_iterations"

    while iteration < max_iterations:
        iteration += 1
        logger.info("=== Loop iteration %d / %d ===", iteration, max_iterations)

        # ── Critic pass ────────────────────────────────────────────────────────
        batch_output = critic.run_batch(samples)
        stats = batch_output["stats"]
        accept_pct = stats["accept_pct"]

        logger.info(
            "Critic iter=%d: accepted=%d rewritten=%d rejected=%d accept_pct=%.1f%%",
            iteration,
            stats["accepted"],
            stats["rewritten"],
            stats["rejected"],
            accept_pct,
        )

        event_logger.log(
            agent="critic",
            iteration=iteration,
            decision=(
                f"accepted={stats['accepted']}/{stats['total']} "
                f"({accept_pct:.1f}%), "
                f"rewritten={stats['rewritten']}, rejected={stats['rejected']}"
            ),
            reason=(
                f"accept_pct={accept_pct:.1f}% vs threshold={strategy['quality_threshold'] * 100:.1f}%"
            ),
            data=dict(stats),
        )

        # ── Early exit: all accepted on first pass ─────────────────────────────
        if iteration == 1 and accept_pct == 100.0:
            logger.info("All samples accepted on first pass — skipping Curriculum.")
            exit_reason = "all_accepted_first_pass"
            samples = _collect_accepted(batch_output)
            break

        # ── Threshold met ──────────────────────────────────────────────────────
        if accept_pct >= strategy["quality_threshold"] * 100:
            logger.info(
                "Quality threshold met (%.1f%% >= %.1f%%) — exiting loop.",
                accept_pct,
                strategy["quality_threshold"] * 100,
            )
            exit_reason = "threshold_met"
            samples = _collect_accepted(batch_output)
            break

        # ── Threshold NOT met: run Curriculum (unless last iteration) ──────────
        if iteration >= max_iterations:
            logger.warning(
                "Max iterations (%d) reached. Final accept_pct=%.1f%%.",
                max_iterations,
                accept_pct,
            )
            exit_reason = "max_iterations"
            samples = _collect_accepted(batch_output)
            break

        rejection_reasons = CurriculumAgent.extract_rejection_reasons(batch_output)
        if not rejection_reasons:
            # Paranoia check: stats say threshold not met but no rejection reasons
            logger.warning(
                "No rejection reasons extracted but accept_pct=%.1f%% < threshold. "
                "Exiting loop to avoid infinite loop.",
                accept_pct,
            )
            exit_reason = "max_iterations"
            samples = _collect_accepted(batch_output)
            break

        logger.info(
            "Running CurriculumAgent with %d rejection reasons...",
            len(rejection_reasons),
        )
        curriculum_output = curriculum_agent.run(
            rejection_reasons=rejection_reasons,
            strategy=strategy,
            iteration=iteration,
        )
        curriculum_outputs.append(curriculum_output)

        event_logger.log(
            agent="curriculum",
            iteration=iteration,
            decision=(
                f"identified {len(curriculum_output['gap_categories'])} gap categories, "
                f"priority={curriculum_output['priority_focus']}"
            ),
            reason=curriculum_output["targeted_prompt"][:120] + "...",
            data={
                "gap_categories": curriculum_output["gap_categories"],
                "priority_focus": curriculum_output["priority_focus"],
                "targeted_prompt": curriculum_output["targeted_prompt"],
            },
        )

        # ── Regenerate with targeted prompt ────────────────────────────────────
        logger.info("Regenerating dataset with targeted Curriculum prompt...")
        samples = _generate_samples(
            chunks=chunks,
            intent=strategy["intent"],
            samples_per_chunk=strategy["samples_per_chunk"],
            custom_prompt=curriculum_output["targeted_prompt"],
        )
        logger.info("Regenerated %d samples", len(samples))

    # ── Build and return final state ───────────────────────────────────────────
    all_events = event_logger.read_all()

    # Final stats: re-run Critic one last time if we exited via max_iterations
    # with no final pass (samples already set above)
    final_stats = stats  # last stats from the loop

    return PipelineLoopState(
        iteration=iteration,
        strategy=strategy,
        final_samples=samples,
        final_stats=final_stats,
        curriculum_outputs=curriculum_outputs,
        events=all_events,
        exit_reason=exit_reason,
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _generate_samples(
    chunks: List[str],
    intent: str,
    samples_per_chunk: int,
    custom_prompt: Optional[str],
) -> List[dict]:
    """
    Wrapper around the existing dataset_generation module.

    IMPORTANT: Update this function body to match the actual
    modifai/core/dataset_generation.py signature.

    The key requirement: `custom_prompt` must be injectable so the
    Curriculum agent's targeted_prompt can improve generation quality.
    """
    # ── ADAPTER: adjust the import and call below to match actual code ──────
    from modifai.core.dataset_generation import generate_dataset  # adjust if needed

    # Map intent string to the existing mode parameter
    mode_map = {"QA": "QA", "instruction": "instruction", "tutor": "tutor"}
    mode = mode_map.get(intent, "QA")

    samples = generate_dataset(
        chunks=chunks,
        mode=mode,
        samples_per_chunk=samples_per_chunk,
        custom_prompt=custom_prompt,   # ← add this param to generate_dataset if missing
    )
    return samples


def _collect_accepted(batch_output: dict) -> List[dict]:
    """
    Return only the accepted and rewritten samples from a Critic batch output.
    Rewritten samples use rewritten_output as their output field.
    """
    accepted = []
    for verdict in batch_output.get("verdicts", []):
        if verdict["verdict"] == "reject":
            continue
        sample = verdict.get("original_sample", {}).copy()
        if verdict["verdict"] == "rewrite" and verdict.get("rewritten_output"):
            sample["output"] = verdict["rewritten_output"]
        accepted.append(sample)
    return accepted
```

> **NOTE TO SESSION C:** The `_collect_accepted()` function assumes verdicts include
> an `"original_sample"` key. Check the actual Critic code. If it doesn't include it,
> you need to pass samples alongside verdicts. Simplest fix: zip the original samples list
> with the verdicts list (they are index-aligned).

---

## File 3: `modifai/agents/tests/test_pipeline_e2e.py`

```python
"""
End-to-end pipeline tests. All AWS calls are mocked.
These tests exercise the full Orchestrator → Critic → Curriculum loop.
"""
from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from modifai.agents.pipeline_loop import run_agentic_loop


# ── Shared fixtures ────────────────────────────────────────────────────────────

GOAL = "Build a Q&A bot for our customer support runbook"

DOC_METADATA = {
    "filename": "support_runbook.pdf",
    "page_count": 20,
    "domain": "customer support",
    "estimated_chunk_count": 40,
}

SAMPLE_CHUNKS = [
    "Step 1: Open the support portal. Step 2: Navigate to tickets. Step 3: Assign priority.",
    "Refund policy: customers may request refunds within 30 days of purchase.",
    "Escalation path: Tier 1 → Tier 2 → Manager. Use Slack #escalations channel.",
]

MOCK_STRATEGY = {
    "intent": "QA",
    "quality_threshold": 0.70,
    "samples_per_chunk": 4,
    "reasoning": "Support FAQ domain.",
}

MOCK_SAMPLES = [
    {"instruction": "What is step 1?", "input": SAMPLE_CHUNKS[0], "output": "Open the support portal."},
    {"instruction": "What is the refund window?", "input": SAMPLE_CHUNKS[1], "output": "30 days."},
]


def _orchestrator_response(strategy: dict) -> dict:
    return {
        "output": {
            "message": {
                "content": [
                    {"toolUse": {"toolUseId": "o1", "name": "set_pipeline_strategy", "input": strategy}}
                ]
            }
        }
    }


def _critic_batch_output(accept_pct: float) -> dict:
    total = len(MOCK_SAMPLES)
    accepted = int(total * accept_pct / 100)
    rejected = total - accepted
    verdicts = []
    for i, sample in enumerate(MOCK_SAMPLES):
        if i < accepted:
            verdicts.append({
                "verdict": "accept", "reason": "good", "rewritten_output": None,
                "original_sample": sample,
            })
        else:
            verdicts.append({
                "verdict": "reject", "reason": "too vague, missing steps",
                "rewritten_output": None, "original_sample": sample,
            })
    return {
        "verdicts": verdicts,
        "stats": {
            "total": total,
            "accepted": accepted,
            "rewritten": 0,
            "rejected": rejected,
            "accept_pct": accept_pct,
        },
    }


def _curriculum_response(gaps: list) -> dict:
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "c1",
                            "name": "analyze_curriculum",
                            "input": {
                                "gap_categories": gaps,
                                "targeted_prompt": (
                                    "Each answer must enumerate all steps. "
                                    "Name specific tools and systems."
                                ),
                                "priority_focus": gaps[0]["name"],
                            },
                        }
                    }
                ]
            }
        }
    }


GAP_CATEGORIES = [
    {
        "name": "lacks_steps",
        "description": "Skips intermediate steps.",
        "example_bad": "Restart server.",
        "example_good": "1. SSH in. 2. Run restart. 3. Check logs.",
    },
    {
        "name": "too_vague",
        "description": "No specific entity names.",
        "example_bad": "Use the tool.",
        "example_good": "Use Datadog.",
    },
    {
        "name": "factual_drift",
        "description": "Introduces unsupported facts.",
        "example_bad": "Uses Redis (not in source).",
        "example_good": "Uses the cache described in the doc.",
    },
]


# ── Test: all accepted on first pass ──────────────────────────────────────────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_all_accepted_first_pass_skips_curriculum(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        # Setup mocks
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=100.0)

        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        assert state["exit_reason"] == "all_accepted_first_pass"
        mock_curriculum_cls.return_value.run.assert_not_called()
        # Events: 1 orchestrator + 1 critic = 2
        assert len(state["events"]) == 2
        assert state["events"][0]["agent"] == "orchestrator"
        assert state["events"][1]["agent"] == "critic"
        assert state["curriculum_outputs"] == []


# ── Test: threshold met after 1 curriculum loop ───────────────────────────────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_loops_once_when_threshold_met_on_second_critic_pass(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY  # threshold=0.70

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        # First pass: 50% accept (below threshold 70%)
        # Second pass: 100% accept (above threshold)
        mock_critic.run_batch.side_effect = [
            _critic_batch_output(accept_pct=50.0),
            _critic_batch_output(accept_pct=100.0),
        ]

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = {
            "gap_categories": GAP_CATEGORIES,
            "targeted_prompt": "Each answer must list all steps.",
            "priority_focus": "lacks_steps",
        }
        mock_curriculum.extract_rejection_reasons = lambda b: ["too vague", "missing steps"]

        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        assert state["exit_reason"] == "threshold_met"
        assert state["iteration"] == 2
        assert len(state["curriculum_outputs"]) == 1
        # Events: orchestrator + critic(iter1) + curriculum(iter1) + critic(iter2) = 4
        assert len(state["events"]) == 4
        agents_in_order = [e["agent"] for e in state["events"]]
        assert agents_in_order == ["orchestrator", "critic", "curriculum", "critic"]


# ── Test: max iterations exhausted ────────────────────────────────────────────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_exits_after_max_iterations(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY  # threshold=0.70

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        # Always 50% — never meets 70% threshold
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=50.0)

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = {
            "gap_categories": GAP_CATEGORIES,
            "targeted_prompt": "Be more specific.",
            "priority_focus": "lacks_steps",
        }
        mock_curriculum.extract_rejection_reasons = lambda b: ["too vague"]

        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        assert state["exit_reason"] == "max_iterations"
        assert state["iteration"] == 3
        # Critic runs 3 times, Curriculum runs 2 times (no curriculum on last iteration)
        assert mock_critic.run_batch.call_count == 3
        assert len(state["curriculum_outputs"]) == 2


# ── Test: event log is written correctly ──────────────────────────────────────

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_event_log_written_to_disk(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MOCK_STRATEGY

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.return_value = _critic_batch_output(accept_pct=100.0)

        mock_gen.return_value = MOCK_SAMPLES

        run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        log_path = Path(event_log)
        assert log_path.exists(), "Event log file was not created"
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 2

        for line in lines:
            event = json.loads(line)
            assert "event_id" in event
            assert "timestamp" in event
            assert "agent" in event
            assert "decision" in event
            assert "iteration" in event
            assert event["agent"] in ("orchestrator", "critic", "curriculum")


# ── Test: critic score improves across iterations (integration sanity check) ──

@patch("modifai.agents.pipeline_loop._generate_samples")
@patch("modifai.agents.pipeline_loop.CurriculumAgent")
@patch("modifai.agents.pipeline_loop.CriticAgent")
@patch("modifai.agents.pipeline_loop.OrchestratorAgent")
def test_accept_pct_increases_across_iterations(
    mock_orch_cls, mock_critic_cls, mock_curriculum_cls, mock_gen
):
    """Verifies the loop runs Curriculum and shows improving accept_pct."""
    with tempfile.TemporaryDirectory() as tmpdir:
        event_log = f"{tmpdir}/events.jsonl"

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = {**MOCK_STRATEGY, "quality_threshold": 0.90}

        mock_critic = MagicMock()
        mock_critic_cls.return_value = mock_critic
        mock_critic.run_batch.side_effect = [
            _critic_batch_output(accept_pct=40.0),
            _critic_batch_output(accept_pct=65.0),
            _critic_batch_output(accept_pct=95.0),  # meets 90% threshold
        ]

        mock_curriculum = MagicMock()
        mock_curriculum_cls.return_value = mock_curriculum
        mock_curriculum.run.return_value = {
            "gap_categories": GAP_CATEGORIES,
            "targeted_prompt": "Be more specific.",
            "priority_focus": "lacks_steps",
        }
        mock_curriculum.extract_rejection_reasons = lambda b: ["vague", "missing steps"]

        mock_gen.return_value = MOCK_SAMPLES

        state = run_agentic_loop(
            goal=GOAL,
            doc_metadata=DOC_METADATA,
            chunks=SAMPLE_CHUNKS,
            max_iterations=3,
            event_log_path=event_log,
        )

        # Confirm it resolved before max iterations
        assert state["exit_reason"] == "threshold_met"
        critic_events = [e for e in state["events"] if e["agent"] == "critic"]
        assert len(critic_events) == 3
```

---

## Session C — Definition of Done

- [ ] `from modifai.agents.pipeline_loop import run_agentic_loop` works cleanly
- [ ] `from modifai.agents.logging_utils import AgentEventLogger` works cleanly
- [ ] All 5 E2E tests pass: `pytest modifai/agents/tests/test_pipeline_e2e.py -v`
- [ ] Event log is written as valid JSONL with one line per event
- [ ] `_generate_samples()` is wired to the actual `dataset_generation.py` (not a stub)
- [ ] `_collect_accepted()` correctly handles accept + rewrite verdicts
- [ ] Verify the Critic `run_batch()` interface matches what the schema says — fix any drift
- [ ] Full pipeline runs end-to-end on the demo PDF (non-mocked) at least once before handoff

---

---

# INTEGRATION CHECKLIST — run this before calling P2

After all three sessions are complete, one person runs through this checklist:

```bash
# 1. Install dependencies (if not already)
pip install boto3 pytest

# 2. Run all agent tests
pytest modifai/agents/tests/ -v

# 3. Smoke test with real AWS (use demo PDF)
python - <<'EOF'
from modifai.core.text_extraction import extract_text
from modifai.core.chunking import chunk_text
from modifai.agents.pipeline_loop import run_agentic_loop

# Use a small real PDF for the smoke test
text = extract_text("demo_doc.pdf")
chunks = chunk_text(text)[:5]  # Only 5 chunks to save cost

state = run_agentic_loop(
    goal="Generate a fine-tuning dataset for customer support Q&A",
    doc_metadata={
        "filename": "demo_doc.pdf",
        "page_count": 10,
        "domain": "customer support",
        "estimated_chunk_count": len(chunks),
    },
    chunks=chunks,
    max_iterations=3,
    event_log_path="smoke_test_events.jsonl",
)
print("Exit reason:", state["exit_reason"])
print("Final accept_pct:", state["final_stats"]["accept_pct"])
print("Events written:", len(state["events"]))
print("Curriculum loops:", len(state["curriculum_outputs"]))
EOF

# 4. Verify event log
python -c "
import json
with open('smoke_test_events.jsonl') as f:
    for line in f:
        e = json.loads(line)
        print(e['timestamp'], e['agent'], '|', e['decision'][:60])
"

# 5. Confirm schemas match P2's expectation
python -c "
from modifai.agents.schemas import OrchestratorOutput
print('OrchestratorOutput fields:', list(OrchestratorOutput.__annotations__.keys()))
# Should print: ['intent', 'quality_threshold', 'samples_per_chunk', 'reasoning']
"
```

## Things to tell P2 explicitly (today)

1. **OrchestratorOutput schema** — send the exact JSON:
   ```json
   {"intent": "QA|instruction|tutor", "quality_threshold": 0.72, "samples_per_chunk": 5, "reasoning": "..."}
   ```

2. **Event log format** — tell P3 the event log lives at the path passed to `run_agentic_loop()`;
   each line is one `AgentEvent` JSON object. Fields P3 needs: `agent`, `iteration`, `decision`,
   `reason`, `timestamp`, `data`.

3. **Loop iteration count** — always 1–3, never 0 for Critic/Curriculum events.
   Orchestrator is always `iteration=0`.

4. **Exit reasons** — `"threshold_met"` | `"max_iterations"` | `"all_accepted_first_pass"` —
   P3 uses these to show the final status badge.

---

# QUICK REFERENCE — Schema Keys

| Agent | Input keys | Output keys |
|-------|-----------|-------------|
| Orchestrator | `goal`, `doc_metadata` | `intent`, `quality_threshold`, `samples_per_chunk`, `reasoning` |
| Critic (single) | `instruction`, `input`, `output` | `verdict`, `reason`, `rewritten_output` |
| Critic (batch) | `[{instruction,input,output}]` | `verdicts[]`, `stats{total,accepted,rewritten,rejected,accept_pct}` |
| Curriculum | `rejection_reasons[]`, `strategy`, `iteration` | `gap_categories[]`, `targeted_prompt`, `priority_focus` |
| Event | — | `event_id`, `timestamp`, `agent`, `iteration`, `decision`, `reason`, `data` |

---

*Built for FAR AWAY 2026 Hackathon — Modifai · Agentic & Autonomous Systems track*
