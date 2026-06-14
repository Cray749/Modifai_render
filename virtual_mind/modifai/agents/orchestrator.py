"""
OrchestratorAgent — decides pipeline strategy from goal + doc metadata.

Uses Bedrock converse API with forced tool-use to guarantee structured JSON output.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from modifai.core.llm_provider import get_llm_provider
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
        self.max_retries = max_retries
        self.provider = get_llm_provider()

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
                schema = _TOOL_SPEC["toolSpec"]["inputSchema"]["json"]
                tool_name = _TOOL_SPEC["toolSpec"]["name"]
                strategy = self.provider.generate(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_message,
                    response_schema=schema,
                    tool_name=tool_name,
                )
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
