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
        # NOTE: ap-south-1 not confirmed for Nova Micro — falling back to us-east-1
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
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
                          Must have {"verdicts": [...]} key.

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
