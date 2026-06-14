"""
KnowledgeAgent — extracts organizational intelligence from document chunks.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, cast

from modifai.core.llm_provider import get_llm_provider

from modifai.agents.schemas import (
    DocMetadata,
    OrchestratorOutput,
    KnowledgeAnalysisOutput,
)

logger = logging.getLogger(__name__)

# ── Bedrock tool definition ────────────────────────────────────────────────────

_TOOL_SPEC = {
    "toolSpec": {
        "name": "analyze_knowledge",
        "description": (
            "Analyze document chunks to extract organizational intelligence including "
            "knowledge domains, expertise areas, departments, key concepts, and workflow candidates."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "knowledge_summary": {
                        "type": "string",
                        "description": "A compressed paragraph summarizing the organizational intelligence found in the chunks."
                    },
                    "domains": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "evidence": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "confidence": {"type": "number"}
                            },
                            "required": ["name", "description", "evidence", "confidence"]
                        }
                    },
                    "expertise": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["name", "confidence"]
                        }
                    },

                    "key_concepts": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "workflows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "steps": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "confidence": {"type": "number"}
                            },
                            "required": ["name", "steps", "confidence"]
                        }
                    }
                },
                "required": ["knowledge_summary", "domains", "expertise", "key_concepts", "workflows"]
            }
        }
    }
}

_SYSTEM_PROMPT = """\
You are the Knowledge Analysis agent for Modifai, an automated intelligence extraction platform.
Your job is to analyze the provided document chunks and discover organizational intelligence.

You must discover and structure:
1. Knowledge Summary (a compressed summary of the intelligence found)
2. Knowledge Domains with evidence phrases and a confidence score 0.0-1.0 (e.g. Human Resources)
3. Expertise Areas with a confidence score 0.0-1.0 (e.g. Employee Onboarding: 0.91)
4. Key Concepts
5. Workflow Candidates (processes with distinct steps) with a confidence score 0.0-1.0

You MUST call the analyze_knowledge tool to output the results deterministically. Output nothing else.
"""

class KnowledgeAgent:
    """
    Analyzes document chunks to extract structured knowledge.

    Usage:
        agent = KnowledgeAgent()
        output = agent.run(
            chunks=["text chunk 1", "text chunk 2"],
            doc_metadata=metadata,
            strategy=orchestrator_strategy
        )
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

    def run(
        self,
        chunks: List[str],
        doc_metadata: DocMetadata,
        strategy: Optional[OrchestratorOutput] = None,
    ) -> KnowledgeAnalysisOutput:
        """
        Analyze document chunks and return structured intelligence.

        Args:
            chunks: List of raw text chunks from the document.
            doc_metadata: DocMetadata TypedDict.
            strategy: Optional OrchestratorOutput with pipeline strategy intent.

        Returns:
            KnowledgeAnalysisOutput containing domains, expertise, workflows, etc.
        """
        user_message = self._build_user_message(chunks, doc_metadata, strategy)
        attempt = 0

        while attempt <= self.max_retries:
            try:
                schema = _TOOL_SPEC["toolSpec"]["inputSchema"]["json"]
                tool_name = _TOOL_SPEC["toolSpec"]["name"]
                output = self.provider.generate(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_message,
                    response_schema=schema,
                    tool_name=tool_name,
                )
                self._validate(output)
                logger.info(
                    "Knowledge Analysis discovered %d domains, %d expertise areas, %d workflows.",
                    len(output.get("domains", [])),
                    len(output.get("expertise", [])),
                    len(output.get("workflows", []))
                )
                return cast(KnowledgeAnalysisOutput, output)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise ValueError(
                        f"KnowledgeAgent failed to produce valid output after "
                        f"{self.max_retries + 1} attempt(s): {exc}"
                    ) from exc
                logger.warning(
                    "KnowledgeAgent output malformed (attempt %d/%d): %s — retrying",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _build_user_message(
        self, 
        chunks: List[str], 
        doc_metadata: DocMetadata,
        strategy: Optional[OrchestratorOutput]
    ) -> str:
        # Join chunks, keeping in mind model context limits.
        # Nova Micro has a 128k context window, which is sufficient for most chunks.
        joined_chunks = "\n\n--- CHUNK ---\n\n".join(chunks)
        
        msg = (
            f"Document metadata:\n"
            f"  filename: {doc_metadata['filename']}\n"
            f"  domain: {doc_metadata['domain']}\n\n"
        )
        if strategy:
            msg += f"Pipeline strategy intent: {strategy['intent']}\n\n"
            
        msg += f"Document content:\n{joined_chunks}\n\n"
        msg += "Analyze the content and call analyze_knowledge."
        return msg

    def _validate(self, output: dict) -> None:
        """Type-check the tool output arrays to ensure safety."""
        if not isinstance(output.get("knowledge_summary"), str):
            raise ValueError("knowledge_summary must be a string")
        if not isinstance(output.get("domains"), list):
            raise ValueError("domains must be a list")
        if not isinstance(output.get("expertise"), list):
            raise ValueError("expertise must be a list")
        if not isinstance(output.get("key_concepts"), list):
            raise ValueError("key_concepts must be a list")
        if not isinstance(output.get("workflows"), list):
            raise ValueError("workflows must be a list")

        for domain in output.get("domains", []):
            if not isinstance(domain.get("evidence"), list):
                raise ValueError("domain evidence must be a list")
            if not (0.0 <= domain.get("confidence", -1) <= 1.0):
                raise ValueError("domain confidence must be between 0.0 and 1.0")
        
        for wf in output.get("workflows", []):
            if not (0.0 <= wf.get("confidence", -1) <= 1.0):
                raise ValueError("workflow confidence must be between 0.0 and 1.0")
        
        for exp in output.get("expertise", []):
            if not (0.0 <= exp.get("confidence", -1) <= 1.0):
                raise ValueError("expertise confidence must be between 0.0 and 1.0")
