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
            agent:     "orchestrator" | "critic" | "curriculum"
            iteration: 0 for orchestrator (pre-loop), 1–3 for loop agents
            decision:  Human-readable one-liner (shown in P3 dashboard)
            data:      Full agent output payload dict
            reason:    Optional explanation (shown in P3 dashboard)

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
