"""Modifai agentic pipeline — Orchestrator, Critic, Curriculum, Training."""

from modifai.agents.orchestrator import OrchestratorAgent
from modifai.agents.curriculum import CurriculumAgent
from modifai.agents.critic import CriticAgent
from modifai.agents.training_agent import TrainingAgent
from modifai.agents.pipeline_loop import run_agentic_loop

__all__ = [
    "OrchestratorAgent",
    "CurriculumAgent",
    "CriticAgent",
    "TrainingAgent",
    "run_agentic_loop",
]
