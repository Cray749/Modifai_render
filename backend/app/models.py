"""
Pydantic request/response models.

Field names use snake_case — the frontend explicitly maps these
(e.g. p.created_at → project.createdAt in JS).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


# ── Project ─────────────────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=3)
    description: str | None = None
    mode: str = Field(..., pattern=r"^(dataset_only|finetune_only|dataset_and_finetune|full)$")
    intent: str | None = None
    base_model: str = "llama-3.1-8b"
    config: dict | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    mode: str
    intent: str | None = None
    base_model: str | None = None
    status: str
    config: dict | None = None
    created_at: str
    updated_at: str


class StartPipelineRequest(BaseModel):
    config: dict | None = None
    uploaded_filenames: list[str] = []


class UploadUrlResponse(BaseModel):
    presigned_url: str
    s3_key: str


# ── Status / Logs / Results ─────────────────────────────────────────────────────

class PipelineStatusResponse(BaseModel):
    project_status: str
    pipeline_status: str  # NOT_STARTED | RUNNING | SUCCEEDED | FAILED


class LogEntry(BaseModel):
    id: str
    timestamp: str
    type: str
    label: str
    summary: str | None = None
    details: Any = None


class LogsResponse(BaseModel):
    logs: list[LogEntry] = []


class StepResult(BaseModel):
    """Flexible dict for per-step result data."""
    pass


class PipelineResultsResponse(BaseModel):
    dataset_download_url: str | None = None
    model_endpoint_url: str | None = None
    training_metrics: dict | None = None
    step_results: dict[str, Any] = {}
    error: dict | None = None
    virtual_mind_agents: list[dict] | None = None
    virtual_mind_automations: list[dict] | None = None
    n8n_url: str | None = None


# ── Evaluate ────────────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    text_sample: str = Field(..., min_length=10)
    intent: str = Field(..., min_length=3)


class EvaluateResponse(BaseModel):
    score: float
    explanation: str


# ── Compare ─────────────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    project_id: str
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = None


class ModelResult(BaseModel):
    response: str
    latency_ms: int
    model_id: str
    error: str | None = None


class CompareResponse(BaseModel):
    base_model: ModelResult
    fine_tuned: ModelResult


# ── Dataset ─────────────────────────────────────────────────────────────────────

class DatasetExample(BaseModel):
    instruction: str
    response: str
    confidence: float | None = None


class DatasetResponse(BaseModel):
    dataset: list[DatasetExample] = []
    total: int = 0


class DatasetSearchResponse(BaseModel):
    results: list[DatasetExample] = []
    total: int = 0


class DatasetUpdateRequest(BaseModel):
    instruction: str
    response: str


class DatasetExportResponse(BaseModel):
    download_url: str
