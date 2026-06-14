"""
Evaluate router — POST /evaluate/

LLM-powered data quality evaluation for the NewProjectPage.
"""

import logging

from fastapi import APIRouter

from app.models import EvaluateRequest, EvaluateResponse
from app.services import llm_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["evaluate"])


@router.post("/evaluate/", response_model=EvaluateResponse)
async def evaluate_data_quality(req: EvaluateRequest):
    """
    Evaluate a text sample for suitability as fine-tuning data.

    Used on the NewProjectPage's "Run Automated Evaluation" step
    to score uploaded document text before starting the pipeline.
    """
    result = llm_service.evaluate_data_quality(req.text_sample, req.intent)
    return result
