"""
Compare router — POST /compare/

Side-by-side base model vs fine-tuned model inference for the ModelComparisonPage.
"""

import logging

from fastapi import APIRouter

from app.models import CompareRequest, CompareResponse
from app.services import llm_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["compare"])


@router.post("/compare/", response_model=CompareResponse)
async def compare_models(req: CompareRequest):
    """
    Run inference on both base model and fine-tuned model side-by-side.

    Measures latency for each and returns the responses for comparison
    on the ModelComparisonPage.
    """
    result = llm_service.compare_models(
        project_id=req.project_id,
        prompt=req.prompt,
        system_prompt=req.system_prompt,
    )
    return result
