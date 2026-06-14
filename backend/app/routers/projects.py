"""
Projects router — all /projects/* endpoints.

Handles project CRUD, file uploads, pipeline execution, status polling,
logs, results, and dataset management.
"""

import logging

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks

from app import database as db
from app.models import (
    CreateProjectRequest,
    ProjectResponse,
    StartPipelineRequest,
    UploadUrlResponse,
    PipelineStatusResponse,
    LogsResponse,
    PipelineResultsResponse,
    DatasetResponse,
    DatasetSearchResponse,
    DatasetUpdateRequest,
    DatasetExportResponse,
)
from app.services import s3_service, sfn_service, dataset_service, virtual_mind_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects", tags=["projects"])


# ── Project CRUD ────────────────────────────────────────────────────────────────

@router.get("/", response_model=list[ProjectResponse])
async def list_projects():
    """List all projects ordered by created_at DESC."""
    projects = db.list_projects()
    return projects


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(req: CreateProjectRequest):
    """Create a new project."""
    project = db.create_project(
        name=req.name,
        description=req.description,
        mode=req.mode,
        intent=req.intent,
        base_model=req.base_model,
        config=req.config,
    )
    logger.info("Created project %s: %s", project["id"], project["name"])
    return project


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Get a single project by ID."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str):
    """Delete a project and its S3 files."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Clean up S3 files
    try:
        s3_service.delete_project_files(project_id)
    except Exception as e:
        logger.warning("Failed to delete S3 files for project %s: %s", project_id, e)

    db.delete_project(project_id)
    logger.info("Deleted project %s", project_id)


# ── File Upload ─────────────────────────────────────────────────────────────────

@router.post("/{project_id}/upload-url", response_model=UploadUrlResponse)
async def get_upload_url(project_id: str, filename: str = Query(...), content_type: str = Query("application/octet-stream")):
    """Generate a presigned S3 PUT URL for file upload."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = s3_service.generate_presigned_upload_url(project_id, filename, content_type)
    return result


# ── Pipeline Execution ──────────────────────────────────────────────────────────

@router.post("/{project_id}/start")
async def start_pipeline(project_id: str, req: StartPipelineRequest, background_tasks: BackgroundTasks):
    """
    Start the pipeline execution for a project.

    Updates the project with the uploaded filenames and config,
    then kicks off a Step Functions execution.
    """
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline is already running")

    # Update project with start parameters
    update_fields = {"status": "running"}
    if req.uploaded_filenames:
        update_fields["uploaded_filenames"] = req.uploaded_filenames
    if req.config:
        merged_config = {**(project.get("config") or {}), **req.config}
        update_fields["config"] = merged_config

    updated_project = db.update_project(project_id, **update_fields)

    # Start or Redrive Step Functions execution
    try:
        execution_arn = project.get("execution_arn")
        redriven = False
        
        # Try to redrive if the previous execution failed
        if project["status"] == "failed" and execution_arn:
            try:
                execution_arn = sfn_service.redrive_execution(execution_arn)
                redriven = True
                logger.info("Redriven pipeline for project %s: %s", project_id, execution_arn)
            except Exception as redrive_e:
                logger.warning("Redrive failed for %s, falling back to fresh start: %s", project_id, redrive_e)
        
        if not redriven:
            execution_arn = sfn_service.start_execution(updated_project)
            db.update_project(project_id, execution_arn=execution_arn)
            logger.info("Started fresh pipeline for project %s: %s", project_id, execution_arn)
            
            # Start the Virtual Mind background task
            # Assume the first uploaded file is the primary document
            if updated_project.get("uploaded_filenames"):
                first_file = updated_project["uploaded_filenames"][0]
                s3_key = f"projects/{project_id}/raw/{first_file}"
                background_tasks.add_task(virtual_mind_service.generate_virtual_mind_background, project_id, s3_key)
            
        return {
            "message": "Pipeline redriven" if redriven else "Pipeline started",
            "execution_arn": execution_arn,
        }
    except Exception as e:
        db.update_project(project_id, status="failed")
        logger.error("Failed to start/redrive pipeline for project %s: %s", project_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to start/redrive pipeline: {str(e)}")


# ── Status Polling ──────────────────────────────────────────────────────────────

@router.get("/{project_id}/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(project_id: str):
    """
    Get the current pipeline status.

    Polls Step Functions and syncs the project status in the DB.
    """
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    execution_arn = project.get("execution_arn")
    sfn_status = sfn_service.get_execution_status(execution_arn)
    pipeline_status = sfn_status["pipeline_status"]

    # Sync project status based on pipeline status
    if pipeline_status == "SUCCEEDED" and project["status"] != "completed":
        db.update_project(project_id, status="completed")
    elif pipeline_status == "FAILED" and project["status"] != "failed":
        db.update_project(project_id, status="failed")

    return {
        "project_status": project["status"] if pipeline_status == "NOT_STARTED" else (
            "completed" if pipeline_status == "SUCCEEDED"
            else "failed" if pipeline_status == "FAILED"
            else "running"
        ),
        "pipeline_status": pipeline_status,
    }


# ── Logs ────────────────────────────────────────────────────────────────────────

@router.get("/{project_id}/logs", response_model=LogsResponse)
async def get_pipeline_logs(project_id: str):
    """Get execution logs for the pipeline."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    logs = sfn_service.get_execution_logs(project.get("execution_arn"))
    return {"logs": logs}


# ── Results ─────────────────────────────────────────────────────────────────────

@router.get("/{project_id}/results", response_model=PipelineResultsResponse)
async def get_pipeline_results(project_id: str):
    """Get pipeline execution results including download URLs and metrics."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    results = sfn_service.get_execution_results(
        project.get("execution_arn"), project_id
    )
    return results


# ── Dataset Management ──────────────────────────────────────────────────────────

@router.get("/{project_id}/dataset", response_model=DatasetResponse)
async def get_dataset(project_id: str):
    """Get the generated training dataset."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return dataset_service.get_dataset(project_id)


@router.get("/{project_id}/dataset/search", response_model=DatasetSearchResponse)
async def search_dataset(project_id: str, q: str = Query(...)):
    """Search within the dataset examples."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return dataset_service.search_dataset(project_id, q)


@router.put("/{project_id}/dataset/{index}")
async def update_dataset_example(project_id: str, index: int, req: DatasetUpdateRequest):
    """Update a single training example by index."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        dataset_service.update_example(project_id, index, req.instruction, req.response)
        return {"message": "Example updated"}
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{project_id}/dataset/{index}")
async def delete_dataset_example(project_id: str, index: int):
    """Delete a single training example by index."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        dataset_service.delete_example(project_id, index)
        return {"message": "Example deleted"}
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{project_id}/dataset/export", response_model=DatasetExportResponse)
async def export_dataset(project_id: str):
    """Generate a presigned download URL for the full JSONL dataset."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        return dataset_service.export_dataset(project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
