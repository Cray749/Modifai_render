"""
SSE streaming router — GET /projects/{id}/stream

Pushes real-time pipeline progress events to the frontend
via Server-Sent Events. The frontend's subscribeToStream()
connects here.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app import database as db
from app.services import sfn_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])

# Polling interval in seconds for Step Functions status checks
_POLL_INTERVAL = 3


@router.get("/projects/{project_id}/stream")
async def stream_pipeline(project_id: str):
    """
    SSE endpoint for real-time pipeline progress.

    Polls Step Functions every few seconds and pushes status events.
    The connection stays open until the pipeline completes or fails.
    """
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    execution_arn = project.get("execution_arn")
    if not execution_arn:
        raise HTTPException(status_code=400, detail="No pipeline execution found")

    return EventSourceResponse(
        _event_generator(project_id, execution_arn),
        media_type="text/event-stream",
    )


async def _event_generator(project_id: str, execution_arn: str):
    """
    Async generator that polls Step Functions and yields SSE events.

    Event types:
    - status: { step, status, progress }
    - log: { message, timestamp }
    - complete: { results }
    - error: { message }
    """
    last_log_count = 0
    terminal_states = ("SUCCEEDED", "FAILED")

    try:
        while True:
            # Check current status
            status_data = sfn_service.get_execution_status(execution_arn)
            pipeline_status = status_data["pipeline_status"]

            # Yield status event
            yield {
                "event": "status",
                "data": json.dumps({
                    "type": "status",
                    "pipeline_status": pipeline_status,
                    "project_id": project_id,
                }),
            }

            # Fetch any new logs
            logs = sfn_service.get_execution_logs(execution_arn)
            if len(logs) > last_log_count:
                new_logs = logs[:len(logs) - last_log_count]  # logs are reverse-order
                for log_entry in reversed(new_logs):
                    yield {
                        "event": "log",
                        "data": json.dumps({
                            "type": "log",
                            "message": log_entry.get("label", ""),
                            "timestamp": log_entry.get("timestamp", ""),
                            "details": log_entry.get("summary"),
                        }),
                    }
                last_log_count = len(logs)

            # If pipeline is done, send final event and close
            if pipeline_status in terminal_states:
                if pipeline_status == "SUCCEEDED":
                    results = sfn_service.get_execution_results(execution_arn, project_id)
                    # Sync DB status
                    db.update_project(project_id, status="completed")
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "type": "complete",
                            "results": results,
                        }),
                    }
                else:
                    db.update_project(project_id, status="failed")
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "type": "error",
                            "message": "Pipeline execution failed",
                        }),
                    }
                break

            await asyncio.sleep(_POLL_INTERVAL)

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled for project %s", project_id)
    except Exception as e:
        logger.error("SSE stream error for project %s: %s", project_id, e)
        yield {
            "event": "error",
            "data": json.dumps({
                "type": "error",
                "message": str(e),
            }),
        }
