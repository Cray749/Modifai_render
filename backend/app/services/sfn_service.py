"""
Step Functions service — start executions, poll status, fetch logs/results.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

from app.config import settings
from app.services import s3_service

logger = logging.getLogger(__name__)

_client = None

# ── Frontend mode → State Machine mode ──────────────────────────────────────────
MODE_MAP = {
    "dataset_only": "DATASET_ONLY",
    "finetune_only": "FINETUNE_ONLY",
    "dataset_and_finetune": "DATASET_AND_FINETUNE",
    "full": "DATASET_AND_FINETUNE_AND_DEPLOY",
}


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("stepfunctions", region_name=settings.AWS_REGION)
    return _client


# ── Start Execution ─────────────────────────────────────────────────────────────

def start_execution(project: dict) -> str:
    """
    Start a Step Functions execution for the given project.

    Args:
        project: Full project dict from the database.

    Returns:
        The execution ARN.
    """
    config = project.get("config", {}) or {}
    uploaded = project.get("uploaded_filenames", []) or []

    sfn_input = {
        "project_id": project["id"],
        "s3_bucket": settings.S3_BUCKET,
        "s3_prefix": project.get("s3_prefix", f"projects/{project['id']}/"),
        "raw_file_keys": [
            f"projects/{project['id']}/raw/{name}" for name in uploaded
        ],
        "document_s3_uris": [
            f"s3://{settings.S3_BUCKET}/projects/{project['id']}/raw/{name}" for name in uploaded
        ],
        "pipeline_mode": MODE_MAP.get(project["mode"], "DATASET_ONLY"),
        "intent": project.get("intent", "question-answering"),
        "config": {
            "samples_per_chunk": config.get(
                "samples_per_chunk", settings.DEFAULT_SAMPLES_PER_CHUNK
            ),
            "quality_threshold": config.get(
                "quality_threshold", settings.DEFAULT_QUALITY_THRESHOLD
            ),
            "openrouter_api_key": settings.OPENROUTER_API_KEY,
        },
        "base_model": project.get("base_model", "llama-3.1-8b"),
    }

    execution_name = f"modifai-{project['id'][:8]}-{uuid.uuid4().hex[:8]}"

    try:
        response = _get_client().start_execution(
            stateMachineArn=settings.STATE_MACHINE_ARN,
            name=execution_name,
            input=json.dumps(sfn_input),
        )
        arn = response["executionArn"]
        logger.info("Started execution %s for project %s", arn, project["id"])
        return arn
    except ClientError as e:
        logger.error("Failed to start execution: %s", e)
        raise

def redrive_execution(execution_arn: str) -> str:
    """
    Redrive a failed Step Functions execution from the state it failed at.
    """
    try:
        _get_client().redrive_execution(executionArn=execution_arn)
        logger.info("Redriving execution %s", execution_arn)
        return execution_arn
    except ClientError as e:
        logger.error("Failed to redrive execution: %s", e)
        raise


# ── Get Status ──────────────────────────────────────────────────────────────────

def get_execution_status(execution_arn: str | None) -> dict:
    """
    Get the current status of a Step Functions execution.

    Returns:
        {
            "pipeline_status": "NOT_STARTED" | "RUNNING" | "SUCCEEDED" | "FAILED",
            "output": dict | None
        }
    """
    if not execution_arn:
        return {"pipeline_status": "NOT_STARTED", "output": None}

    try:
        response = _get_client().describe_execution(executionArn=execution_arn)
        status = response.get("status", "RUNNING")

        # Map Step Functions status to our status values
        status_map = {
            "RUNNING": "RUNNING",
            "SUCCEEDED": "SUCCEEDED",
            "FAILED": "FAILED",
            "TIMED_OUT": "FAILED",
            "ABORTED": "FAILED",
        }

        output = None
        if response.get("output"):
            try:
                output = json.loads(response["output"])
            except json.JSONDecodeError:
                output = {"raw": response["output"]}

        return {
            "pipeline_status": status_map.get(status, "RUNNING"),
            "output": output,
        }
    except ClientError as e:
        logger.error("Failed to get execution status: %s", e)
        return {"pipeline_status": "NOT_STARTED", "output": None}


# ── Get Logs ────────────────────────────────────────────────────────────────────

def get_execution_logs(execution_arn: str | None) -> list[dict]:
    """
    Fetch the execution history and parse it into the format the frontend expects.

    Returns a list of:
        {
            "id": str,
            "timestamp": str (ISO 8601),
            "type": str,
            "label": str,
            "summary": str | None,
            "details": dict | None,
        }
    """
    if not execution_arn:
        return []

    try:
        client = _get_client()
        events = []

        # Paginate through ALL events (AWS caps each page at 1000, but we may have >100)
        next_token = None
        while True:
            kwargs = {
                "executionArn": execution_arn,
                "reverseOrder": True,
                "maxResults": 1000,
            }
            if next_token:
                kwargs["nextToken"] = next_token

            response = client.get_execution_history(**kwargs)

            for event in response.get("events", []):
                try:
                    parsed = _parse_sfn_event(event)
                    if parsed:
                        events.append(parsed)
                except Exception as parse_err:
                    logger.warning("Skipping unparseable event: %s", parse_err)

            next_token = response.get("nextToken")
            if not next_token:
                break

        return events
    except ClientError as e:
        logger.error("Failed to get execution logs: %s", e)
        return []
    except Exception as e:
        logger.error("Unexpected error fetching logs: %s", e)
        return []


def _parse_sfn_event(event: dict) -> dict | None:
    """Parse a single Step Functions history event into our log format."""
    event_type = event.get("type", "")
    timestamp = event.get("timestamp")

    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.isoformat()
    else:
        timestamp_str = str(timestamp)

    event_id = str(event.get("id", uuid.uuid4().hex[:8]))

    # Extract label from different event types
    label = event_type
    summary = None
    details = None

    # State transition events
    if "stateEnteredEventDetails" in event:
        state_details = event["stateEnteredEventDetails"]
        state_name = state_details.get("name", "Unknown")
        label = f"Entered: {state_name}"
        if state_details.get("input"):
            try:
                details = json.loads(state_details["input"])
            except (json.JSONDecodeError, TypeError):
                pass

    elif "stateExitedEventDetails" in event:
        state_details = event["stateExitedEventDetails"]
        state_name = state_details.get("name", "Unknown")
        label = f"Exited: {state_name}"
        if state_details.get("output"):
            try:
                output = json.loads(state_details["output"])
                details = output
                # Extract a summary from common output fields
                if isinstance(output, dict):
                    if "status" in output:
                        summary = f"Status: {output['status']}"
                    elif "action" in output:
                        summary = f"Action: {output['action']}"
            except (json.JSONDecodeError, TypeError):
                pass

    elif "executionStartedEventDetails" in event:
        label = "Execution Started"
        event_type = "ExecutionStarted"

    elif "executionSucceededEventDetails" in event:
        label = "Execution Succeeded"
        event_type = "ExecutionSucceeded"
        summary = "Pipeline completed successfully"

    elif "executionFailedEventDetails" in event:
        label = "Execution Failed"
        event_type = "ExecutionFailed"
        fail_details = event["executionFailedEventDetails"]
        summary = fail_details.get("cause", "Unknown error")
        details = {
            "error": fail_details.get("error"),
            "cause": fail_details.get("cause"),
        }

    elif "taskFailedEventDetails" in event:
        fail_details = event["taskFailedEventDetails"]
        label = "Task Failed"
        summary = fail_details.get("cause", "Task failed")

    elif "lambdaFunctionFailedEventDetails" in event:
        fail_details = event["lambdaFunctionFailedEventDetails"]
        label = "Lambda Failed"
        summary = fail_details.get("cause", "Lambda function failed")

    else:
        # Skip noisy internal events
        skip_types = [
            "TaskStateAborted", "MapIterationStarted", "MapIterationSucceeded",
            "MapStateStarted", "MapStateSucceeded",
        ]
        if event_type in skip_types:
            return None

    return {
        "id": event_id,
        "timestamp": timestamp_str,
        "type": event_type,
        "label": label,
        "summary": summary,
        "details": details,
    }


# ── Get Results ─────────────────────────────────────────────────────────────────

def get_execution_results(execution_arn: str | None, project_id: str) -> dict:
    """
    Parse the Step Functions execution output into the frontend's expected shape.
    Also generates presigned download URLs for artifacts.

    Returns:
        {
            "dataset_download_url": str | None,
            "model_endpoint_url": str | None,
            "training_metrics": dict | None,
            "step_results": dict,
            "error": dict | None,
        }
    """
    result = {
        "dataset_download_url": None,
        "model_endpoint_url": None,
        "training_metrics": None,
        "step_results": {},
        "error": None,
    }

    if not execution_arn:
        return result

    status_data = get_execution_status(execution_arn)
    output = status_data.get("output")

    # Generate dataset download URL if the file exists
    dataset_key = f"projects/{project_id}/dataset/clean_dataset.jsonl"
    if s3_service.check_file_exists(dataset_key):
        try:
            result["dataset_download_url"] = s3_service.generate_presigned_download_url(
                dataset_key
            )
        except Exception:
            pass

    if status_data["pipeline_status"] == "FAILED":
        # Try to extract error info from logs
        logs = get_execution_logs(execution_arn)
        for log in logs:
            if log["type"] in ("ExecutionFailed", "TaskFailed", "LambdaFunctionFailed"):
                result["error"] = {
                    "error": log.get("summary", "Pipeline failed"),
                    "cause": log.get("details", {}).get("cause", "Check logs for details"),
                }
                break

    if output and isinstance(output, dict):
        # Extract step results from the Step Functions output
        # The state machine output accumulates results from each step
        step_results = {}

        if "chunks" in output:
            chunks = output["chunks"]
            step_results["chunking"] = {
                "chunk_count": len(chunks.get("chunk_uris", [])),
                "total_words": chunks.get("total_words", 0),
            }

        if "dataset" in output:
            dataset = output["dataset"]
            if isinstance(dataset, list):
                total_examples = sum(
                    len(chunk_result.get("samples", []))
                    for chunk_result in dataset
                    if isinstance(chunk_result, dict)
                )
                step_results["generation"] = {
                    "example_count": total_examples,
                    "chunks_processed": len(dataset),
                    "chunks_failed": 0,
                }

        if "dataset_evaluation" in output:
            eval_data = output["dataset_evaluation"]
            step_results["quality_control"] = {
                "total_input": eval_data.get("total_input", 0),
                "kept": eval_data.get("kept", 0),
                "discarded": eval_data.get("discarded", 0),
                "duplicates_removed": eval_data.get("duplicates_removed", 0),
                "threshold": eval_data.get("threshold", 0.7),
            }

        if "job_info" in output:
            job_info = output["job_info"]
            result["training_metrics"] = {
                "job_name": job_info.get("job_name"),
                "duration_min": job_info.get("duration_min"),
                "final_loss": job_info.get("final_loss"),
            }
            step_results["fine_tuning"] = result["training_metrics"]

        if "deployment" in output:
            deploy = output["deployment"]
            result["model_endpoint_url"] = deploy.get("endpoint_url")
            step_results["deployment"] = {
                "endpoint_url": deploy.get("endpoint_url"),
            }

        # Add upload info from the input files
        if "raw_file_keys" in output:
            step_results["upload"] = {
                "raw_file_keys": output["raw_file_keys"],
            }

        result["step_results"] = step_results
        
        # Inject Virtual Mind Integration results when successful
        if status_data["pipeline_status"] == "SUCCEEDED":
            try:
                s3 = s3_service._get_client()
                report_key = f"projects/{project_id}/virtual_mind_report.json"
                resp = s3.get_object(Bucket=settings.S3_BUCKET, Key=report_key)
                report_data = json.loads(resp["Body"].read().decode("utf-8"))
                
                # Extract Agents from report
                discovered_agents = report_data.get("virtual_mind", {}).get("agents", [])
                result["virtual_mind_agents"] = discovered_agents if discovered_agents else []
                
                # Ensure agents are running (survives backend restarts)
                from app.services import virtual_mind_service
                if discovered_agents:
                    virtual_mind_service.ensure_agents_running(discovered_agents, report_data.get("knowledge_analysis", {}))
                
                # Extract Automations from report
                automations = report_data.get("automation_catalog", {}).get("automation_opportunities", [])
                result["virtual_mind_automations"] = automations if automations else []
                
                result["n8n_url"] = "http://localhost:5678"
            except s3_service.ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    logger.warning(f"Virtual Mind report not found for {project_id}. Background task might have failed or is still running.")
                else:
                    logger.error(f"Failed to fetch Virtual Mind report: {e}")
                result["virtual_mind_agents"] = []
                result["virtual_mind_automations"] = []
            except Exception as e:
                logger.error(f"Error parsing Virtual Mind report for {project_id}: {e}")
                result["virtual_mind_agents"] = []
                result["virtual_mind_automations"] = []

    return result
