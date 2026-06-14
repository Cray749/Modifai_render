import os
import requests
import logging

logger = logging.getLogger(__name__)

def deploy_workflow_to_n8n(workflow_json):
    """
    Authenticate with n8n API, create workflow, return workflow URL.
    """
    n8n_host = os.environ.get("N8N_HOST", "http://localhost:5678").rstrip("/")
    api_key = os.environ.get("N8N_API_KEY", "")
    
    url = f"{n8n_host}/api/v1/workflows"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-N8N-API-KEY"] = api_key
        
    try:
        response = requests.post(url, json=workflow_json, headers=headers)
        response.raise_for_status()
        data = response.json()
        workflow_id = data.get("id")
        
        # Return the editor URL (adjust if standard URL pattern differs)
        editor_url = f"{n8n_host}/workflow/{workflow_id}"
        return editor_url
    except requests.exceptions.RequestException as e:
        logger.error("Failed to deploy workflow to n8n: %s", e)
        if hasattr(e, 'response') and e.response is not None:
            logger.error("Response content: %s", e.response.text)
        raise
