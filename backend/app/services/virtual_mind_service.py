"""
Virtual Mind Service
Dynamically imports the modifai package and runs the Virtual Mind engine asynchronously.
"""

import sys
import os
import json
import logging
import tempfile
import urllib.request
import subprocess
import socket
import re
from datetime import datetime, timezone

from app.services.s3_service import _get_client
from app.config import settings

logger = logging.getLogger(__name__)

# Add the virtual_mind directory to the python path so we can import modifai
VIRTUAL_MIND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../virtual_mind"))
if VIRTUAL_MIND_PATH not in sys.path:
    sys.path.append(VIRTUAL_MIND_PATH)

from modifai.agents.knowledge_agent import KnowledgeAgent
from modifai.agents.agent_discovery import AgentDiscoveryAgent
from modifai.agents.virtual_mind_builder import VirtualMindBuilder
from modifai.agents.automation_discovery import AutomationDiscoveryAgent
from modifai.generators.agent_generator import generate_agent_package
from pypdf import PdfReader

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def ensure_agents_running(agents: list, knowledge: dict):
    """
    Checks if the agents defined in the report are actively running on their ports.
    If not (e.g. after a server restart), respawns them on the SAME port so URLs remain valid.
    """
    script_path = os.path.join(os.path.dirname(__file__), "run_agent.py")
    
    for agent in agents:
        endpoint = agent.get("endpoint")
        if not endpoint:
            continue
            
        match = re.search(r':(\d+)/', endpoint)
        if not match:
            continue
            
        port = int(match.group(1))
        
        # Check if port is alive
        is_active = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                is_active = True
                
        if not is_active:
            name = agent.get("name", "Generated Agent")
            logger.info(f"Respawning inactive agent {name} on port {port}...")
            
            agent_id = re.sub(r'[^a-zA-Z0-9]+', '-', name).strip('-').lower()
            package = generate_agent_package(agent, knowledge)
            
            package_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8')
            json.dump(package, package_tmp)
            package_tmp.close()
            
            subprocess.Popen([
                sys.executable, script_path, str(port), agent_id, package_tmp.name
            ])

def generate_virtual_mind_background(project_id: str, s3_key: str):
    """
    Background task to extract text, run Virtual Mind agents, and save the report to S3.
    """
    logger.info(f"Starting Virtual Mind generation for project {project_id} (key: {s3_key})")
    
    s3_client = _get_client()
    
    # 1. Download PDF to temp file
    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_pdf_path = tmp.name
            
        s3_client.download_file(settings.S3_BUCKET, s3_key, temp_pdf_path)
        logger.info(f"Downloaded PDF for {project_id} to temp path")
        
        # 2. Extract Text
        reader = PdfReader(temp_pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
        logger.info(f"Extracted {len(chunks)} chunks from {len(reader.pages)} pages.")
        
        metadata = {
            "filename": os.path.basename(s3_key),
            "page_count": len(reader.pages),
            "domain": "general",
            "estimated_chunk_count": len(chunks),
        }
        
        # 3. Knowledge Analysis
        logger.info("Running KnowledgeAgent...")
        ka = KnowledgeAgent()
        knowledge = ka.run(chunks=chunks, doc_metadata=metadata)
        
        # 4. Agent Discovery
        logger.info("Running AgentDiscoveryAgent...")
        ada = AgentDiscoveryAgent()
        agents = ada.run(knowledge=knowledge)
        
        # 5. Virtual Mind
        logger.info("Building VirtualMind...")
        vmb = VirtualMindBuilder()
        virtual_mind = vmb.build(knowledge=knowledge, agents=agents)
        
        # Deploy Agents Dynamically
        logger.info("Spawning agent endpoints...")
        script_path = os.path.join(os.path.dirname(__file__), "run_agent.py")
        
        for agent in virtual_mind.get("agents", []):
            name = agent.get("name", "Generated Agent")
            agent_id = re.sub(r'[^a-zA-Z0-9]+', '-', name).strip('-').lower()
            
            # Generate package
            package = generate_agent_package(agent, knowledge)
            
            # Save package to temp file
            package_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8')
            json.dump(package, package_tmp)
            package_tmp.close()
            
            port = get_free_port()
            
            # Spawn daemon process
            subprocess.Popen([
                sys.executable, script_path, str(port), agent_id, package_tmp.name
            ])
            
            # Inject endpoint directly into the virtual mind payload
            endpoint_url = f"http://127.0.0.1:{port}/chat/{agent_id}"
            agent["endpoint"] = endpoint_url
            logger.info(f"Spawned {name} on {endpoint_url}")
        
        # 6. Automation Discovery
        logger.info("Running AutomationDiscoveryAgent...")
        auda = AutomationDiscoveryAgent()
        automation_catalog = auda.run(knowledge=knowledge, virtual_mind=virtual_mind)
        
        # 7. Upload Combined Demo Artifact to S3
        report = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "knowledge_analysis": knowledge,
            "virtual_mind": virtual_mind,
            "automation_catalog": automation_catalog,
        }
        
        report_s3_key = f"projects/{project_id}/virtual_mind_report.json"
        s3_client.put_object(
            Bucket=settings.S3_BUCKET,
            Key=report_s3_key,
            Body=json.dumps(report, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Successfully uploaded virtual mind report to {report_s3_key}")
        
    except Exception as e:
        logger.error(f"Virtual Mind generation failed for {project_id}: {e}", exc_info=True)
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
