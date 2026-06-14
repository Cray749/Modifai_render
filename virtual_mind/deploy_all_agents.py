import json
import logging
import sys
import threading
import time
import uvicorn
import re
import uuid
from datetime import datetime, timezone

from modifai.generators.agent_generator import generate_agent_package
from modifai.services.agent_deployer import deploy_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

def emit_event(agent: str, decision: str, data: dict):
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "iteration": 0,
        "decision": decision,
        "reason": None,
        "data": data
    }
    try:
        with open("events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass

def start_server():
    uvicorn.run("modifai.runtime.agent_runtime:app", host="127.0.0.1", port=8000, log_level="warning")

def main():
    report_path = "virtual_mind_report.json"
    logger.info("Loading %s...", report_path)
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        logger.error("Report file %s not found. Please run the virtual mind pipeline first.", report_path)
        sys.exit(1)
        
    vm = report.get("virtual_mind", {})
    agents = vm.get("agents", [])
    knowledge = report.get("knowledge_analysis", {})
    
    if not agents:
        logger.error("No agents found in the Virtual Mind report.")
        sys.exit(1)
        
    logger.info("=" * 60)
    logger.info("Discovered %d agents. Deploying...", len(agents))
    logger.info("=" * 60)
    
    deployed_urls = []
    
    for agent in agents:
        name = agent.get("name", "Generated Agent")
        logger.info("\nDeploying Agent: %s", name)
        
        # 1. Generate Agent Package
        package = generate_agent_package(agent, knowledge)
        emit_event("agent_generation", f"Generated package for {name}", package)
        
        # 2. Deploy Agent to Runtime
        agent_url = deploy_agent(package, base_url="http://127.0.0.1:8000")
        emit_event("agent_deployment", f"Deployed {name} to {agent_url}", {"url": agent_url})
        
        agent_id = re.sub(r'[^a-zA-Z0-9]+', '-', name).strip('-').lower()
        ui_url = f"http://127.0.0.1:8000/chat/{agent_id}"
        
        deployed_urls.append((name, agent_url, ui_url))
        
    logger.info("\n" + "=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info("=" * 60)
    for name, api, ui in deployed_urls:
        logger.info("Agent: %s", name)
        logger.info("  API: %s", api)
        logger.info("  UI : %s", ui)
        
    logger.info("\nStarting agent runtime server...")
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    logger.info("\nServer is active. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    main()
