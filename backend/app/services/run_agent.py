import sys
import json
import uvicorn
import logging

# Ensure virtual_mind is in path
import os
VIRTUAL_MIND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../virtual_mind"))
if VIRTUAL_MIND_PATH not in sys.path:
    sys.path.append(VIRTUAL_MIND_PATH)

from modifai.runtime.agent_runtime import register_agent, app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_agent.py <port> <agent_id> <package_json_path>")
        sys.exit(1)
        
    port = int(sys.argv[1])
    agent_id = sys.argv[2]
    package_path = sys.argv[3]
    
    logger.info(f"Loading agent package from {package_path}")
    with open(package_path, "r", encoding="utf-8") as f:
        package = json.load(f)
        
    register_agent(agent_id, package)
    
    logger.info(f"Starting dedicated Uvicorn server for {agent_id} on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
