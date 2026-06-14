import logging
import re
from modifai.agents.schemas import AgentPackage
from modifai.runtime.agent_runtime import register_agent

logger = logging.getLogger(__name__)

def deploy_agent(package: AgentPackage, base_url: str = "http://localhost:8000") -> str:
    """
    Deploys an AgentPackage into the runtime.
    Creates an endpoint dynamically and returns the URL.
    """
    name = package.get("name", "generic-agent")
    
    # Convert "HR Agent" -> "hr-agent"
    agent_id = re.sub(r'[^a-zA-Z0-9]+', '-', name).strip('-').lower()
    
    # Register in runtime
    register_agent(agent_id, package)
    
    # Return URL
    agent_url = f"{base_url}/agents/{agent_id}"
    logger.info("Deployed agent '%s' to %s", name, agent_url)
    
    return agent_url
