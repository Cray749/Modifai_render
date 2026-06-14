import pytest
from unittest.mock import patch

from modifai.services.agent_deployer import deploy_agent
from modifai.agents.schemas import AgentPackage
from modifai.runtime.agent_runtime import registered_agents

@pytest.fixture(autouse=True)
def clear_agents():
    registered_agents.clear()
    yield

def test_deploy_agent_registers_and_returns_url():
    package: AgentPackage = {
        "name": "Compliance Agent",
        "system_prompt": "You are Compliance Agent.",
        "description": "Compliance tasks",
        "specialization": "Compliance and Ethics",
        "capabilities": [],
        "knowledge_domains": [],
        "instructions": [],
        "version": "1.0"
    }
    
    url = deploy_agent(package, base_url="http://localhost:8000")
    
    assert url == "http://localhost:8000/agents/compliance-agent"
    assert "compliance-agent" in registered_agents
    assert registered_agents["compliance-agent"]["name"] == "Compliance Agent"
