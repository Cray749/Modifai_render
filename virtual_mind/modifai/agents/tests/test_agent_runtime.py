import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from modifai.runtime.agent_runtime import app, register_agent, registered_agents
from modifai.agents.schemas import AgentPackage

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_agents():
    registered_agents.clear()
    yield

def test_runtime_agent_not_found():
    response = client.post("/agents/unknown-agent", json={"message": "hello"})
    assert response.status_code == 404

@patch("modifai.runtime.agent_runtime.get_llm_provider")
def test_runtime_chat_message_handling(mock_get_provider):
    # Setup mock provider
    mock_provider = MagicMock()
    mock_provider.generate.return_value = {"answer": "I am an HR agent."}
    mock_get_provider.return_value = mock_provider
    
    # Register agent
    package: AgentPackage = {
        "name": "HR Agent",
        "system_prompt": "You are HR Agent.",
        "description": "HR tasks",
        "specialization": "Human Resources",
        "capabilities": [],
        "knowledge_domains": [],
        "instructions": ["Be polite."],
        "version": "1.0"
    }
    register_agent("hr-agent", package)
    
    # Send message
    response = client.post("/agents/hr-agent", json={"message": "Who are you?"})
    
    assert response.status_code == 200
    assert response.json() == {"answer": "I am an HR agent."}
    
    # Verify provider called with correct injected prompt
    mock_provider.generate.assert_called_once()
    call_args = mock_provider.generate.call_args[1]
    assert "Who are you?" in call_args["user_prompt"]
    assert "You are HR Agent." in call_args["system_prompt"]
    assert "Be polite." in call_args["system_prompt"]
    assert "response_schema" in call_args

def test_chat_ui_endpoint():
    # Register agent
    package: AgentPackage = {
        "name": "HR Agent UI Test",
        "system_prompt": "You are HR Agent.",
        "description": "HR tasks",
        "specialization": "Human Resources",
        "capabilities": [],
        "knowledge_domains": [],
        "instructions": [],
        "version": "1.0"
    }
    register_agent("hr-agent-ui", package)
    
    response = client.get("/chat/hr-agent-ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Chat with HR Agent UI Test" in response.text
