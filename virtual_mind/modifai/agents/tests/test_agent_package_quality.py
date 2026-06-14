import pytest
from modifai.generators.agent_generator import generate_agent_package
from modifai.agents.schemas import DiscoveredAgent

def test_no_placeholders_in_package():
    # Provide an agent and knowledge
    agent: DiscoveredAgent = {
        "name": "Employee Relations Agent",
        "specialization": "Employee Relations",
        "description": "Handles workplace conduct",
        "reasoning": "We need it",
        "source_domains": ["Employee Relations"],
        "source_expertise": ["Leave Management"],
        "capabilities": [
            {"name": "Leave Management", "description": "manages leave"}
        ]
    }
    
    knowledge = {
        "knowledge_summary": "Summary of manual.",
        "domains": [
            {"name": "Employee Relations", "description": "ER desc", "evidence": ["Leave Policy"]}
        ],
        "workflows": [
            {"name": "Leave App", "steps": ["Leave Policy", "Approval"]}
        ]
    }
    
    package = generate_agent_package(agent, knowledge)
    
    prompt = package["system_prompt"]
    
    # Assert no generic stubs
    stubs = ["TODO", "Example instructions", "Placeholder", "You are a helpful assistant"]
    for stub in stubs:
        assert stub.lower() not in prompt.lower()
        
    # Assert context injection
    assert "Summary of manual." in prompt
    assert "Leave App" in prompt
    assert "Approval" in prompt
    assert "Leave Policy" in prompt
