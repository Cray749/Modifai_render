import pytest
from modifai.agents.schemas import DiscoveredAgent
from modifai.generators.agent_generator import generate_agent_package

def test_generate_agent_package():
    discovered: DiscoveredAgent = {
        "name": "HR Agent",
        "description": "Handles HR tasks",
        "specialization": "Human Resources",
        "reasoning": "Reason",
        "confidence": 0.9,
        "source_domains": ["HR Policy"],
        "source_expertise": ["Onboarding"],
        "capabilities": [
            {"name": "Answer HR", "description": "Answers HR policy"}
        ]
    }
    
    knowledge = {
        "knowledge_summary": "We are a company.",
        "domains": [
            {
                "name": "HR Policy",
                "description": "HR stuff",
                "evidence": ["Some rule"]
            }
        ],
        "workflows": [
            {
                "name": "Onboarding",
                "steps": ["Do this", "Some rule"]
            }
        ]
    }
    
    package = generate_agent_package(discovered, knowledge)
    
    assert package["name"] == "HR Agent"
    assert package["specialization"] == "Human Resources"
    assert "You are the HR Agent" in package["system_prompt"]
    assert "HR Policy" in package["system_prompt"]
    assert "Capability: Answer HR" in package["system_prompt"]
    assert "Some rule" in package["system_prompt"]
    assert "We are a company." in package["system_prompt"]
    assert package["version"] == "1.0"
    assert len(package["instructions"]) > 0
    assert package["knowledge_domains"] == ["HR Policy"]
