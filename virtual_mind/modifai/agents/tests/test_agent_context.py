import pytest
from modifai.generators.agent_context_builder import build_agent_context

def test_build_agent_context_includes_summary_and_evidence():
    agent = {
        "name": "Test Agent",
        "specialization": "Testing",
        "description": "Tests things",
        "reasoning": "Because we need tests",
        "source_domains": ["QA"],
        "capabilities": [{"name": "Run tests", "description": "run unit tests"}]
    }
    
    knowledge = {
        "knowledge_summary": "We do QA all day.",
        "domains": [
            {
                "name": "QA",
                "description": "Quality Assurance",
                "evidence": ["Test policy v1", "Automation framework"]
            }
        ],
        "workflows": [
            {
                "name": "Testing workflow",
                "steps": ["Write test", "Test policy v1"]
            }
        ]
    }
    
    prompt = build_agent_context(agent, knowledge)
    
    assert "You are the Test Agent" in prompt
    assert "Because we need tests" in prompt
    assert "We do QA all day." in prompt
    assert "Quality Assurance" in prompt
    assert "Test policy v1" in prompt
    assert "Testing workflow" in prompt
    assert "Write test" in prompt
    assert "run unit tests" in prompt
