# pyrefly: ignore [missing-import]
import pytest
import os
os.environ["LLM_PROVIDER"] = "bedrock"
from unittest.mock import MagicMock, patch
from modifai.agents.knowledge_agent import KnowledgeAgent

DOC_METADATA = {
    "filename": "support_runbook.pdf",
    "page_count": 20,
    "domain": "customer support",
    "estimated_chunk_count": 40,
}

STRATEGY = {
    "intent": "QA",
    "quality_threshold": 0.70,
    "samples_per_chunk": 4,
    "reasoning": "Support FAQ domain.",
}

def test_knowledge_agent_success():
    """Test that the agent successfully parses domains, expertise, and workflows."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "analyze_knowledge",
                                "input": {
                                    "knowledge_summary": "Summary of Support",
                                    "domains": [{"name": "Support", "description": "Support desk operations", "evidence": ["Support manual"], "confidence": 0.95}],
                                    "expertise": [{"name": "Ticketing", "confidence": 0.95}],
                                    "key_concepts": ["Tickets"],
                                    "workflows": [{"name": "Ticket Triage", "steps": ["Open", "Assign", "Close"], "confidence": 0.9}]
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        agent = KnowledgeAgent(model_id="test", region="us-east-1")
        result = agent.run(chunks=["Hello", "World"], doc_metadata=DOC_METADATA, strategy=STRATEGY)
        
        assert len(result["domains"]) == 1
        assert result["domains"][0]["name"] == "Support"
        assert len(result["expertise"]) == 1
        assert result["workflows"][0]["name"] == "Ticket Triage"
        
def test_knowledge_agent_validation_failure():
    """Test that the agent raises ValueError on missing expected keys."""
    with patch("boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "analyze_knowledge",
                                "input": {
                                    "knowledge_summary": "Summary",
                                    "domains": [],
                                    "expertise": [],
                                    "key_concepts": [],
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        agent = KnowledgeAgent(model_id="test", region="us-east-1", max_retries=0)
        with pytest.raises(ValueError):
            agent.run(chunks=["Hello"], doc_metadata=DOC_METADATA)
