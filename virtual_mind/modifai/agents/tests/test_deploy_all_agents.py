import pytest
import json
import os
from unittest.mock import patch, MagicMock
from modifai.services.agent_deployer import deploy_agent

@patch("deploy_all_agents.deploy_agent")
@patch("deploy_all_agents.start_server")
def test_deploy_all_agents_flow(mock_start_server, mock_deploy_agent, tmp_path):
    import deploy_all_agents
    
    # Mock report
    report = {
        "knowledge_analysis": {
            "knowledge_summary": "Summary"
        },
        "virtual_mind": {
            "agents": [
                {"name": "Agent 1", "specialization": "Spec 1"},
                {"name": "Agent 2", "specialization": "Spec 2"}
            ]
        }
    }
    
    report_file = tmp_path / "virtual_mind_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f)
        
    with patch("deploy_all_agents.logger"):
        with patch("builtins.open", return_value=open(report_file, "r")):
            # Prevent infinite loop by throwing KeyboardInterrupt on the first sleep
            with patch("time.sleep", side_effect=KeyboardInterrupt):
                deploy_all_agents.main()
                
    assert mock_deploy_agent.call_count == 2
    mock_start_server.assert_called_once()
