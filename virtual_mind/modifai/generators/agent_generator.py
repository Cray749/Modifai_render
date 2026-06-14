from typing import Dict, Any
from modifai.agents.schemas import DiscoveredAgent, AgentPackage
from modifai.generators.agent_context_builder import build_agent_context

def generate_agent_package(agent: DiscoveredAgent, knowledge: Dict[str, Any]) -> AgentPackage:
    """
    Deterministically convert a DiscoveredAgent from the Virtual Mind into a 
    deployable AgentPackage, completely grounded in the organizational knowledge.
    """
    name = agent.get("name", "Organizational Expert")
    description = agent.get("description", "")
    specialization = agent.get("specialization", "")
    capabilities = agent.get("capabilities", [])
    source_domains = agent.get("source_domains", [])
    starter_questions = agent.get("starter_questions", agent.get("example_questions", []))
    
    # Use context builder to create the comprehensive system prompt
    system_prompt = build_agent_context(agent, knowledge)
    
    # Generate strict operational instructions based on capabilities
    instructions = [
        f"You are the {name}, focusing purely on {specialization}.",
        "Answer questions accurately within your specific organizational domains.",
        "Refuse to answer generic questions or questions outside your explicit domains.",
        "Use the evidence and workflows provided in your system prompt to ground every answer."
    ]
    
    package: AgentPackage = {
        "name": name,
        "system_prompt": system_prompt,
        "description": description,
        "specialization": specialization,
        "capabilities": capabilities,
        "knowledge_domains": source_domains,
        "instructions": instructions,
        "starter_questions": starter_questions,
        "version": "1.0"
    }
    
    return package
