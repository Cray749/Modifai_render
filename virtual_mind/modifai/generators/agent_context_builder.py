from typing import Dict, Any

def build_agent_context(agent: Dict[str, Any], knowledge: Dict[str, Any]) -> str:
    """
    Builds the canonical system prompt foundation by injecting rich organizational 
    context into the agent's prompt.
    """
    name = agent.get("name", "Organizational Expert")
    specialization = agent.get("specialization", "General Knowledge")
    description = agent.get("description", "")
    reasoning = agent.get("reasoning", "")
    source_domains = agent.get("source_domains", [])
    source_expertise = agent.get("source_expertise", [])
    capabilities = agent.get("capabilities", [])

    k_summary = knowledge.get("knowledge_summary", "")
    k_domains = knowledge.get("domains", [])
    k_expertise = knowledge.get("expertise", [])
    k_workflows = knowledge.get("workflows", [])
    k_concepts = knowledge.get("key_concepts", [])

    prompt = []

    # 1. Identity & Reasoning
    prompt.append(f"You are the {name}, an organizational expert specialized in {specialization}.")
    prompt.append(f"Description: {description}")
    if reasoning:
        prompt.append(f"Reasoning for your existence: {reasoning}")
    prompt.append("\n--- ORGANIZATION KNOWLEDGE SUMMARY ---")
    if k_summary:
        prompt.append(k_summary)

    # 2. Key Concepts
    if k_concepts:
        prompt.append("\n--- KEY ORGANIZATIONAL CONCEPTS ---")
        for concept in k_concepts:
            prompt.append(f"- {concept}")

    # 3. Domains & Evidence
    prompt.append("\n--- YOUR SOURCE DOMAINS & EVIDENCE ---")
    for d_name in source_domains:
        # Find the domain in knowledge
        domain_obj = next((d for d in k_domains if d.get("name") == d_name), None)
        if domain_obj:
            prompt.append(f"Domain: {d_name}")
            prompt.append(f"Description: {domain_obj.get('description', '')}")
            evidence = domain_obj.get("evidence", [])
            if evidence:
                prompt.append("Supporting Evidence:")
                for ev in evidence:
                    prompt.append(f"  * {ev}")

    # 4. Workflows (Filtered by agent's domains or expertise)
    # A workflow belongs to the agent if its steps overlap with the agent's domain evidence,
    # or if we just inject workflows that are globally relevant.
    # To be precise, if any evidence in the agent's domain matches a workflow step, or if we map them.
    # We will inject workflows whose name or steps relate to the agent's specialization or source expertise.
    # For now, let's inject all workflows that mention the agent's domain/expertise or just globally.
    # Actually, the prompt says "If a workflow belongs to the agent's domain: Inject workflow details."
    relevant_workflows = []
    # Collect all evidence from the agent's domains
    agent_evidence_set = set()
    for d_name in source_domains:
        domain_obj = next((d for d in k_domains if d.get("name") == d_name), None)
        if domain_obj:
            agent_evidence_set.update(domain_obj.get("evidence", []))

    for w in k_workflows:
        # If any step is in the agent's domain evidence, it belongs to the agent
        # Or if the workflow name contains words from specialization
        w_steps = set(w.get("steps", []))
        if w_steps.intersection(agent_evidence_set) or specialization.lower() in w.get("name", "").lower():
            relevant_workflows.append(w)
        else:
            # Fallback heuristic: check if source expertise overlaps with workflow name
            for exp in source_expertise:
                if exp.lower() in w.get("name", "").lower():
                    relevant_workflows.append(w)
                    break

    # Deduplicate workflows
    unique_workflows = {w["name"]: w for w in relevant_workflows}.values()
    
    if unique_workflows:
        prompt.append("\n--- RELEVANT WORKFLOWS ---")
        for w in unique_workflows:
            prompt.append(f"Workflow: {w.get('name')}")
            for step in w.get("steps", []):
                prompt.append(f"  - {step}")

    # 5. Capability Hardening
    prompt.append("\n--- YOUR OPERATIONAL INSTRUCTIONS & CAPABILITIES ---")
    for cap in capabilities:
        cap_name = cap.get("name", "")
        cap_desc = cap.get("description", "")
        # Transform into operational instruction
        prompt.append(f"Capability: {cap_name}")
        prompt.append(f"Operational Instruction: Answer questions regarding and execute tasks related to {cap_desc.lower()}.")

    prompt.append("\n--- FINAL DIRECTIVES ---")
    prompt.append("You must answer purely based on the organizational knowledge and workflows provided above.")
    prompt.append("Do not hallucinate policies. If asked something outside your domain, politely decline.")

    return "\n".join(prompt)
