import json

def generate_n8n_workflow(blueprint, workflow_name):
    """
    Converts automation_blueprint into valid n8n workflow JSON.
    Uses deterministic action mappings (no LLM generation initially).
    """
    trigger_text = blueprint.get("trigger", "Manual Trigger")
    actions = blueprint.get("actions", [])
    
    nodes = []
    connections = {}
    
    # Add trigger node
    # Clean the name to prevent invalid characters in n8n node names if any
    trigger_node_name = f"Trigger: {trigger_text}"[:50].strip()
    nodes.append({
        "parameters": {},
        "id": "trigger_node_1",
        "name": trigger_node_name,
        "type": "n8n-nodes-base.manualTrigger",
        "typeVersion": 1,
        "position": [250, 300]
    })
    
    prev_node_name = trigger_node_name
    x_pos = 250
    
    for i, action in enumerate(actions):
        x_pos += 200
        node_id = f"action_node_{i+2}"
        # Node names must be unique and valid
        node_name = f"Action {i+1}: {action}"[:50].strip()
        
        nodes.append({
            "parameters": {},
            "id": node_id,
            "name": node_name,
            "type": "n8n-nodes-base.noOp",
            "typeVersion": 1,
            "position": [x_pos, 300]
        })
        
        # Connect previous node to current node
        connections[prev_node_name] = {
            "main": [
                [
                    {
                        "node": node_name,
                        "type": "main",
                        "index": 0
                    }
                ]
            ]
        }
        prev_node_name = node_name

    workflow = {
        "name": workflow_name,
        "nodes": nodes,
        "connections": connections,
        "settings": {}
    }
    
    return workflow
