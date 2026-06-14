import json
import logging
import webbrowser
import sys
from n8n_generator import generate_n8n_workflow
from n8n_deployer import deploy_workflow_to_n8n

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

def main():
    report_path = "virtual_mind_report.json"
    logger.info("Loading %s...", report_path)
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        logger.error("Report file %s not found. Please run the virtual mind pipeline first.", report_path)
        sys.exit(1)
        
    catalog = report.get("automation_catalog", {})
    opportunities = catalog.get("automation_opportunities", [])
    
    if not opportunities:
        logger.error("No automation opportunities found in the report.")
        sys.exit(1)
        
    # Select the first high-impact opportunity
    target_opp = opportunities[0]
    name = target_opp.get("name", "Generated Workflow")
    blueprint = target_opp.get("automation_blueprint", {})
    
    if not blueprint:
        logger.error("The selected opportunity does not contain an 'automation_blueprint'.")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Selected Automation Opportunity: %s", name)
    logger.info("Trigger: %s", blueprint.get("trigger", "Manual"))
    logger.info("Actions: %d", len(blueprint.get("actions", [])))
    logger.info("=" * 60)
    
    # 1. Generate n8n workflow
    logger.info("\n[1/3] Converting blueprint to n8n workflow JSON...")
    workflow_name = f"Modifai: {name}"
    workflow_json = generate_n8n_workflow(blueprint, workflow_name=workflow_name)
    
    # 2. Deploy to n8n
    logger.info("\n[2/3] Deploying workflow to n8n API...")
    try:
        workflow_url = deploy_workflow_to_n8n(workflow_json)
        logger.info("      Successfully created workflow in n8n!")
        logger.info("      Workflow URL: %s", workflow_url)
        
        # 3. Open in browser
        logger.info("\n[3/3] Opening workflow in browser automatically...")
        webbrowser.open(workflow_url)
        
        logger.info("\nDone! MVP deployment successful.")
    except Exception as e:
        logger.error("\nDeployment failed.")
        logger.error("Is n8n running locally on http://localhost:5678?")
        logger.error("Did you set N8N_API_KEY if your instance requires authentication?")
        logger.error("Details: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
