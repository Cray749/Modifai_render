"""
run_real.py — real document validation for Sprint 2.5 + Sprint 3.

Runs the full knowledge layer against an actual PDF without mocks:
  1. KnowledgeAgent      → raw_knowledge_analysis.json
  2. AgentDiscoveryAgent → raw_discovered_agents.json
  3. VirtualMindBuilder  → raw_virtual_mind.json
  4. AutomationDiscovery → raw_automation_catalog.json
  5. Combined            → virtual_mind_report.json  (primary demo artifact)

Requires AWS credentials in environment or ~/.aws/credentials.
"""
import json
import logging
from datetime import datetime, timezone

from modifai.agents.knowledge_agent import KnowledgeAgent
from modifai.agents.agent_discovery import AgentDiscoveryAgent
from modifai.agents.virtual_mind_builder import VirtualMindBuilder
from modifai.agents.automation_discovery import AutomationDiscoveryAgent
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


PDF_PATH = "modifai_4person_todo.pdf"
MODEL_ID = "amazon.nova-micro-v1:0"
REGION = "us-east-1"


def main():
    # ── Step 0: Extract text ──────────────────────────────────────────────────
    logger.info("Reading PDF: %s", PDF_PATH)
    reader = PdfReader(PDF_PATH)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
    logger.info("Extracted %d chunks from %d pages.", len(chunks), len(reader.pages))

    metadata = {
        "filename": PDF_PATH,
        "page_count": len(reader.pages),
        "domain": "project management / team tasks",
        "estimated_chunk_count": len(chunks),
    }

    # ── Step 1: Knowledge Analysis ────────────────────────────────────────────
    logger.info("Running KnowledgeAgent...")
    ka = KnowledgeAgent(model_id=MODEL_ID, region=REGION)
    knowledge = ka.run(chunks=chunks, doc_metadata=metadata)

    with open("raw_knowledge_analysis.json", "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2)
    logger.info("Saved raw_knowledge_analysis.json")

    # ── Step 2: Agent Discovery ───────────────────────────────────────────────
    logger.info("Running AgentDiscoveryAgent...")
    ada = AgentDiscoveryAgent(model_id=MODEL_ID, region=REGION)
    agents = ada.run(knowledge=knowledge)

    with open("raw_discovered_agents.json", "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2)
    logger.info("Saved raw_discovered_agents.json")

    # ── Step 3: Virtual Mind ──────────────────────────────────────────────────
    logger.info("Building VirtualMind...")
    vmb = VirtualMindBuilder()
    virtual_mind = vmb.build(knowledge=knowledge, agents=agents)

    with open("raw_virtual_mind.json", "w", encoding="utf-8") as f:
        json.dump(virtual_mind, f, indent=2)
    logger.info("Saved raw_virtual_mind.json")

    # ── Step 4: Automation Discovery ─────────────────────────────────────────
    logger.info("Running AutomationDiscoveryAgent...")
    auda = AutomationDiscoveryAgent(model_id=MODEL_ID, region=REGION)
    automation_catalog = auda.run(knowledge=knowledge, virtual_mind=virtual_mind)

    with open("raw_automation_catalog.json", "w", encoding="utf-8") as f:
        json.dump(automation_catalog, f, indent=2)
    logger.info("Saved raw_automation_catalog.json")

    # ── Step 5: Combined demo artifact ───────────────────────────────────────
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "knowledge_analysis": knowledge,
        "virtual_mind": virtual_mind,
        "automation_catalog": automation_catalog,
    }

    with open("virtual_mind_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("virtual_mind_report.json generated successfully.")
    logger.info(
        "  Knowledge domains  : %d", len(knowledge.get("domains", []))
    )
    logger.info(
        "  Discovered agents  : %d", len(agents)
    )
    logger.info(
        "  Automations found  : %d (high-impact: %d)",
        automation_catalog["total_opportunities"],
        automation_catalog["high_impact_count"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
