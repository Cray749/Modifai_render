"""
test_virtual_mind.py — End-to-end local validation of the Virtual Mind pipeline.

Usage:
    set LLM_PROVIDER=gemini
    set GEMINI_API_KEY=YOUR_KEY
    python test_virtual_mind.py

Outputs:
    virtual_mind_report.json
    events.jsonl
"""
import json
import logging
import os
from pypdf import PdfReader

from modifai.agents.pipeline_loop import run_agentic_loop

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

PDF_PATH = "examples/sample_hr_handbook.pdf"
EVENT_LOG_PATH = "test_events.jsonl"
REPORT_PATH = "virtual_mind_report.json"

def main():
    logger.info("--- Testing Virtual Mind Pipeline ---")
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    logger.info("Provider: %s", provider)

    if not os.path.exists(PDF_PATH):
        logger.error("PDF not found: %s", PDF_PATH)
        return

    # 1. Read PDF
    logger.info("Reading PDF: %s", PDF_PATH)
    reader = PdfReader(PDF_PATH)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    
    # Simple chunking for test
    chunks = [text[i:i + 12000] for i in range(0, len(text), 12000)]
    logger.info("Extracted %d chunks from %d pages.", len(chunks), len(reader.pages))

    metadata = {
        "filename": PDF_PATH,
        "page_count": len(reader.pages),
        "domain": "project management / team tasks",
        "estimated_chunk_count": len(chunks),
    }

    # 2. Run Agentic Loop (Sprint 1, 2, 3)
    logger.info("Running Agentic Loop...")
    
    state = run_agentic_loop(
        goal="Automate our team task management and workflows",
        doc_metadata=metadata,
        chunks=chunks,
        max_iterations=1,
        event_log_path=EVENT_LOG_PATH,
    )

    logger.info("Agentic Loop complete. Exit reason: %s", state["exit_reason"])

    # 3. Export Demo Artifact
    report = {
        "knowledge_analysis": state.get("knowledge_analysis"),
        "virtual_mind": state.get("virtual_mind"),
        "automation_catalog": state.get("automation_discovery_output"),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("virtual_mind_report.json generated successfully.")
    
    vm = report.get("virtual_mind") or {}
    catalog = report.get("automation_catalog") or {}
    ka = report.get("knowledge_analysis") or {}
    
    logger.info("  Knowledge domains  : %d", len(ka.get("domains", [])))
    logger.info("  Discovered agents  : %d", len(vm.get("agents", [])))
    logger.info("  Automations found  : %d (high-impact: %d)",
                catalog.get("total_opportunities", 0),
                catalog.get("high_impact_count", 0))
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
