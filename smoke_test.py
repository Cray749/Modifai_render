"""
Smoke test for the Modifai agentic pipeline.
Run from: python smoke_test.py

This makes REAL AWS Bedrock calls -- expect 1-3 minutes to complete.
"""
import os
import logging

# Show log messages so you can see progress in real time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_MODEL_ID"] = "amazon.nova-micro-v1:0"

from modifai.agents.pipeline_loop import run_agentic_loop

print("=" * 60)
print("Modifai Pipeline Smoke Test")
print("=" * 60)
print("Making real Bedrock calls — this takes 1-3 minutes...")
print()

chunks = [
    "Step 1: Open the support portal. Step 2: Click on Tickets. Step 3: Assign priority label.",
    "Refund policy: All customers may request a refund within 30 days of purchase by contacting support.",
    "Escalation path: Tier 1 handles basic queries. Tier 2 handles technical issues. Manager handles escalations.",
]

state = run_agentic_loop(
    goal="Generate a fine-tuning dataset for customer support Q&A",
    doc_metadata={
        "filename": "demo_doc.pdf",
        "page_count": 5,
        "domain": "customer support",
        "estimated_chunk_count": 3,
    },
    chunks=chunks,
    max_iterations=2,
    event_log_path="smoke_events.jsonl",
)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Exit reason:      {state['exit_reason']}")
print(f"Iterations ran:   {state['iteration']}")
print(f"Final samples:    {len(state['final_samples'])}")
print(f"Accept pct:       {state['final_stats']['accept_pct']}%")
print(f"Events logged:    {len(state['events'])}")
print(f"Curriculum loops: {len(state['curriculum_outputs'])}")
print()
print("Events written to: smoke_events.jsonl")
print()

# Show each event for transparency
for event in state["events"]:
    print(f"  [{event['agent'].upper():12s}] iter={event['iteration']}  {event['decision']}")

print()
passed = state["exit_reason"] in ("threshold_met", "max_iterations", "all_accepted_first_pass")
print("Smoke test PASSED [OK]" if passed else "Something unexpected happened.")
