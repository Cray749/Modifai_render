# Modifai — Product Requirements Document (PRD)
**Far-Away Hackathon | Virtual Mind Platform**
**Version:** 2.0 (Post Sprint 6.5)
**Last Updated:** June 2026
**Status:** Implementation Complete — Ready for Demo

---

## Executive Summary

**Modifai** is an enterprise-grade AI platform that transforms raw organizational documents into a **live, deployable Virtual Mind** — a structured team of specialized AI agents grounded entirely in the company's own knowledge.

The system takes a document (e.g., an HR manual) and **automatically**:
1. Extracts knowledge, workflows, and expertise domains
2. Discovers which AI agents *should* exist based on that knowledge
3. Generates and deploys those agents as live HTTP endpoints with chat UIs
4. Discovers and deploys automation workflows directly into n8n

> **One-sentence pitch:** Upload a PDF. Get a team of grounded organizational AI experts deployed in minutes — without writing a single prompt by hand.

---

## 1. Problem Statement

Organizations hold enormous institutional knowledge locked inside PDFs, manuals, SOPs, and policy documents. Today this knowledge is:

- **Inaccessible** — employees must search manually
- **Inconsistent** — answers differ by person
- **Not actionable** — no automation emerges from it
- **Not intelligent** — existing chatbots are generic, not grounded in company policy

**Modifai solves all four** by making documents deployable.

---

## 2. Vision

```
Upload Document
      ↓
Modifai reads, understands, and structures knowledge
      ↓
Virtual Mind is generated (team of specialized AI agents)
      ↓
Agents are deployed as live chat endpoints
      ↓
Automation workflows are pushed to n8n
      ↓
Organization has instant, grounded AI infrastructure
```

---

## 3. Core Design Principles

| Principle | Implementation |
|---|---|
| **No RAG** | All knowledge is injected structurally at prompt time |
| **No LangGraph/CrewAI** | Pure FastAPI runtime, zero orchestration frameworks |
| **No Hallucination** | Every agent prompt contains hard evidence chains from source docs |
| **Provider Agnostic** | Works with Bedrock, Gemini, or OpenRouter |
| **Deterministic** | Agent generation is reproducible — same document → same agents |
| **One-click deploy** | `python deploy_all_agents.py` → live endpoints |

---

## 4. Full Pipeline Architecture

### 4.1 End-to-End Flow

```
                    ┌─────────────┐
                    │  PDF / Doc  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Chunking   │  (4k–8k token windows)
                    └──────┬──────┘
                           │
                 ┌─────────▼──────────┐
                 │  OrchestratorAgent  │  (Strategy selection)
                 └─────────┬──────────┘
                           │
            ┌──────────────▼──────────────┐
            │   Dataset Generation Layer   │
            │  (instruction-tuned samples) │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │        CriticAgent          │
            │   (quality scoring loop)    │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │      CurriculumAgent        │
            │  (difficulty sequencing)    │
            └──────────────┬──────────────┘
                           │
          ┌────────────────▼────────────────┐
          │     Knowledge Analysis Layer     │  Sprint 1
          │ domains · expertise · workflows  │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │      Agent Discovery Engine     │  Sprint 2
          │   What agents should exist?     │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │         Virtual Mind            │  Sprint 2
          │  Named agents · specializations │
          └────┬──────────────────────┬─────┘
               │                      │
  ┌────────────▼──────┐    ┌──────────▼──────────┐
  │  Automation        │    │   Agent Deployment  │  Sprint 5 & 6
  │  Discovery Engine  │    │   Engine            │
  └────────────┬──────┘    └──────────┬──────────┘
               │                      │
  ┌────────────▼──────┐    ┌──────────▼──────────┐
  │   n8n Workflow     │    │  FastAPI Runtime     │
  │   Generator &      │    │  (live chat APIs +  │
  │   Deployer         │    │   Browser UI)        │
  └───────────────────┘    └──────────────────────┘
```

### 4.2 Agentic Pipeline Loop

The core pipeline runs as a **multi-agent agentic loop**, not a simple script:

1. **OrchestratorAgent** — Analyzes document metadata and selects generation strategy (intent: instruction/conversational/reasoning, quality threshold, samples per chunk)
2. **DatasetGenerator** — Generates training Q&A pairs from each chunk using `return_raw=True` to preserve LLM's raw text for downstream parsing
3. **CriticAgent** — Scores every sample on faithfulness, coherence, and completeness. Rejects below-threshold samples.
4. **CurriculumAgent** — Sequences approved samples by complexity from foundational → advanced
5. **KnowledgeAgent** — Performs deep domain analysis to produce structured `KnowledgeAnalysisOutput`
6. **AgentDiscoveryAgent** — Reasons over knowledge to determine which specialized agents should exist and why
7. **VirtualMindBuilder** — Assembles final report with explainability, confidence scoring, and evidence chains

---

## 5. Sprint-by-Sprint Breakdown

### Sprint 1 — Knowledge Analysis Layer

**Goal:** Extract structured organizational intelligence from raw document chunks.

**Output Schema — `KnowledgeAnalysisOutput`:**
```json
{
  "knowledge_summary": "...",
  "domains": [
    {
      "name": "Human Resources",
      "description": "...",
      "evidence": ["Recruitment Policy", "Selection Procedure", "..."],
      "confidence": 0.98
    }
  ],
  "expertise": [
    {"name": "Employee Onboarding", "confidence": 0.91}
  ],
  "key_concepts": ["Recruitment Policy", "Code of Conduct", "..."],
  "workflows": [
    {
      "name": "Recruitment Process",
      "steps": ["Recruitment Policy", "Selection Procedure", "Induction", "Probation Period Reviews"],
      "confidence": 0.90
    }
  ]
}
```

**Key Design Decision:** Knowledge analysis is the single source of truth — no component after Sprint 1 may re-analyze raw documents.

---

### Sprint 2 — Agent Discovery + Virtual Mind Generation

**Goal:** Automatically determine what agents should exist, why they exist, and what they should know.

**AgentDiscovery takes:** `KnowledgeAnalysisOutput`
**AgentDiscovery produces:** `VirtualMind` (named, specialized agent roster with reasoning)

**Real output from HR Manual:**
| Agent | Specialization | Confidence |
|---|---|---|
| HR Policy Agent | Human Resources | 0.97 |
| Employee Benefits Agent | Employee Benefits | 0.93 |
| Workplace Conduct Agent | Workplace Conduct | 0.92 |
| Health and Safety Agent | Health and Safety | 0.94 |
| Legal Compliance Agent | Legal Compliance | 0.91 |

**Critical Rule:** Agents are discovered from knowledge — not from user configuration.

---

### Sprint 2.5 — Architecture Hardening

**Goal:** Add explainability and evidence-based validation before production.

**Added:**
- `reasoning` field on every `DiscoveredAgent` — the agent knows *why* it was created
- `confidence` scoring — every agent and domain has a float 0–1 confidence score
- `evidence` chains — all domain claims are backed by document citations
- `knowledge_summary` — organization-level synthesis

**Architectural impact:** Made the Virtual Mind auditable and self-explanatory to non-technical stakeholders.

---

### Sprint 3 — Automation Discovery Engine

**Goal:** Identify business processes that can be automated, scored by ROI potential.

**Input:** `KnowledgeAnalysisOutput` + `VirtualMind`
**Output:** `AutomationCatalog`

**Discovered from HR Manual:**
| Automation | Score | Business Impact | Key Benefit |
|---|---|---|---|
| Recruitment Process Automation | 85 | 🔴 High | 40% faster hiring, 10 hrs/cycle saved |
| Performance Appraisal Automation | 80 | 🔴 High | 3 hrs/appraisal, 50% fewer errors |
| Employee Onboarding Automation | 75 | 🔴 High | 25% fewer onboarding errors, 5 hrs/hire saved |

Each automation includes a **blueprint** with:
- `trigger` — what event starts the workflow
- `actions` — discrete step-by-step operations

---

### Sprint 5 — n8n Deployment Engine

**Goal:** Automatically deploy discovered automation workflows into a live n8n instance.

**Flow:**
```
AutomationOpportunity
       ↓
n8n_generator.py   →  Valid n8n workflow JSON (deterministic mapping)
       ↓
n8n_deployer.py    →  POST to n8n API → Workflow URL returned
       ↓
deploy_automation.py →  Opens workflow in browser automatically
```

**Key file: `n8n_generator.py`**
Uses deterministic action-to-node mappings. No LLM involved in workflow generation.

**Key file: `n8n_deployer.py`**
Authenticates via n8n API key. Creates workflow. Returns URL.

**CLI usage:**
```bash
python deploy_automation.py
```

**Result:** n8n workflow appears in editor automatically.

---

### Sprint 6 — Agent Deployment Engine

**Goal:** Deploy discovered agents as live, callable HTTP endpoints with browser-accessible chat UIs.

**New modules created:**
| File | Responsibility |
|---|---|
| `modifai/generators/agent_generator.py` | `DiscoveredAgent` → `AgentPackage` |
| `modifai/runtime/agent_runtime.py` | FastAPI app with dynamic routing |
| `modifai/services/agent_deployer.py` | Registration + URL generation |
| `deploy_agent.py` | Single-agent CLI deployment script |
| `deploy_all_agents.py` | Batch deployment of all 5 agents |

**AgentPackage schema:**
```python
class AgentPackage(TypedDict):
    name: str               # e.g. "HR Policy Agent"
    system_prompt: str      # Full context-injected prompt
    description: str
    specialization: str
    capabilities: List[AgentCapability]
    knowledge_domains: List[str]
    instructions: List[str] # Operational behavioral rules
    version: str
```

**Runtime Architecture:**
- FastAPI application with dynamic route registration
- Routes: `POST /agents/{agent_id}` — inference endpoint
- Routes: `GET /chat/{agent_id}` — browser chat UI (zero frontend build needed)
- All requests use the `BaseLLMProvider` abstraction (Gemini / OpenRouter / Bedrock)
- Responses forced to `{"answer": "..."}` via `response_schema`

**Live output from `deploy_all_agents.py`:**
```
✓ HR Policy Agent      → http://127.0.0.1:8000/agents/hr-policy-agent
✓ Employee Benefits    → http://127.0.0.1:8000/agents/employee-benefits-agent
✓ Workplace Conduct    → http://127.0.0.1:8000/agents/workplace-conduct-agent
✓ Health and Safety    → http://127.0.0.1:8000/agents/health-and-safety-agent
✓ Legal Compliance     → http://127.0.0.1:8000/agents/legal-compliance-agent
```

---

### Sprint 6.5 — Agent Intelligence Hardening

**Goal:** Transform deployed agents from generic chatbots with custom names into genuine organizational experts grounded in company knowledge.

**The Problem:** Previous agent prompts contained only name + capabilities. No organizational context.

**The Fix — `agent_context_builder.py`:**

`build_agent_context(agent, knowledge)` assembles a structured system prompt with 6 sections:

1. **Identity & Reasoning** — Who the agent is and *why it was created*
2. **Organization Knowledge Summary** — The global `knowledge_summary` from Sprint 1
3. **Key Organizational Concepts** — Core terminology from the document
4. **Source Domains & Evidence** — Specific domain citations from the document
5. **Relevant Workflows** — Step-by-step procedures from the agent's domain automatically injected
6. **Operational Instructions** — Capabilities transformed into behavioral directives

**Before Sprint 6.5 (generic prompt):**
```
You are HR Agent, an AI assistant specialized in Human Resources.
Answer questions accurately within the domain of Human Resources.
```

**After Sprint 6.5 (organizational expert prompt):**
```
You are the HR Policy Agent, an organizational expert specialized in Human Resources.
Reasoning for your existence: The document contains comprehensive HR policies and 
structured workflows for recruitment, onboarding, and employment status management.

--- ORGANIZATION KNOWLEDGE SUMMARY ---
The document is an HR manual for 'Company Name' (likely Marque Impex) that outlines 
comprehensive HR policies, procedures, and employee benefits...

--- KEY ORGANIZATIONAL CONCEPTS ---
- Recruitment Policy
- Employee Benefits
- Code of Conduct
- Performance Management
- Workplace Ethics

--- YOUR SOURCE DOMAINS & EVIDENCE ---
Domain: Human Resources
Supporting Evidence:
  * Recruitment Policy
  * Selection Procedure
  * Employee Benefits
  * Performance Management System
  * Grievance Redressal Policy

--- RELEVANT WORKFLOWS ---
Workflow: Recruitment Process
  - Recruitment Policy
  - Selection Procedure
  - Induction
  - Probation Period Reviews

--- YOUR OPERATIONAL INSTRUCTIONS & CAPABILITIES ---
Capability: Recruitment Process Management
Operational Instruction: Answer questions regarding and execute tasks related to 
automating and managing the end-to-end recruitment process from advertisement to induction.

--- FINAL DIRECTIVES ---
You must answer purely based on the organizational knowledge and workflows provided above.
Do not hallucinate policies. If asked something outside your domain, politely decline.
```

---

## 6. Provider Abstraction Layer

**Critical engineering contribution:** All LLM calls are routed through a unified `BaseLLMProvider` interface, making the entire system provider-agnostic.

```python
class BaseLLMProvider(ABC):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict] = None,
        tool_name: Optional[str] = None,
        return_raw: bool = False,  # Sprint 6.5 addition
        **kwargs                   # temperature, max_tokens, top_p
    ) -> Any: ...
```

### Supported Providers

| Provider | Class | Config |
|---|---|---|
| AWS Bedrock | `BedrockProvider` | `LLM_PROVIDER=bedrock` |
| Google Gemini | `GeminiProvider` | `LLM_PROVIDER=gemini` |
| OpenRouter | `OpenRouterProvider` | `LLM_PROVIDER=openrouter` |

### `return_raw` Flag (Sprint 6.5 Bug Fix)
**Root cause:** Dataset generation's `_parse_generation_response()` expected a raw string, but all providers were returning parsed Python objects via `safe_json_generation()`.

**Fix:** Added `return_raw=True` parameter. When set, `safe_json_generation()` returns the raw LLM text without JSON parsing — allowing dataset generation to perform its own parsing while all other pipeline components continue using structured outputs.

### OpenRouter Fallback Chain
```python
self.fallback_models = [
    "deepseek/deepseek-chat-v3",
    "qwen/qwen3-235b-a22b",
    "google/gemini-2.5-flash-lite"
]
```
On any error (timeout, rate limit, malformed JSON), automatically retries with the next model.

---

## 7. Schema Architecture

The system's typed contract system (`modifai/agents/schemas.py`) ensures pipeline consistency:

```
KnowledgeAnalysisOutput
    ↓ consumed by
VirtualMind (contains list of DiscoveredAgent)
    ↓ consumed by
AgentPackage (deployable unit)
    ↓ registered in
AgentRuntime (FastAPI endpoints)

AutomationCatalog (contains list of AutomationOpportunity)
    ↓ converted by
n8n_generator.py
    ↓ deployed by
n8n_deployer.py
```

---

## 8. Test Coverage

All components have corresponding test files:

| Test File | Coverage |
|---|---|
| `test_agent_generator.py` | AgentPackage generation from DiscoveredAgent + Knowledge |
| `test_agent_runtime.py` | FastAPI endpoints, 404 on unknown agents, response format |
| `test_agent_deployment.py` | Agent registration, URL construction |
| `test_agent_context.py` | Context builder injects summary, evidence, workflows |
| `test_agent_package_quality.py` | No placeholders ("TODO", "helpful assistant", etc.) |
| `test_deploy_all_agents.py` | Multi-agent batch deployment flow |
| `test_agent_discovery.py` | Discovery from knowledge analysis |
| `test_automation_discovery.py` | Automation opportunity scoring |
| `test_knowledge_agent.py` | Knowledge extraction pipeline |
| `test_pipeline_e2e.py` | End-to-end agentic loop |
| `test_orchestrator.py` | Strategy selection logic |
| `test_curriculum.py` | Curriculum sequencing |

**Test run result (Sprint 6.5):** `8 passed in 1.39s ✓`

---

## 9. Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Runtime API | FastAPI + Uvicorn |
| Data Validation | Pydantic v2, TypedDict |
| LLM Providers | AWS Bedrock, Google Gemini, OpenRouter |
| Workflow Automation | n8n (self-hosted, REST API) |
| Document Parsing | PyPDF2 (PdfReader) |
| Testing | pytest |
| Repository | GitHub — `Cray749/Modifai_Far-Away-Hackathon` |
| Branch | `agents` |

---

## 10. Deployment Instructions

### Prerequisites
```bash
pip install fastapi uvicorn google-generativeai requests pypdf2 pytest
```

### Environment Variables
```bash
# Choose ONE:
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=deepseek/deepseek-chat-v3

# OR:
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...

# OR:
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
```

### Run the Full Pipeline
```bash
python test_virtual_mind.py          # Generates virtual_mind_report.json
python deploy_all_agents.py          # Deploys all 5 agents to FastAPI runtime
python deploy_automation.py          # Deploys workflows to n8n
```

### Chat with an Agent (curl)
```bash
curl -X POST "http://127.0.0.1:8000/agents/hr-policy-agent" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the steps in the recruitment process?"}'
# → {"answer": "The Recruitment Process at Marque Impex consists of..."}
```

---

## 11. Demonstrated Results (HR Manual Demo)

**Source document:** `375731965-HR-Manual-Employee-Handbook.pdf` (57 pages)
**Chunks processed:** ~13 at 8,000 tokens each
**Provider used:** OpenRouter (DeepSeek v3)

### Generated Agents
| Agent | Endpoint | Specialization |
|---|---|---|
| HR Policy Agent | `/agents/hr-policy-agent` | Human Resources |
| Employee Benefits Agent | `/agents/employee-benefits-agent` | Employee Benefits |
| Workplace Conduct Agent | `/agents/workplace-conduct-agent` | Workplace Conduct |
| Health and Safety Agent | `/agents/health-and-safety-agent` | Health and Safety |
| Legal Compliance Agent | `/agents/legal-compliance-agent` | Legal Compliance |

### Generated Automations
| Automation | Score | Savings |
|---|---|---|
| Recruitment Process Automation | 85/100 | 10 hrs/cycle, 40% faster |
| Performance Appraisal Automation | 80/100 | 3 hrs/appraisal, 50% fewer errors |
| Employee Onboarding Automation | 75/100 | 5 hrs/hire, 25% fewer errors |

---

## 12. PPT Presentation Structure (Recommended)

### Slide 1 — Title
**Modifai: Your Documents, Deployed as AI**
*Far-Away Hackathon | June 2026*

### Slide 2 — The Problem (30 seconds)
- Knowledge is locked in PDFs
- Generic chatbots hallucinate
- No automation emerges from documentation
- Visual: Stack of PDFs → Question mark

### Slide 3 — The Solution (30 seconds)
- Modifai reads your documents
- Understands your organization's structure
- Deploys grounded AI agents automatically
- Visual: PDF → Virtual Mind → Live APIs

### Slide 4 — Full Pipeline (1 minute)
- Show the ASCII pipeline diagram from Section 4.1
- Emphasize: document in → agents out, no manual prompting

### Slide 5 — Knowledge Analysis (30 seconds)
- Show real output: domains, expertise, evidence, workflows
- Highlight: everything is evidence-backed, not made up

### Slide 6 — Virtual Mind (30 seconds)
- Show the 5 agents table with confidence scores
- Key message: agents are *discovered*, not configured

### Slide 7 — Automation Discovery (30 seconds)
- Show the 3 automations with scores and estimated savings
- Key message: 18+ hours saved per cycle, automatically identified

### Slide 8 — Live Demo: Agent Chat (1–2 minutes)
- Run `python deploy_all_agents.py` live or show recording
- Open `http://127.0.0.1:8000/chat/hr-policy-agent`
- Ask: "What is the recruitment process?"
- Show the grounded, citation-aware response

### Slide 9 — Live Demo: n8n Workflow (30 seconds)
- Show n8n editor with the generated Recruitment Automation workflow
- Key message: from document to deployed workflow, fully automatic

### Slide 10 — Provider Agnostic Architecture (30 seconds)
- Show the 3 providers: Bedrock, Gemini, OpenRouter
- `return_raw` flag diagram for dataset generation contract
- Key message: works anywhere, no vendor lock-in

### Slide 11 — Technical Stack + Tests
- Technology table
- `8 passed in 1.39s ✓`

### Slide 12 — What Makes Modifai Different
| Feature | Modifai | Generic Chatbots |
|---|---|---|
| Grounding | Evidence-based from your docs | Hallucination-prone |
| Agent discovery | Automatic from knowledge | Manual configuration |
| Automation discovery | Automatic + scored | Manual identification |
| Deployment | One CLI command | Complex infrastructure |
| Provider lock-in | None (3 providers) | Vendor-specific |

### Slide 13 — Future Roadmap
- MCP integration for tool-calling agents
- Cross-agent communication
- Frontend dashboard for Virtual Mind management
- Fine-tuning loop integration
- Multi-document organizational intelligence

### Slide 14 — Repository + Thank You
`github.com/Cray749/Modifai_Far-Away-Hackathon` (branch: `agents`)

---

## Appendix A — Key Files Reference

| File | Purpose |
|---|---|
| `modifai/agents/knowledge_agent.py` | Extracts structured knowledge from chunks |
| `modifai/agents/agent_discovery.py` | Discovers agents from knowledge analysis |
| `modifai/agents/virtual_mind_builder.py` | Assembles the complete Virtual Mind report |
| `modifai/agents/automation_discovery.py` | Discovers and scores automation opportunities |
| `modifai/generators/agent_context_builder.py` | Builds rich, grounded system prompts |
| `modifai/generators/agent_generator.py` | Converts DiscoveredAgent → AgentPackage |
| `modifai/runtime/agent_runtime.py` | FastAPI server with dynamic agent endpoints |
| `modifai/services/agent_deployer.py` | Registers agents and generates URLs |
| `modifai/core/llm_provider.py` | Provider abstraction (Bedrock/Gemini/OpenRouter) |
| `n8n_generator.py` | Converts automation blueprints → n8n JSON |
| `n8n_deployer.py` | Deploys workflows to n8n via REST API |
| `deploy_agent.py` | Single-agent deployment CLI |
| `deploy_all_agents.py` | Batch deployment of all Virtual Mind agents |
| `virtual_mind_report.json` | Complete pipeline output (knowledge + agents + automations) |

---

*Document generated from the Modifai codebase at `agents` branch.*
*All outputs shown are real — generated from a 57-page HR manual.*
