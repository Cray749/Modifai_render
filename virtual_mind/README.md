# Modifai Virtual Mind Platform

**Far-Away Hackathon Submission**

Modifai is an enterprise-grade AI platform that transforms raw organizational documents into a **live, deployable Virtual Mind**—a structured team of specialized AI agents grounded entirely in the company's own knowledge.

## 🚀 The Vision

Organizations hold enormous institutional knowledge locked inside PDFs, manuals, SOPs, and policy documents. Modifai solves this by making documents deployable.

Upload a document (e.g., an HR manual), and Modifai will automatically:
1. Extract knowledge, workflows, and expertise domains.
2. Discover which AI agents *should* exist based on that knowledge.
3. Generate and deploy those agents as live HTTP endpoints with chat UIs.
4. Discover and deploy automation workflows directly into n8n.

No RAG pipelines to configure. No LangChain orchestration. No manual prompting. Just autonomous, deterministic deployment.

## ⚙️ Architecture & Features

- **Knowledge-First Discovery**: Agents are discovered from the actual content of your documents, not manually configured.
- **Evidence-Backed Grounding**: Every deployed agent uses strict context injection. They know *why* they exist and cite exact document workflows.
- **Provider Agnostic**: Seamlessly swap between AWS Bedrock, Google Gemini, and OpenRouter for LLM inference.
- **One-Click Deployment**: Run a single script to generate a FastAPI server hosting all discovered agents with interactive UI endpoints.
- **Automation Discovery**: Automatically scans organizational processes and scores them for ROI, exporting direct-to-n8n workflows.

For a deep dive into the engineering, pipeline design, and sprint-by-sprint implementation, please see the [Modifai PRD](docs/Modifai_PRD.md).

## 🛠️ Installation

```bash
# Clone the repository (agents branch)
git clone -b agents https://github.com/Cray749/Modifai_Far-Away-Hackathon.git
cd Modifai_Far-Away-Hackathon

# Install required dependencies
pip install fastapi uvicorn google-generativeai requests pypdf2 pytest
```

## 🔑 Configuration

Modifai is completely provider-agnostic. Choose your preferred LLM backend by setting the appropriate environment variables.

**Option A: OpenRouter (Recommended)**
```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_MODEL=deepseek/deepseek-chat-v3
```

**Option B: Google Gemini**
```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=AIzaSy...
```

**Option C: AWS Bedrock**
```bash
export LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
# Assumes standard AWS credentials in ~/.aws/credentials
```

## 🎮 Quick Start

### 1. Build the Virtual Mind
Process a document to extract knowledge, discover agents, and find automation opportunities:
```bash
python test_virtual_mind.py
```
*(This uses `examples/sample_hr_handbook.pdf` by default and generates the entire Virtual Mind architecture).*

### 2. Deploy All Discovered Agents
Launch the FastAPI runtime and deploy every discovered organizational expert:
```bash
python deploy_all_agents.py
```

You'll see output like this:
```
✓ HR Policy Agent      → http://127.0.0.1:8000/agents/hr-policy-agent
✓ Employee Benefits    → http://127.0.0.1:8000/agents/employee-benefits-agent
...
```
You can chat with them directly in your browser via the provided `/chat/{agent_id}` UI endpoints!

### 3. Deploy Automation Workflows
Push discovered automation opportunities directly to n8n:
```bash
python deploy_automation.py
```

## 🧪 Testing

The platform includes a comprehensive test suite covering the entire multi-agent loop, deployment logic, and context generation.

```bash
pytest
```

## 📁 Repository Structure

- `modifai/agents/`: Core discovery and knowledge extraction loop.
- `modifai/core/`: LLM provider abstraction, document chunking, and pipeline utilities.
- `modifai/generators/`: Dynamic system prompt builders and n8n workflow generators.
- `modifai/runtime/`: The live FastAPI agent server.
- `modifai/services/`: Agent registration and deployment handlers.
- `examples/`: Sample documents used for pipeline testing.
- `docs/`: Extensive project documentation and planning notes.

---
*Built for the Far-Away Hackathon.*
