import logging
import json
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from modifai.agents.schemas import AgentPackage
from modifai.core.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

app = FastAPI(title="Modifai Agent Runtime")
# In-memory store of registered agents
registered_agents = {}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/agents/{agent_id}")
def chat_with_agent(agent_id: str, req: ChatRequest):
    if agent_id not in registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    package: AgentPackage = registered_agents[agent_id]
    
    provider = get_llm_provider()
    
    # Construct the runtime system prompt combining the package details
    system_prompt = package["system_prompt"]
    system_prompt += "\n\nAdditional Instructions:\n"
    for inst in package.get("instructions", []):
        system_prompt += f"- {inst}\n"
        
    logger.info("Agent %s received message: %s", package["name"], req.message)
    
    try:
        # We don't have a specific schema, just plain text response
        # wait, BaseLLMProvider.generate requires JSON return? 
        # By default our get_llm_provider setup uses safe_json_generation if response_schema is passed,
        # but if response_schema is None, BedrockProvider currently returns JSON if we force tool, else plain text.
        # Let's use a strict response_schema to ensure the response is {"answer": "..."} across all providers.
        
        response_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"]
        }
        
        
        raw_output = provider.generate(
            system_prompt=system_prompt,
            user_prompt=req.message,
            response_schema=response_schema
        )
        
        answer = raw_output.get("answer", "I could not generate an answer.")
        
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "agent_runtime",
            "iteration": 0,
            "decision": f"Handled chat for {agent_id}",
            "reason": None,
            "data": {"agent_id": agent_id, "user_message": req.message, "answer": answer}
        }
        try:
            with open("events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass
            
        return ChatResponse(answer=answer)
        
    except Exception as e:
        logger.error("Error generating agent response: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

def register_agent(agent_id: str, package: AgentPackage):
    """Register an agent package into the runtime."""
    registered_agents[agent_id] = package
    logger.info("Agent registered: /agents/%s", agent_id)

@app.get("/chat/{agent_id}", response_class=HTMLResponse)
def get_chat_ui(agent_id: str):
    if agent_id not in registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    package = registered_agents[agent_id]
    name = package.get("name", agent_id)
    specialization = package.get("specialization", "Virtual Assistant")
    description = package.get("description", "I am ready to help.")
    starter_questions = package.get("starter_questions", package.get("example_questions", []))
    starter_questions_json = json.dumps(starter_questions)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{name}</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <script>
            tailwind.config = {{
                darkMode: 'class',
                theme: {{
                    fontFamily: {{
                        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'monospace'],
                    }},
                    extend: {{
                        colors: {{
                            vercel: {{
                                ink: '#171717',
                                canvas: '#ffffff',
                                'canvas-soft': '#fafafa',
                                hairline: '#ebebeb',
                                primary: '#0070f3',
                                'dark-canvas': '#000000',
                                'dark-canvas-soft': '#111111',
                                'dark-hairline': '#333333',
                                'dark-ink': '#ededed',
                            }}
                        }}
                    }}
                }}
            }}
        </script>
        <style>
            body {{ font-family: 'Inter', sans-serif; -webkit-font-smoothing: antialiased; }}
            .scroll-hide::-webkit-scrollbar {{ display: none; }}
            
            /* Vercel Light Theme */
            .v-bg-canvas {{ background-color: theme('colors.vercel.canvas'); }}
            .v-bg-canvas-soft {{ background-color: theme('colors.vercel.canvas-soft'); }}
            .v-text-ink {{ color: theme('colors.vercel.ink'); }}
            .v-border-hairline {{ border-color: theme('colors.vercel.hairline'); }}
            
            .v-msg-user {{ background-color: theme('colors.vercel.ink'); color: #fff; }}
            .v-msg-agent {{ background-color: theme('colors.vercel.canvas'); border: 1px solid theme('colors.vercel.hairline'); color: theme('colors.vercel.ink'); }}
            
            /* Vercel Dark Theme */
            .dark .v-bg-canvas {{ background-color: theme('colors.vercel.dark-canvas'); }}
            .dark .v-bg-canvas-soft {{ background-color: theme('colors.vercel.dark-canvas-soft'); }}
            .dark .v-text-ink {{ color: theme('colors.vercel.dark-ink'); }}
            .dark .v-border-hairline {{ border-color: theme('colors.vercel.dark-hairline'); }}
            
            .dark .v-msg-user {{ background-color: theme('colors.vercel.dark-ink'); color: #000; }}
            .dark .v-msg-agent {{ background-color: theme('colors.vercel.dark-canvas-soft'); border: 1px solid theme('colors.vercel.dark-hairline'); color: theme('colors.vercel.dark-ink'); }}
            
            .v-shadow {{ box-shadow: 0 1px 1px rgba(0,0,0,0.02), 0 2px 2px rgba(0,0,0,0.04); }}
            .dark .v-shadow {{ box-shadow: 0 1px 1px rgba(0,0,0,0.2); }}
        </style>
    </head>
    <body class="h-screen w-full flex flex-col v-bg-canvas-soft v-text-ink overflow-hidden transition-colors duration-200">
        
        <!-- Header -->
        <header class="v-bg-canvas v-border-hairline border-b sticky top-0 z-10 px-6 py-4 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="h-8 w-8 rounded-md bg-vercel-ink text-white dark:bg-vercel-dark-ink dark:text-black flex items-center justify-center">
                    <i data-lucide="bot" class="w-5 h-5"></i>
                </div>
                <div>
                    <h1 class="text-sm font-semibold tracking-tight leading-tight">{name}</h1>
                    <p class="text-[11px] text-gray-500 font-mono tracking-tight uppercase">{specialization}</p>
                </div>
            </div>
            <button onclick="toggleTheme()" class="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-zinc-800 transition-colors">
                <i data-lucide="moon" id="theme-icon" class="w-4 h-4 text-gray-500"></i>
            </button>
        </header>

        <!-- Chat Area -->
        <main class="flex-1 overflow-y-auto p-4 sm:p-6 scroll-hide space-y-5 flex flex-col items-center" id="chat-box">
            <!-- Welcome Message -->
            <div class="flex justify-start w-full max-w-3xl">
                <div class="max-w-[85%] sm:max-w-[75%] rounded-xl px-5 py-4 v-msg-agent v-shadow text-[14px] leading-relaxed">
                    <p>I am the <strong class="font-semibold">{name}</strong>. {description}</p>
                </div>
            </div>
        </main>

        <!-- Input Area -->
        <footer class="v-bg-canvas v-border-hairline border-t py-4 px-4 sm:px-6 z-10">
            <div class="max-w-3xl mx-auto flex flex-col gap-3">
                
                <!-- Starter Questions -->
                <div class="flex flex-wrap gap-2 transition-all duration-300" id="starter-questions">
                    <!-- Injected via JS -->
                </div>

                <!-- Input Form -->
                <form onsubmit="sendMessage(event)" class="relative flex items-center">
                    <input type="text" id="user-input" 
                           class="w-full v-bg-canvas-soft v-border-hairline border rounded-md pl-4 pr-12 py-3 text-[14px] v-text-ink placeholder-gray-400 focus:outline-none focus:border-gray-400 dark:focus:border-gray-600 transition-colors"
                           placeholder="Message {name}..." autocomplete="off">
                    <button type="submit" 
                            class="absolute right-2 p-1.5 rounded-md bg-vercel-ink text-white dark:bg-vercel-dark-ink dark:text-black disabled:opacity-30 disabled:cursor-not-allowed transition-opacity"
                            id="send-btn">
                        <i data-lucide="arrow-up" class="w-4 h-4"></i>
                    </button>
                </form>
                <div class="text-center">
                    <p class="text-[11px] text-gray-400 font-mono">Modifai Virtual Mind Runtime</p>
                </div>
            </div>
        </footer>

        <script>
            lucide.createIcons();
            
            // Theme Management
            function initTheme() {{
                const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                if (isDark) document.documentElement.classList.add('dark');
                updateThemeIcon();
            }}
            
            function toggleTheme() {{
                document.documentElement.classList.toggle('dark');
                updateThemeIcon();
            }}
            
            function updateThemeIcon() {{
                const isDark = document.documentElement.classList.contains('dark');
                document.getElementById('theme-icon').setAttribute('data-lucide', isDark ? 'sun' : 'moon');
                lucide.createIcons();
            }}
            
            initTheme();

            const starterQuestions = {starter_questions_json};
            let hasInteracted = false;
            
            // Render starter questions
            const promptsContainer = document.getElementById('starter-questions');
            starterQuestions.forEach(q => {{
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = "px-3 py-1.5 text-[12px] font-medium rounded-full v-bg-canvas v-border-hairline border v-text-ink hover:bg-gray-100 dark:hover:bg-zinc-800 transition-all text-left truncate max-w-full v-shadow";
                btn.textContent = q;
                btn.title = q;
                btn.onclick = () => {{
                    document.getElementById('user-input').value = q;
                    sendMessage(new Event('submit'));
                }};
                promptsContainer.appendChild(btn);
            }});

            async function sendMessage(e) {{
                e?.preventDefault();
                const input = document.getElementById('user-input');
                const btn = document.getElementById('send-btn');
                const text = input.value.trim();
                if (!text) return;

                // Hide starter questions on first interaction
                if (!hasInteracted) {{
                    promptsContainer.style.opacity = '0';
                    promptsContainer.style.height = '0px';
                    promptsContainer.style.overflow = 'hidden';
                    setTimeout(() => promptsContainer.remove(), 300);
                    hasInteracted = true;
                }}

                const chatBox = document.getElementById('chat-box');
                
                // Add user message
                chatBox.innerHTML += `
                    <div class="flex justify-end w-full max-w-3xl animate-in fade-in slide-in-from-bottom-2">
                        <div class="max-w-[85%] sm:max-w-[75%] rounded-xl px-5 py-3 v-msg-user v-shadow text-[14px] leading-relaxed">
                            <p>${{text}}</p>
                        </div>
                    </div>
                `;
                
                input.value = '';
                input.disabled = true;
                btn.disabled = true;
                chatBox.scrollTop = chatBox.scrollHeight;

                // Add loading indicator
                const loadingId = 'loading-' + Date.now();
                chatBox.innerHTML += `
                    <div id="${{loadingId}}" class="flex justify-start w-full max-w-3xl animate-in fade-in">
                        <div class="max-w-[85%] sm:max-w-[75%] rounded-xl px-5 py-4 v-msg-agent v-shadow flex items-center gap-2">
                            <div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-pulse"></div>
                            <div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-pulse" style="animation-delay: 0.2s"></div>
                            <div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-pulse" style="animation-delay: 0.4s"></div>
                        </div>
                    </div>
                `;
                chatBox.scrollTop = chatBox.scrollHeight;

                try {{
                    const response = await fetch('/agents/{agent_id}', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{message: text}})
                    }});
                    const data = await response.json();
                    
                    document.getElementById(loadingId).remove();

                    if (data.answer) {{
                        // Basic markdown parsing
                        const formattedAnswer = data.answer
                            .replace(/\\*\\*(.*?)\\*\\*/g, '<strong class="font-semibold">$1</strong>')
                            .replace(/\\n/g, '<br>');

                        chatBox.innerHTML += `
                            <div class="flex justify-start w-full max-w-3xl animate-in fade-in slide-in-from-bottom-2 mt-2">
                                <div class="max-w-[85%] sm:max-w-[75%] rounded-xl px-5 py-4 v-msg-agent v-shadow text-[14px] leading-relaxed break-words">
                                    <p>${{formattedAnswer}}</p>
                                </div>
                            </div>
                        `;
                    }} else {{
                        throw new Error(JSON.stringify(data));
                    }}
                }} catch (err) {{
                    document.getElementById(loadingId)?.remove();
                    chatBox.innerHTML += `
                        <div class="flex justify-start w-full max-w-3xl mt-2">
                            <div class="max-w-[85%] sm:max-w-[75%] rounded-xl px-5 py-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900 text-red-600 dark:text-red-400 text-[14px]">
                                <p>Error: ${{err.message}}</p>
                            </div>
                        </div>
                    `;
                }}

                input.disabled = false;
                btn.disabled = false;
                input.focus();
                chatBox.scrollTop = chatBox.scrollHeight;
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

