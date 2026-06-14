"""
Modifai Backend — FastAPI Application

Serves all API endpoints for the Modifai frontend at /api/v1/*.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import projects, evaluate, compare, stream

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Modifai API",
    description="Backend API for the Modifai LLM fine-tuning platform",
    version="1.0.0",
)

# CORS — allow the Vite dev server and any configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Initializing database...")
    init_db()
    logger.info("Modifai API ready — http://%s:%s", settings.HOST, settings.PORT)


# ── Health Check ────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "service": "modifai-api"}


# ── Mount Routers ───────────────────────────────────────────────────────────────

app.include_router(projects.router, prefix="/api/v1")
app.include_router(evaluate.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
app.include_router(stream.router, prefix="/api/v1")
