"""
Configuration settings for the Modifai backend.
All values are read from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # ── AWS ─────────────────────────────────────────────────────────────────────
    AWS_REGION: str = os.environ.get("AWS_REGION", "ap-south-1")
    S3_BUCKET: str = os.environ.get("S3_BUCKET", "modifai-data")

    # Step Functions state machine ARN
    STATE_MACHINE_ARN: str = os.environ.get(
        "STATE_MACHINE_ARN",
        ""  # Must be set for pipeline execution
    )

    # ── LLM (OpenRouter) ────────────────────────────────────────────────────────
    OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
    OR_SECRET_NAME: str = os.environ.get("OR_SECRET_NAME", "modifai/or")
    OR_PRIMARY_MODEL: str = os.environ.get("OR_MODEL", "deepseek/deepseek-chat-v3")
    OR_FALLBACK_1: str = os.environ.get("OR_FALLBACK_1", "qwen/qwen3-235b-a22b")
    OR_FALLBACK_2: str = os.environ.get("OR_FALLBACK_2", "google/gemini-2.5-flash-lite")
    OR_MAX_RETRIES: int = int(os.environ.get("OR_MAX_RETRIES", "3"))

    # ── Database ────────────────────────────────────────────────────────────────
    DATABASE_PATH: str = os.environ.get("DATABASE_PATH", "modifai.db")

    # ── Server ──────────────────────────────────────────────────────────────────
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", "8000"))
    CORS_ORIGINS: list[str] = os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")

    # ── Pipeline Defaults ───────────────────────────────────────────────────────
    DEFAULT_SAMPLES_PER_CHUNK: int = int(os.environ.get("DEFAULT_SAMPLES_PER_CHUNK", "5"))
    DEFAULT_QUALITY_THRESHOLD: float = float(os.environ.get("DEFAULT_QUALITY_THRESHOLD", "0.7"))


settings = Settings()
