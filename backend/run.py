#!/usr/bin/env python3
"""
Entry point for the Modifai backend server.

Usage:
    python run.py
    # or
    uvicorn app.main:app --reload --port 8000
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
        env_file=".env",
    )
