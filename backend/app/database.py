"""
SQLite database for project metadata.

Uses synchronous sqlite3 — simple, zero-setup, persistent.
The DB file lives at settings.DATABASE_PATH (default: modifai.db).
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

from app.config import settings


# ── Schema ──────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    description        TEXT,
    mode               TEXT NOT NULL,
    intent             TEXT,
    base_model         TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    config             TEXT,
    s3_prefix          TEXT,
    execution_arn      TEXT,
    uploaded_filenames TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
"""


def init_db():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def _get_conn():
    """Yield a sqlite3 connection with row_factory set to dict."""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    # Parse JSON fields back into Python objects
    if d.get("config"):
        try:
            d["config"] = json.loads(d["config"])
        except (json.JSONDecodeError, TypeError):
            d["config"] = {}
    else:
        d["config"] = {}
    if d.get("uploaded_filenames"):
        try:
            d["uploaded_filenames"] = json.loads(d["uploaded_filenames"])
        except (json.JSONDecodeError, TypeError):
            d["uploaded_filenames"] = []
    else:
        d["uploaded_filenames"] = []
    return d


# ── CRUD ────────────────────────────────────────────────────────────────────────

def create_project(
    name: str,
    mode: str,
    description: str | None = None,
    intent: str | None = None,
    base_model: str | None = None,
    config: dict | None = None,
) -> dict:
    """Insert a new project and return it as a dict."""
    project_id = str(uuid.uuid4())
    now = _now()
    s3_prefix = f"projects/{project_id}/"
    config_json = json.dumps(config) if config else None

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO projects (id, name, description, mode, intent, base_model,
                                  status, config, s3_prefix, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
            """,
            (project_id, name, description, mode, intent, base_model,
             config_json, s3_prefix, now, now),
        )
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _row_to_dict(row)


def get_project(project_id: str) -> dict | None:
    """Fetch a single project by ID."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _row_to_dict(row)


def list_projects() -> list[dict]:
    """Return all projects ordered by created_at DESC."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


def update_project(project_id: str, **fields) -> dict | None:
    """Update specific fields on a project. Returns updated project or None."""
    if not fields:
        return get_project(project_id)

    # Serialize JSON fields
    if "config" in fields and isinstance(fields["config"], dict):
        fields["config"] = json.dumps(fields["config"])
    if "uploaded_filenames" in fields and isinstance(fields["uploaded_filenames"], list):
        fields["uploaded_filenames"] = json.dumps(fields["uploaded_filenames"])

    fields["updated_at"] = _now()

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [project_id]

    with _get_conn() as conn:
        conn.execute(
            f"UPDATE projects SET {set_clause} WHERE id = ?",
            values,
        )
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _row_to_dict(row)


def delete_project(project_id: str) -> bool:
    """Delete a project. Returns True if deleted, False if not found."""
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0
