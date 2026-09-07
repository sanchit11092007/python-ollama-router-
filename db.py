"""Local SQLite session and memory store.

The previous PostgreSQL-only implementation made an otherwise offline product
lose persistence whenever a separate database server was unavailable.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


_DB_PATH = Path(__file__).with_name("agent_otg.sqlite3")
_local = threading.local()
_ready = False
_error = ""


def _connection() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


def init_db() -> bool:
    global _ready, _error
    try:
        conn = _connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_name TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL REFERENCES agent_sessions(session_name) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON agent_messages(session_name, id);
            CREATE TABLE IF NOT EXISTS project_memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        _ready, _error = True, ""
        return True
    except Exception as exc:
        _ready, _error = False, str(exc)
        return False


def is_ready() -> bool:
    return _ready or init_db()


def get_error() -> str:
    return _error


def create_session(session_name: str) -> bool:
    try:
        if not is_ready():
            return False
        _connection().execute("INSERT OR IGNORE INTO agent_sessions(session_name) VALUES (?)", (session_name,))
        _connection().commit()
        return True
    except Exception:
        return False


def save_message(session_name: str, role: str, content: str) -> bool:
    try:
        if not create_session(session_name):
            return False
        _connection().execute(
            "INSERT INTO agent_messages(session_name, role, content) VALUES (?, ?, ?)",
            (session_name, role, content),
        )
        _connection().commit()
        return True
    except Exception:
        return False


def get_all_sessions(limit: int = 20) -> list[dict]:
    if not is_ready():
        return []
    rows = _connection().execute("""
        SELECT s.session_name, s.created_at, COUNT(m.id) AS message_count,
               COALESCE(MAX(m.created_at), s.created_at) AS last_activity
        FROM agent_sessions s LEFT JOIN agent_messages m ON m.session_name = s.session_name
        GROUP BY s.session_name ORDER BY last_activity DESC LIMIT ?
    """, (max(1, min(limit, 200)),)).fetchall()
    return [dict(row) for row in rows]


def get_session_messages(session_name: str, limit: int = 50) -> list[dict]:
    if not is_ready():
        return []
    rows = _connection().execute("""
        SELECT role, content, created_at FROM (
            SELECT role, content, created_at, id FROM agent_messages
            WHERE session_name = ? ORDER BY id DESC LIMIT ?
        ) ORDER BY id ASC
    """, (session_name, max(1, min(limit, 500)))).fetchall()
    return [dict(row) for row in rows]


def get_session_count() -> int:
    return int(_connection().execute("SELECT COUNT(*) FROM agent_sessions").fetchone()[0]) if is_ready() else 0


def delete_session(session_name: str) -> bool:
    try:
        _connection().execute("DELETE FROM agent_sessions WHERE session_name = ?", (session_name,))
        _connection().commit()
        return True
    except Exception:
        return False


def set_project_memory(key: str, value: str) -> bool:
    try:
        _connection().execute("""
            INSERT INTO project_memory(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
        """, (key, value))
        _connection().commit()
        return True
    except Exception:
        return False


def get_project_memory(key: str) -> str | None:
    row = _connection().execute("SELECT value FROM project_memory WHERE key = ?", (key,)).fetchone() if is_ready() else None
    return str(row[0]) if row else None
