"""
db.py — PostgreSQL session storage for Agent OTG
=================================================

All chat messages are stored securely in a local PostgreSQL database instead
of plain JSON files. Only localhost connections are used (no network exposure).

Database schema (auto-created on first run):

  agent_sessions
  ─────────────────────────────────────────
  id            SERIAL PRIMARY KEY
  session_name  VARCHAR(255) UNIQUE NOT NULL
  created_at    TIMESTAMPTZ DEFAULT NOW()

  agent_messages
  ─────────────────────────────────────────
  id            SERIAL PRIMARY KEY
  session_name  VARCHAR(255) NOT NULL  → FK to agent_sessions
  role          VARCHAR(50)  NOT NULL  ('user' or 'assistant')
  content       TEXT         NOT NULL
  created_at    TIMESTAMPTZ DEFAULT NOW()

Configuration:
  Edit db_config.json to set your PostgreSQL credentials.
  All keys: host, port, dbname, user, password.
"""

import os
import json
import threading
from typing import Optional

# ── Config loading ─────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "db_config.json")


def _load_config() -> dict:
    """Load PostgreSQL credentials from db_config.json."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg
    except FileNotFoundError:
        raise RuntimeError(
            f"db_config.json not found at {_CONFIG_PATH}. "
            "Create it with your PostgreSQL credentials."
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"db_config.json is not valid JSON: {exc}")


# ── Connection pool (thread-local single connection per thread) ────────────────
_local = threading.local()
_db_ready: Optional[bool] = None   # None = not yet checked
_db_error: str = ""


def _get_conn():
    """
    Return a live psycopg2 connection for the current thread.
    Reconnects automatically if the connection is closed or broken.
    """
    import psycopg2
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed:
        cfg  = _load_config()
        conn = psycopg2.connect(
            host     = cfg.get("host",     "localhost"),
            port     = int(cfg.get("port", 5432)),
            dbname   = cfg.get("dbname",   "agent_otg"),
            user     = cfg.get("user",     "postgres"),
            password = cfg.get("password", ""),
            connect_timeout = 5,
            application_name = "agent_otg",
        )
        conn.autocommit = False
        _local.conn = conn
    return conn


# ── Schema bootstrap ───────────────────────────────────────────────────────────

_DDL_SESSIONS = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    id           SERIAL PRIMARY KEY,
    session_name VARCHAR(255) UNIQUE NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_DDL_MESSAGES = """
CREATE TABLE IF NOT EXISTS agent_messages (
    id           SERIAL PRIMARY KEY,
    session_name VARCHAR(255) NOT NULL
                    REFERENCES agent_sessions(session_name)
                    ON DELETE CASCADE,
    role         VARCHAR(50)  NOT NULL,
    content      TEXT         NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
"""

_DDL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON agent_messages(session_name, created_at);
"""


def init_db() -> bool:
    """
    Connect to PostgreSQL and create tables if they don't exist.
    Returns True on success, False if the database is unavailable.
    Sets the module-level _db_ready flag so callers can check availability.
    """
    global _db_ready, _db_error
    if _db_ready is True:
        return True

    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(_DDL_SESSIONS)
            cur.execute(_DDL_MESSAGES)
            cur.execute(_DDL_INDEX)
        conn.commit()
        _db_ready = True
        _db_error = ""
        print("[db] PostgreSQL session store initialised OK.", flush=True)
        return True
    except Exception as exc:
        _db_ready = False
        _db_error = str(exc)
        print(f"[db] WARNING: PostgreSQL unavailable — {exc}", flush=True)
        print("[db]   Sessions will not be persisted. Check db_config.json.", flush=True)
        return False


def is_ready() -> bool:
    """Return True if the database was successfully initialised."""
    return _db_ready is True


def get_error() -> str:
    """Return the last DB error message (empty string if no error)."""
    return _db_error


# ── Session operations ─────────────────────────────────────────────────────────

def create_session(session_name: str) -> bool:
    """
    Register a new session in agent_sessions.
    Returns True on success.
    """
    if not is_ready():
        return False
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_sessions (session_name) VALUES (%s) ON CONFLICT DO NOTHING",
                (session_name,),
            )
        conn.commit()
        return True
    except Exception as exc:
        print(f"[db] create_session error: {exc}", flush=True)
        try:
            _get_conn().rollback()
        except Exception:
            pass
        return False


def save_message(session_name: str, role: str, content: str) -> bool:
    """
    Append one message to agent_messages.
    Returns True on success.
    """
    if not is_ready():
        return False
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_messages (session_name, role, content)
                VALUES (%s, %s, %s)
                """,
                (session_name, role, content),
            )
        conn.commit()
        return True
    except Exception as exc:
        print(f"[db] save_message error: {exc}", flush=True)
        try:
            _get_conn().rollback()
        except Exception:
            pass
        return False


def get_all_sessions(limit: int = 20) -> list[dict]:
    """
    Return a list of sessions (most recent first) with message counts.
    Each dict has: session_name, created_at, message_count.
    """
    if not is_ready():
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    s.session_name,
                    s.created_at,
                    COUNT(m.id)          AS message_count,
                    MAX(m.created_at)    AS last_activity
                FROM agent_sessions s
                LEFT JOIN agent_messages m ON m.session_name = s.session_name
                GROUP BY s.session_name, s.created_at
                ORDER BY s.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "session_name":   row[0],
                "created_at":     row[1].strftime("%Y-%m-%d %H:%M:%S") if row[1] else "",
                "message_count":  int(row[2]),
                "last_activity":  row[3].strftime("%Y-%m-%d %H:%M:%S") if row[3] else "",
            }
            for row in rows
        ]
    except Exception as exc:
        print(f"[db] get_all_sessions error: {exc}", flush=True)
        return []


def get_session_messages(session_name: str, limit: int = 50) -> list[dict]:
    """
    Return the last `limit` messages for a given session (oldest first).
    Each dict has: role, content, created_at.
    """
    if not is_ready():
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, created_at
                FROM (
                    SELECT role, content, created_at
                    FROM agent_messages
                    WHERE session_name = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub
                ORDER BY created_at ASC
                """,
                (session_name, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "role":       row[0],
                "content":    row[1],
                "created_at": row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else "",
            }
            for row in rows
        ]
    except Exception as exc:
        print(f"[db] get_session_messages error: {exc}", flush=True)
        return []


def get_session_count() -> int:
    """Return total number of sessions in the database."""
    if not is_ready():
        return 0
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM agent_sessions")
            return int(cur.fetchone()[0])
    except Exception:
        return 0


def delete_session(session_name: str) -> bool:
    """
    Delete a session and all its messages (CASCADE).
    Returns True on success.
    """
    if not is_ready():
        return False
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM agent_sessions WHERE session_name = %s",
                (session_name,),
            )
        conn.commit()
        return True
    except Exception as exc:
        print(f"[db] delete_session error: {exc}", flush=True)
        try:
            _get_conn().rollback()
        except Exception:
            pass
        return False
