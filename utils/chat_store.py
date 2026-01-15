# utils/chat_store.py
"""PostgreSQL‑backed store for chat sessions and their Q&A history.

The original implementation used a JSON file for persistence. This version
stores data in a PostgreSQL database defined by ``Config.DATABASE_URL``. It
creates the required tables on import and provides the same public API:
``add_chat``, ``add_qa`` and ``list_chats``.
"""

import os
import threading
from datetime import datetime
from typing import List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import Config

# A simple thread‑safe connection wrapper. For a production system a connection
# pool (e.g., ``psycopg2.pool.ThreadedConnectionPool``) would be preferable, but
# the lightweight approach below keeps the code easy to understand.

_CONN_LOCK = threading.Lock()
_CONN = None


def _get_connection():
    global _CONN
    with _CONN_LOCK:
        if _CONN is None:
            _CONN = psycopg2.connect(Config.DATABASE_URL, cursor_factory=RealDictCursor)
        return _CONN


def _init_db():
    conn = _get_connection()
    cur = conn.cursor()
    # Create chats table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL
        );
        """
    )
    # Create qa table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS qa (
            id SERIAL PRIMARY KEY,
            chat_id TEXT REFERENCES chats(chat_id) ON DELETE CASCADE,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL
        );
        """
    )
    conn.commit()
    cur.close()


# Initialise tables when the module is imported.
_init_db()


def add_chat(chat_id: str) -> None:
    """Create a new chat entry if it does not already exist."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chats (chat_id, created_at)
        VALUES (%s, %s)
        ON CONFLICT (chat_id) DO NOTHING;
        """,
        (chat_id, datetime.utcnow()),
    )
    conn.commit()
    cur.close()


def add_qa(chat_id: str, question: str, answer: str) -> None:
    """Append a question‑answer pair to the specified chat."""
    # Ensure the chat exists.
    add_chat(chat_id)
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO qa (chat_id, question, answer, timestamp)
        VALUES (%s, %s, %s, %s);
        """,
        (chat_id, question, answer, datetime.utcnow()),
    )
    conn.commit()
    cur.close()


def list_chats(page: int = 1, page_size: int = 5) -> List[Dict[str, Any]]:
    """Return a paginated list of chats with up to five recent Q&A entries.

    Parameters
    ----------
    page: int
        1‑based page number.
    page_size: int
        Number of chats per page.
    """
    conn = _get_connection()
    cur = conn.cursor()
    offset = (page - 1) * page_size
    cur.execute(
        """
        SELECT chat_id, created_at
        FROM chats
        ORDER BY chat_id
        LIMIT %s OFFSET %s;
        """,
        (page_size, offset),
    )
    chats = cur.fetchall()
    result: List[Dict[str, Any]] = []
    for chat in chats:
        cur.execute(
            """
            SELECT question, answer, timestamp
            FROM qa
            WHERE chat_id = %s
            ORDER BY timestamp DESC
            LIMIT 5;
            """,
            (chat["chat_id"],),
        )
        recent_qa = cur.fetchall()
        # Convert timestamps to ISO‑8601 strings for consistency with the JSON version.
        for qa in recent_qa:
            qa["timestamp"] = qa["timestamp"].isoformat() + "Z"
        result.append(
            {
                "chat_id": chat["chat_id"],
                "created_at": chat["created_at"].isoformat() + "Z",
                "qa": recent_qa,
            }
        )
    cur.close()
    return result
