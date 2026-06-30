import os
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS spam_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                name TEXT DEFAULT 'Default Key',
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------- USERS ---------------- #

def create_user(username: str, password_hash: str) -> int:
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, now),
        )
        return cursor.lastrowid


def get_user_by_username(username: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        return dict(row) if row else None


def get_user_by_id(user_id: int):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

        return dict(row) if row else None


def user_count() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM users"
        ).fetchone()

        return row["count"]


# ---------------- API KEYS ---------------- #

def create_api_key(user_id: int, api_key: str, name: str = "Default Key"):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO api_keys (user_id, api_key, name, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, api_key, name, now),
        )

        return cursor.lastrowid


def get_api_key(api_key: str):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM api_keys
            WHERE api_key = ? AND active = 1
            """,
            (api_key,),
        ).fetchone()

        return dict(row) if row else None


def list_api_keys(user_id: int):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, created_at, last_used, active
            FROM api_keys
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()

        return [dict(row) for row in rows]


def revoke_api_key(key_id: int):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE api_keys
            SET active = 0
            WHERE id = ?
            """,
            (key_id,),
        )


def update_last_used(api_key: str):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE api_keys
            SET last_used = ?
            WHERE api_key = ?
            """,
            (now, api_key),
        )


def log_spam_detection(label: str, confidence: float) -> int:
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO spam_history (label, confidence, detected_at)
            VALUES (?, ?, ?)
            """,
            (label, confidence, now),
        )

        return cursor.lastrowid


def get_spam_history(limit: int = 100):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, label, confidence, detected_at
            FROM spam_history
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]


def is_spam_label(label: str) -> bool:
    return label.lower() in {"deepfake", "fake", "spam"}