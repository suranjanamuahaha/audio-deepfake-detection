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
                user_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected_at TEXT NOT NULL,
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


def create_user(username: str, password_hash: str):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users(username,password_hash,created_at)
            VALUES(?,?,?)
            """,
            (username, password_hash, now),
        )

        return cursor.lastrowid


def get_user_by_username(username: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username,),
        ).fetchone()

        return dict(row) if row else None


def user_count():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM users"
        ).fetchone()

        return row["count"]


def log_spam_detection(user_id, label, confidence):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO spam_history(
                user_id,
                label,
                confidence,
                detected_at
            )
            VALUES(?,?,?,?)
            """,
            (
                user_id,
                label,
                confidence,
                now,
            ),
        )

        return cursor.lastrowid


def get_spam_history(user_id, limit=100):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                label,
                confidence,
                detected_at
            FROM spam_history
            WHERE user_id=?
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (
                user_id,
                limit,
            ),
        ).fetchall()

        return [dict(r) for r in rows]


def is_spam_label(label):
    return label.lower() in {
        "deepfake",
        "fake",
        "spam",
    }