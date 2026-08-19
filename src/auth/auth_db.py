"""
User Authentication and Data Persistence Database
SQLite-backed user accounts, personal chat sessions, and patient reports.
"""

import os
import json
import sqlite3
import uuid
import time
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path(__file__).parent.parent.parent / "data" / "users.db"


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)

    # User chats table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_chats (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            data_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)

    # Patient reports table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_reports (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            vitals_json TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            uploaded_at INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# User Operations
# ============================================================

def create_user(name: str, email: str, password: str) -> dict:
    name = (name or "").strip()
    email = (email or "").strip().lower()
    if not name or not email or not password:
        raise ValueError("Name, email, and password are required.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Check if email exists
    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cur.fetchone():
        conn.close()
        raise ValueError("An account with this email already exists.")

    user_id = str(uuid.uuid4())
    pw_hash = generate_password_hash(password)
    now = int(time.time() * 1000)

    cur.execute(
        "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, name, email, pw_hash, now)
    )
    conn.commit()
    conn.close()

    return {"id": user_id, "name": name, "email": email, "created_at": now}


def verify_user(email: str, password: str) -> dict | None:
    email = (email or "").strip().lower()
    if not email or not password:
        return None

    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, password_hash, created_at FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    if check_password_hash(row["password_hash"], password):
        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "created_at": row["created_at"]
        }
    return None


def get_user_by_id(user_id: str) -> dict | None:
    if not user_id:
        return None
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


# ============================================================
# Chat Persistence Operations
# ============================================================

def save_user_chat(user_id: str, chat_id: str, title: str, chat_data: dict) -> None:
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    now = int(time.time() * 1000)

    cur.execute("""
        INSERT INTO user_chats (id, user_id, title, data_json, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            data_json = excluded.data_json,
            updated_at = excluded.updated_at
    """, (chat_id, user_id, title, json.dumps(chat_data), now))

    conn.commit()
    conn.close()


def get_user_chats(user_id: str) -> dict:
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, data_json, updated_at FROM user_chats WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()

    chats = {}
    for r in rows:
        try:
            data = json.loads(r["data_json"])
            data["id"] = r["id"]
            data["title"] = r["title"]
            data["updatedAt"] = r["updated_at"]
            chats[r["id"]] = data
        except Exception:
            pass
    return chats


def delete_user_chat(user_id: str, chat_id: str) -> bool:
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM user_chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# ============================================================
# Patient Report Operations
# ============================================================

def save_patient_report(user_id: str, filename: str, vitals_dict: dict, summary_text: str) -> dict:
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    report_id = str(uuid.uuid4())
    now = int(time.time() * 1000)

    cur.execute("""
        INSERT INTO patient_reports (id, user_id, filename, vitals_json, summary_text, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (report_id, user_id, filename, json.dumps(vitals_dict), summary_text, now))

    conn.commit()
    conn.close()

    return {
        "id": report_id,
        "filename": filename,
        "vitals": vitals_dict,
        "summary": summary_text,
        "uploaded_at": now
    }


def get_latest_patient_report(user_id: str) -> dict | None:
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, filename, vitals_json, summary_text, uploaded_at FROM patient_reports WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 1",
        (user_id,)
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    try:
        vitals = json.loads(row["vitals_json"])
    except Exception:
        vitals = {}

    return {
        "id": row["id"],
        "filename": row["filename"],
        "vitals": vitals,
        "summary": row["summary_text"],
        "uploaded_at": row["uploaded_at"]
    }


def delete_patient_report(user_id: str, report_id: str | None = None) -> bool:
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    if report_id:
        cur.execute("DELETE FROM patient_reports WHERE id = ? AND user_id = ?", (report_id, user_id))
    else:
        cur.execute("DELETE FROM patient_reports WHERE user_id = ?", (user_id,))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected > 0
