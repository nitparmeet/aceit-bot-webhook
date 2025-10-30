# analytics.py
import sqlite3
import datetime

DB_PATH = "analytics.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            event TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def upsert_user(user):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (user_id, username, first_name, last_name, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            username=excluded.username,
            first_name=excluded.first_name,
            last_name=excluded.last_name
    """, (
        user.id,
        user.username,
        user.first_name,
        user.last_name,
        datetime.datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def log_event(user_id, event):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events (user_id, event, timestamp)
        VALUES (?, ?, ?)
    """, (user_id, event, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
