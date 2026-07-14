from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from license_server.config import settings
from license_server.security import hash_password


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path or settings.db_path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=15000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}


def _add_column(connection: sqlite3.Connection, table: str, definition: str) -> None:
    name = definition.split()[0]
    if name not in _columns(connection, table):
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def migrate(path: Path | None = None) -> None:
    with connect(path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS licenses (
                key_hash TEXT PRIMARY KEY, plan TEXT NOT NULL, expires_at TEXT,
                device_id TEXT, device_name TEXT, activation_hash TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
        """)
        for definition in (
            "id TEXT", "duration_days INTEGER", "created_at TEXT", "activated_at TEXT",
            "customer_email TEXT", "payment_reference TEXT", "admin_note TEXT",
            "updated_at TEXT", "revoked_at TEXT"
        ):
            _add_column(connection, "licenses", definition)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_licenses_id ON licenses(id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_licenses_activation_hash ON licenses(activation_hash)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_licenses_email ON licenses(customer_email)")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_licenses_payment_reference ON licenses(payment_reference) WHERE payment_reference IS NOT NULL")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
                role TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
                last_login_at TEXT
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                token_hash TEXT PRIMARY KEY, admin_id TEXT NOT NULL, expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL, FOREIGN KEY(admin_id) REFERENCES admins(id) ON DELETE CASCADE
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, admin_id TEXT, action TEXT NOT NULL,
                resource_type TEXT NOT NULL, resource_id TEXT, details TEXT,
                ip_address TEXT, created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS email_otps (
                email TEXT PRIMARY KEY, code_hash TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                expires_at TEXT NOT NULL, resend_at TEXT NOT NULL, created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS email_verifications (
                token_hash TEXT PRIMARY KEY, email TEXT NOT NULL, expires_at TEXT NOT NULL,
                used_at TEXT, created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                reference_id TEXT PRIMARY KEY, email TEXT NOT NULL, plan TEXT NOT NULL,
                amount_usd TEXT NOT NULL, status TEXT NOT NULL, checkout_url TEXT,
                created_at TEXT NOT NULL, confirmed_at TEXT, confirmed_by TEXT,
                license_id TEXT UNIQUE
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS promo_codes (
                code TEXT PRIMARY KEY, discount_percent INTEGER NOT NULL,
                max_uses INTEGER, uses_count INTEGER NOT NULL DEFAULT 0,
                expires_at TEXT, active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        _add_column(connection, "payments", "promo_code TEXT")
        stamp = now_iso()
        connection.execute("UPDATE licenses SET id=lower(hex(randomblob(16))) WHERE id IS NULL")
        connection.execute("UPDATE licenses SET duration_days=CASE plan WHEN 'monthly' THEN 31 WHEN 'six_months' THEN 183 ELSE 366 END WHERE duration_days IS NULL")
        connection.execute("UPDATE licenses SET created_at=? WHERE created_at IS NULL", (stamp,))
        connection.execute("UPDATE licenses SET updated_at=? WHERE updated_at IS NULL", (stamp,))

        if settings.bootstrap_email and settings.bootstrap_password:
            existing = connection.execute("SELECT 1 FROM admins LIMIT 1").fetchone()
            if not existing:
                import uuid
                connection.execute(
                    "INSERT INTO admins(id,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (str(uuid.uuid4()), settings.bootstrap_email, hash_password(settings.bootstrap_password), "owner", stamp),
                )
