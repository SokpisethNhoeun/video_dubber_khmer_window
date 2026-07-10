from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Khmer Video Dubber License API")
DB_PATH = Path(os.getenv("LICENSE_DB_PATH", "licenses.db"))
PLAN_DAYS = {"monthly": 31, "six_months": 183, "yearly": 366}
PLAN_PRICES_USD = {"monthly": 11.99, "six_months": 59.99, "yearly": 99.99}


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS licenses (
            key_hash TEXT PRIMARY KEY, plan TEXT NOT NULL, expires_at TEXT NOT NULL,
            device_id TEXT, device_name TEXT, activation_hash TEXT, active INTEGER NOT NULL DEFAULT 1
        )
    """)
    return connection


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CreateLicense(BaseModel):
    plan: str = Field(pattern="^(monthly|six_months|yearly)$")


class ActivateLicense(BaseModel):
    license_key: str
    device_id: str
    device_name: str = ""


class ValidateLicense(BaseModel):
    activation_token: str
    device_id: str


def require_admin(value: str | None) -> None:
    expected = os.getenv("LICENSE_ADMIN_TOKEN", "")
    if not expected or not secrets.compare_digest(value or "", expected):
        raise HTTPException(401, "Invalid admin token.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "plans": PLAN_PRICES_USD}


@app.post("/v1/admin/licenses")
def create_license(body: CreateLicense, authorization: str | None = Header(default=None)) -> dict:
    require_admin(authorization.removeprefix("Bearer ").strip() if authorization else None)
    key = "KVD-" + "-".join(secrets.token_hex(3).upper() for _ in range(4))
    expires = datetime.now(timezone.utc) + timedelta(days=PLAN_DAYS[body.plan])
    with db() as connection:
        connection.execute(
            "INSERT INTO licenses (key_hash, plan, expires_at) VALUES (?, ?, ?)",
            (digest(key), body.plan, expires.isoformat()),
        )
    return {"license_key": key, "plan": body.plan, "expires_at": expires.isoformat()}


@app.post("/v1/licenses/activate")
def activate(body: ActivateLicense) -> dict:
    with db() as connection:
        row = connection.execute("SELECT * FROM licenses WHERE key_hash = ?", (digest(body.license_key),)).fetchone()
        if not row or not row["active"]:
            raise HTTPException(404, "License key is invalid or disabled.")
        if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            raise HTTPException(403, "Subscription has expired.")
        if row["device_id"] and row["device_id"] != body.device_id:
            raise HTTPException(409, "This key is already activated on another device.")
        token = secrets.token_urlsafe(32)
        connection.execute(
            "UPDATE licenses SET device_id=?, device_name=?, activation_hash=? WHERE key_hash=?",
            (body.device_id, body.device_name, digest(token), row["key_hash"]),
        )
    return {"activation_token": token, "plan": row["plan"], "expires_at": row["expires_at"], "message": "License activated for this device."}


@app.post("/v1/licenses/validate")
def validate(body: ValidateLicense) -> dict:
    with db() as connection:
        row = connection.execute("SELECT * FROM licenses WHERE activation_hash = ?", (digest(body.activation_token),)).fetchone()
    valid = bool(row and row["active"] and row["device_id"] == body.device_id and datetime.fromisoformat(row["expires_at"]) > datetime.now(timezone.utc))
    if not valid:
        raise HTTPException(403, "License is invalid, expired, or belongs to another device.")
    return {"valid": True, "plan": row["plan"], "expires_at": row["expires_at"], "message": "Subscription is active."}
