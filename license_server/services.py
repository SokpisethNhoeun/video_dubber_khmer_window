from __future__ import annotations

import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from license_server.database import connect, now_iso
from license_server.security import digest

PLAN_DAYS = {"monthly": 31, "six_months": 183, "yearly": 366}
PLAN_PRICES_USD = {"monthly": "11.99", "six_months": "59.99", "yearly": "99.99"}


def audit(admin: dict, action: str, resource_type: str, resource_id: str = "", details: dict | None = None, ip: str = "") -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO audit_logs(admin_id,action,resource_type,resource_id,details,ip_address,created_at) VALUES(?,?,?,?,?,?,?)",
            (admin.get("id"), action, resource_type, resource_id, json.dumps(details or {}), ip, now_iso()),
        )


def public_license(row) -> dict:
    value = dict(row)
    value.pop("key_hash", None)
    value.pop("activation_hash", None)
    value["active"] = bool(value["active"])
    return value


def create_license(plan: str, email: str = "", note: str = "", payment_reference: str = "") -> tuple[str, dict]:
    key = "KVD-" + "-".join(secrets.token_hex(3).upper() for _ in range(4))
    identifier = str(uuid.uuid4())
    stamp = now_iso()
    with connect() as connection:
        connection.execute(
            """INSERT INTO licenses(key_hash,id,plan,duration_days,expires_at,active,created_at,updated_at,customer_email,payment_reference,admin_note)
               VALUES(?,?,?,?,NULL,1,?,?,?,?,?)""",
            (digest(key), identifier, plan, PLAN_DAYS[plan], stamp, stamp, email or None, payment_reference or None, note),
        )
        row = connection.execute("SELECT * FROM licenses WHERE id=?", (identifier,)).fetchone()
    return key, public_license(row)


def confirm_payment_and_create_license(reference: str, admin_id: str) -> tuple[str, dict, dict]:
    """Atomically mark one payment paid and issue exactly one license."""
    key = "KVD-" + "-".join(secrets.token_hex(3).upper() for _ in range(4))
    identifier = str(uuid.uuid4())
    stamp = now_iso()
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        payment = connection.execute("SELECT * FROM payments WHERE reference_id=?", (reference,)).fetchone()
        if not payment:
            raise HTTPException(404, "Payment not found.")
        if payment["status"] == "paid":
            raise HTTPException(409, "Payment was already confirmed.")
        if payment["status"] != "waiting":
            raise HTTPException(409, "Payment cannot be confirmed from its current state.")
        connection.execute(
            """INSERT INTO licenses(key_hash,id,plan,duration_days,expires_at,active,created_at,updated_at,customer_email,payment_reference,admin_note)
               VALUES(?,?,?,?,NULL,1,?,?,?,?,?)""",
            (digest(key), identifier, payment["plan"], PLAN_DAYS[payment["plan"]], stamp, stamp, payment["email"], reference, ""),
        )
        connection.execute(
            "UPDATE payments SET status='paid',confirmed_at=?,confirmed_by=?,license_id=? WHERE reference_id=?",
            (stamp, admin_id, identifier, reference),
        )
        license_row = connection.execute("SELECT * FROM licenses WHERE id=?", (identifier,)).fetchone()
    return key, public_license(license_row), dict(payment)


def get_license(identifier: str):
    with connect() as connection:
        row = connection.execute("SELECT * FROM licenses WHERE id=?", (identifier,)).fetchone()
    if not row:
        raise HTTPException(404, "License not found.")
    return row


def set_license_state(identifier: str, active: bool) -> dict:
    get_license(identifier)
    stamp = now_iso()
    with connect() as connection:
        connection.execute(
            "UPDATE licenses SET active=?, revoked_at=?, updated_at=? WHERE id=?",
            (int(active), None if active else stamp, stamp, identifier),
        )
    return public_license(get_license(identifier))


def activate_license(key: str, device_id: str, device_name: str) -> tuple[str, dict]:
    now = datetime.now(timezone.utc)
    with connect() as connection:
        row = connection.execute("SELECT * FROM licenses WHERE key_hash=?", (digest(key),)).fetchone()
        if not row or not row["active"]:
            raise HTTPException(404, "License key is invalid or disabled.")
        if row["expires_at"] and datetime.fromisoformat(row["expires_at"]) <= now:
            raise HTTPException(403, "Subscription has expired.")
        if row["device_id"] and row["device_id"] != device_id:
            raise HTTPException(409, "This key is already activated on another device.")
        expires = row["expires_at"] or (now + timedelta(days=row["duration_days"] or PLAN_DAYS[row["plan"]])).isoformat()
        activated = row["activated_at"] or now.isoformat()
        token = secrets.token_urlsafe(32)
        connection.execute(
            "UPDATE licenses SET device_id=?,device_name=?,activation_hash=?,activated_at=?,expires_at=?,updated_at=? WHERE key_hash=?",
            (device_id, device_name, digest(token), activated, expires, now.isoformat(), row["key_hash"]),
        )
    return token, {"plan": row["plan"], "expires_at": expires, "activated_at": activated}
