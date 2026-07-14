from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")

from license_server.config import settings
from license_server.database import connect, migrate, now_iso
from license_server.security import digest, hash_password, verify_password
from license_server.services import (
    activate_license, confirm_payment_and_create_license, create_license, set_license_state,
)


@pytest.fixture()
def license_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "licenses.db")
    migrate()
    return settings.db_path


def test_license_duration_starts_on_first_activation(license_db):
    key, created = create_license("monthly", "buyer@example.com")
    assert created["expires_at"] is None
    token, activated = activate_license(key, "device-identifier-1234", "Laptop")
    assert token
    expiry = datetime.fromisoformat(activated["expires_at"])
    assert 30 <= (expiry - datetime.now(timezone.utc)).days <= 31


def test_revoke_invalidates_activation_record(license_db):
    key, created = create_license("yearly")
    token, _ = activate_license(key, "device-identifier-1234", "")
    set_license_state(created["id"], False)
    with connect() as connection:
        row = connection.execute("SELECT * FROM licenses WHERE activation_hash=?", (digest(token),)).fetchone()
    assert row["active"] == 0


def test_admin_password_hash_is_salted_and_verifiable():
    first = hash_password("correct-horse-battery")
    second = hash_password("correct-horse-battery")
    assert first != second
    assert verify_password("correct-horse-battery", first)
    assert not verify_password("incorrect-password", first)


def test_legacy_schema_is_migrated_without_changing_expiration(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    expected = "2030-01-01T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE licenses(key_hash TEXT PRIMARY KEY, plan TEXT NOT NULL, expires_at TEXT NOT NULL, device_id TEXT, device_name TEXT, activation_hash TEXT, active INTEGER NOT NULL DEFAULT 1)")
        connection.execute("INSERT INTO licenses(key_hash,plan,expires_at) VALUES('hash','monthly',?)", (expected,))
    monkeypatch.setattr(settings, "db_path", path)
    migrate()
    with connect() as connection:
        row = connection.execute("SELECT * FROM licenses WHERE key_hash='hash'").fetchone()
    assert row["id"]
    assert row["duration_days"] == 31
    assert row["expires_at"] == expected


def test_payment_confirmation_is_atomic_and_idempotent(license_db):
    with connect() as connection:
        connection.execute("INSERT INTO payments(reference_id,email,plan,amount_usd,status,created_at) VALUES(?,?,?,?,?,?)", ("PAY-TEST", "buyer@example.com", "monthly", "11.99", "waiting", now_iso()))
    key, license_data, payment = confirm_payment_and_create_license("PAY-TEST", "owner-id")
    assert key.startswith("KVD-")
    assert payment["email"] == "buyer@example.com"
    with connect() as connection:
        row = connection.execute("SELECT * FROM payments WHERE reference_id='PAY-TEST'").fetchone()
        assert row["license_id"] == license_data["id"]
    with pytest.raises(Exception) as error:
        confirm_payment_and_create_license("PAY-TEST", "owner-id")
    assert getattr(error.value, "status_code", None) == 409
