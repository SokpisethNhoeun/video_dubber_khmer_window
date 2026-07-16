from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from license_server.config import settings
from license_server.database import connect, migrate, now_iso
from license_server.main import app
from license_server.security import digest


@pytest.fixture()
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_path = tmp_path / "licenses.db"
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "admin_token", "test-token")
    migrate()
    return TestClient(app)


def test_promo_code_lifecycle_and_checkout(client: TestClient) -> None:
    # 1. Create a promo code
    headers = {"Authorization": "Bearer test-token"}
    res = client.post(
        "/v1/admin/promo-codes",
        headers=headers,
        json={
            "code": "SAVE20",
            "discount_percent": 20,
            "max_uses": 2,
        },
    )
    assert res.status_code == 200
    assert res.json()["payload"]["code"] == "SAVE20"
    assert res.json()["payload"]["discount_percent"] == 20

    # 2. List promo codes
    res = client.get("/v1/admin/promo-codes", headers=headers)
    assert res.status_code == 200
    items = res.json()["payload"]
    assert len(items) == 1
    assert items[0]["code"] == "SAVE20"
    assert items[0]["uses_count"] == 0

    # 3. Create a verification token for checkout
    verification_token = "valid-verification-token-12345678"
    expires = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    with connect() as connection:
        connection.execute(
            "INSERT INTO email_verifications(token_hash,email,expires_at,created_at) VALUES(?,?,?,?)",
            (digest(verification_token), "buyer@example.com", expires, now_iso()),
        )

    # 4. Checkout with valid promo code (usage 1)
    res = client.post(
        "/v1/payments/checkout",
        json={
            "email": "buyer@example.com",
            "plan": "monthly",
            "email_verification_token": verification_token,
            "promo_code": "SAVE20",
        },
    )
    assert res.status_code == 200
    ref_id = res.json()["reference_id"]
    assert ref_id.startswith("PAY-")

    # Verify amount_usd was discounted: 11.99 - 20% = 9.59
    with connect() as connection:
        pay_row = connection.execute("SELECT * FROM payments WHERE reference_id=?", (ref_id,)).fetchone()
        assert pay_row["amount_usd"] == "9.59"
        assert pay_row["promo_code"] == "SAVE20"

        promo_row = connection.execute("SELECT * FROM promo_codes WHERE code='SAVE20'").fetchone()
        assert promo_row["uses_count"] == 1

    # 5. Create another verification token for checkout (usage 2)
    verification_token_2 = "valid-verification-token-87654321"
    with connect() as connection:
        connection.execute(
            "INSERT INTO email_verifications(token_hash,email,expires_at,created_at) VALUES(?,?,?,?)",
            (digest(verification_token_2), "buyer@example.com", expires, now_iso()),
        )

    # Checkout again (usage 2)
    res = client.post(
        "/v1/payments/checkout",
        json={
            "email": "buyer@example.com",
            "plan": "monthly",
            "email_verification_token": verification_token_2,
            "promo_code": "SAVE20",
        },
    )
    assert res.status_code == 200

    # 6. Create third verification token for checkout (exhausted limit check)
    verification_token_3 = "valid-verification-token-11223344"
    with connect() as connection:
        connection.execute(
            "INSERT INTO email_verifications(token_hash,email,expires_at,created_at) VALUES(?,?,?,?)",
            (digest(verification_token_3), "buyer@example.com", expires, now_iso()),
        )

    # Checkout third time (should fail due to max_uses limit)
    res = client.post(
        "/v1/payments/checkout",
        json={
            "email": "buyer@example.com",
            "plan": "monthly",
            "email_verification_token": verification_token_3,
            "promo_code": "SAVE20",
        },
    )
    assert res.status_code == 400
    assert "usage limit reached" in res.json()["detail"]

    # 7. Create expired promo code
    past_expiry = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    res = client.post(
        "/v1/admin/promo-codes",
        headers=headers,
        json={
            "code": "EXPIRED",
            "discount_percent": 10,
            "expires_at": past_expiry,
        },
    )
    assert res.status_code == 200

    # Verification token for expired test
    verification_token_expired = "expired-verification-token"
    with connect() as connection:
        connection.execute(
            "INSERT INTO email_verifications(token_hash,email,expires_at,created_at) VALUES(?,?,?,?)",
            (digest(verification_token_expired), "buyer@example.com", expires, now_iso()),
        )

    # Checkout with expired promo code
    res = client.post(
        "/v1/payments/checkout",
        json={
            "email": "buyer@example.com",
            "plan": "monthly",
            "email_verification_token": verification_token_expired,
            "promo_code": "EXPIRED",
        },
    )
    assert res.status_code == 400
    assert "has expired" in res.json()["detail"]

    # 8. Delete promo code
    res = client.delete("/v1/admin/promo-codes/SAVE20", headers=headers)
    assert res.status_code == 200

    res = client.get("/v1/admin/promo-codes", headers=headers)
    items = res.json()["payload"]
    # Only EXPIRED is left
    assert len(items) == 1
    assert items[0]["code"] == "EXPIRED"


def test_validate_promo_endpoint(client: TestClient) -> None:
    # Create a promo code
    headers = {"Authorization": "Bearer test-token"}
    client.post(
        "/v1/admin/promo-codes",
        headers=headers,
        json={"code": "OFF50", "discount_percent": 50},
    )
    
    # Test valid promo validation
    res = client.post("/v1/payments/validate-promo", json={"promo_code": "OFF50", "plan": "monthly"})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert res.json()["discount_percent"] == 50
    
    # Test invalid promo validation
    res = client.post("/v1/payments/validate-promo", json={"promo_code": "INVALID", "plan": "monthly"})
    assert res.status_code == 400
    assert "invalid or disabled" in res.json()["detail"]


def test_checkout_generates_scannable_qr_code_with_minimum_amount(client: TestClient) -> None:
    # Create a 100% discount promo code
    headers = {"Authorization": "Bearer test-token"}
    client.post(
        "/v1/admin/promo-codes",
        headers=headers,
        json={"code": "FREE", "discount_percent": 100},
    )
    
    verification_token = "verification-token-free-plan"
    expires = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    with connect() as connection:
        connection.execute(
            "INSERT INTO email_verifications(token_hash,email,expires_at,created_at) VALUES(?,?,?,?)",
            (digest(verification_token), "buyer@example.com", expires, now_iso()),
        )
        
    res = client.post(
        "/v1/payments/checkout",
        json={
            "email": "buyer@example.com",
            "plan": "monthly",
            "email_verification_token": verification_token,
            "promo_code": "FREE",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "qr_string" in data
    # The qr_string should be a Bakong link with at least 0.01 amount
    assert "amount=0.01" in data["qr_string"]

