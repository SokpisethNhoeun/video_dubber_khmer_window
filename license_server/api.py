from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from license_server.config import settings
from license_server.database import connect, now_iso
from license_server.dependencies import current_admin, rate_limit, require_roles
from license_server.mail import send_email
from license_server.schemas import (
    ActivateLicenseRequest, CheckoutRequest, ConfirmPaymentRequest, CreateAdminRequest,
    CreateLicenseRequest, ExtendLicenseRequest, LoginRequest, NoteRequest, OtpRequest,
    OtpVerifyRequest, ValidateLicenseRequest, CreatePromoCodeRequest, ValidatePromoRequest,
)
from license_server.security import digest, hash_password, verify_password
from license_server.services import (
    PLAN_PRICES_USD, activate_license, audit, confirm_payment_and_create_license,
    create_license, get_license, public_license, set_license_state,
)

router = APIRouter()


def payload(data=None, message: str = "Request successful") -> dict:
    return {"success": True, "message": message, "payload": data, "timestamp": now_iso()}


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict:
    with connect() as connection:
        connection.execute("SELECT 1").fetchone()
    return {"status": "ready"}


@router.post("/v1/admin/auth/login")
def admin_login(body: LoginRequest, request: Request) -> dict:
    rate_limit(request, "admin-login", 8, 300)
    with connect() as connection:
        admin = connection.execute("SELECT * FROM admins WHERE email=?", (str(body.email).lower(),)).fetchone()
        if not admin or not admin["active"] or not verify_password(body.password, admin["password_hash"]):
            raise HTTPException(401, "Invalid email or password.")
        token = secrets.token_urlsafe(48)
        expires = datetime.now(timezone.utc) + timedelta(hours=settings.session_hours)
        connection.execute(
            "INSERT INTO admin_sessions(token_hash,admin_id,expires_at,created_at) VALUES(?,?,?,?)",
            (digest(token), admin["id"], expires.isoformat(), now_iso()),
        )
        connection.execute("UPDATE admins SET last_login_at=? WHERE id=?", (now_iso(), admin["id"]))
    result = {"access_token": token, "expires_at": expires.isoformat(), "admin": {"email": admin["email"], "role": admin["role"]}}
    audit(dict(admin), "login", "admin", admin["id"], ip=request.client.host if request.client else "")
    return payload(result, "Signed in.")


@router.post("/v1/admin/auth/logout")
def admin_logout(request: Request, admin: dict = Depends(current_admin)) -> dict:
    auth = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if admin["id"] != "legacy-token":
        with connect() as connection:
            connection.execute("DELETE FROM admin_sessions WHERE token_hash=?", (digest(auth),))
    return payload(None, "Signed out.")


@router.get("/v1/admin/me")
def admin_me(admin: dict = Depends(current_admin)) -> dict:
    return payload({"id": admin["id"], "email": admin["email"], "role": admin["role"]})


@router.post("/v1/admin/admins")
def add_admin(body: CreateAdminRequest, admin: dict = Depends(require_roles("owner"))) -> dict:
    identifier = str(uuid.uuid4())
    try:
        with connect() as connection:
            connection.execute(
                "INSERT INTO admins(id,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                (identifier, str(body.email).lower(), hash_password(body.password), body.role, now_iso()),
            )
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise HTTPException(409, "An admin with this email already exists.") from exc
        raise
    audit(admin, "create", "admin", identifier, {"email": str(body.email), "role": body.role})
    return payload({"id": identifier, "email": str(body.email), "role": body.role}, "Admin created.")


@router.post("/v1/admin/licenses")
def add_license(body: CreateLicenseRequest, request: Request, admin: dict = Depends(require_roles("owner", "support", "finance"))) -> dict:
    key, license_data = create_license(body.plan, str(body.customer_email or ""), body.admin_note)
    audit(admin, "create", "license", license_data["id"], {"plan": body.plan}, request.client.host if request.client else "")
    return payload({"license_key": key, **license_data}, "License created.")


@router.get("/v1/admin/licenses")
def list_licenses(
    q: str = Query(default="", max_length=200), status: str = Query(default="all", pattern="^(all|active|revoked|expired|unactivated)$"),
    page: int = Query(default=1, ge=1), page_size: int = Query(default=25, ge=1, le=100),
    admin: dict = Depends(current_admin),
) -> dict:
    clauses, values = [], []
    if q:
        clauses.append("(customer_email LIKE ? OR id LIKE ? OR device_name LIKE ? OR payment_reference LIKE ?)")
        values.extend([f"%{q}%"] * 4)
    if status == "active": clauses.append("active=1 AND (expires_at IS NULL OR expires_at>?)"); values.append(now_iso())
    elif status == "revoked": clauses.append("active=0")
    elif status == "expired": clauses.append("expires_at IS NOT NULL AND expires_at<=?"); values.append(now_iso())
    elif status == "unactivated": clauses.append("activated_at IS NULL")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with connect() as connection:
        total = connection.execute("SELECT count(*) AS n FROM licenses" + where, values).fetchone()["n"]
        rows = connection.execute(
            "SELECT * FROM licenses" + where + " ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*values, page_size, (page - 1) * page_size],
        ).fetchall()
    return payload({"items": [public_license(row) for row in rows], "total": total, "page": page, "page_size": page_size})


@router.get("/v1/admin/licenses/{identifier}")
def license_detail(identifier: str, admin: dict = Depends(current_admin)) -> dict:
    return payload(public_license(get_license(identifier)))


def _state_action(identifier: str, active: bool, admin: dict, action: str) -> dict:
    result = set_license_state(identifier, active)
    audit(admin, action, "license", identifier)
    return payload(result, f"License {action}d.")


@router.post("/v1/admin/licenses/{identifier}/revoke")
def revoke_license(identifier: str, admin: dict = Depends(require_roles("owner", "support"))) -> dict:
    return _state_action(identifier, False, admin, "revoke")


@router.post("/v1/admin/licenses/{identifier}/restore")
def restore_license(identifier: str, admin: dict = Depends(require_roles("owner", "support"))) -> dict:
    return _state_action(identifier, True, admin, "restore")


@router.post("/v1/admin/licenses/{identifier}/reset-device")
def reset_device(identifier: str, admin: dict = Depends(require_roles("owner", "support"))) -> dict:
    get_license(identifier)
    with connect() as connection:
        connection.execute("UPDATE licenses SET device_id=NULL,device_name=NULL,activation_hash=NULL,updated_at=? WHERE id=?", (now_iso(), identifier))
    audit(admin, "reset_device", "license", identifier)
    return payload(public_license(get_license(identifier)), "Device binding reset.")


@router.post("/v1/admin/licenses/{identifier}/extend")
def extend_license(identifier: str, body: ExtendLicenseRequest, admin: dict = Depends(require_roles("owner", "finance"))) -> dict:
    row = get_license(identifier)
    base = max(datetime.now(timezone.utc), datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else datetime.now(timezone.utc))
    expires = base + timedelta(days=body.days)
    with connect() as connection:
        connection.execute("UPDATE licenses SET expires_at=?,updated_at=? WHERE id=?", (expires.isoformat(), now_iso(), identifier))
    audit(admin, "extend", "license", identifier, {"days": body.days})
    return payload(public_license(get_license(identifier)), "License extended.")


@router.patch("/v1/admin/licenses/{identifier}/note")
def update_note(identifier: str, body: NoteRequest, admin: dict = Depends(require_roles("owner", "support"))) -> dict:
    get_license(identifier)
    with connect() as connection:
        connection.execute("UPDATE licenses SET admin_note=?,updated_at=? WHERE id=?", (body.admin_note, now_iso(), identifier))
    audit(admin, "update_note", "license", identifier)
    return payload(public_license(get_license(identifier)), "Note updated.")


@router.post("/v1/licenses/activate")
def activate(body: ActivateLicenseRequest, request: Request) -> dict:
    rate_limit(request, "activate", 15, 300)
    token, info = activate_license(body.license_key.strip().upper(), body.device_id, body.device_name)
    return {"activation_token": token, **info, "message": "License activated for this device."}


@router.post("/v1/licenses/validate")
def validate(body: ValidateLicenseRequest, request: Request) -> dict:
    rate_limit(request, "validate", 120, 60)
    with connect() as connection:
        row = connection.execute("SELECT * FROM licenses WHERE activation_hash=?", (digest(body.activation_token),)).fetchone()
    valid = bool(row and row["active"] and row["device_id"] == body.device_id and row["expires_at"] and datetime.fromisoformat(row["expires_at"]) > datetime.now(timezone.utc))
    if not valid:
        raise HTTPException(403, "License is invalid, expired, or belongs to another device.")
    return {"valid": True, "plan": row["plan"], "expires_at": row["expires_at"], "activated_at": row["activated_at"], "message": "Subscription is active."}


@router.post("/v1/auth/email-otp/request")
def request_otp(body: OtpRequest, request: Request) -> dict:
    rate_limit(request, "otp", 5, 600)
    email = str(body.email).lower()
    now = datetime.now(timezone.utc)
    with connect() as connection:
        existing = connection.execute("SELECT * FROM email_otps WHERE email=?", (email,)).fetchone()
        if existing and datetime.fromisoformat(existing["resend_at"]) > now:
            raise HTTPException(429, "Wait before requesting another verification code.")
        code = f"{secrets.randbelow(1_000_000):06d}"
        expires = now + timedelta(minutes=settings.otp_minutes)
        resend = now + timedelta(seconds=60)
        connection.execute(
            "INSERT OR REPLACE INTO email_otps(email,code_hash,attempts,expires_at,resend_at,created_at) VALUES(?,?,0,?,?,?)",
            (email, digest(code), expires.isoformat(), resend.isoformat(), now.isoformat()),
        )
    try:
        send_email(email, "Khmer Video Dubber verification code", f"Your verification code is {code}. It expires in {settings.otp_minutes} minutes.")
    except RuntimeError as exc:
        raise HTTPException(503, "Email delivery is not available right now. Please contact support to receive your verification code.") from exc
    return {"message": "Verification code sent. Check your inbox.", "expires_in_seconds": settings.otp_minutes * 60, "resend_after_seconds": 60}


@router.post("/v1/auth/email-otp/verify")
def verify_otp(body: OtpVerifyRequest, request: Request) -> dict:
    rate_limit(request, "otp-verify", 10, 600)
    email = str(body.email).lower()
    with connect() as connection:
        row = connection.execute("SELECT * FROM email_otps WHERE email=?", (email,)).fetchone()
        if not row or row["attempts"] >= 5 or datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            raise HTTPException(400, "Verification code is invalid or expired.")
        if not secrets.compare_digest(row["code_hash"], digest(body.code)):
            connection.execute("UPDATE email_otps SET attempts=attempts+1 WHERE email=?", (email,))
            raise HTTPException(400, "Verification code is invalid or expired.")
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(minutes=30)
        connection.execute("DELETE FROM email_otps WHERE email=?", (email,))
        connection.execute("INSERT INTO email_verifications(token_hash,email,expires_at,created_at) VALUES(?,?,?,?)", (digest(token), email, expires.isoformat(), now_iso()))
    return {"email_verification_token": token, "message": "Email verified."}


@router.post("/v1/payments/checkout")
def checkout(body: CheckoutRequest, request: Request) -> dict:
    rate_limit(request, "checkout", 10, 600)
    email = str(body.email).lower()
    with connect() as connection:
        verified = connection.execute(
            "SELECT * FROM email_verifications WHERE token_hash=? AND email=? AND used_at IS NULL AND expires_at>?",
            (digest(body.email_verification_token), email, now_iso()),
        ).fetchone()
        if not verified:
            raise HTTPException(403, "Email verification is invalid or expired.")

        discount_percent = 0
        promo_code = body.promo_code.strip().upper() if body.promo_code else None
        if promo_code:
            promo = connection.execute("SELECT * FROM promo_codes WHERE code=?", (promo_code,)).fetchone()
            if not promo or not promo["active"]:
                raise HTTPException(400, "Promo code is invalid or disabled.")
            if promo["expires_at"] and datetime.fromisoformat(promo["expires_at"]) <= datetime.now(timezone.utc):
                raise HTTPException(400, "Promo code has expired.")
            if promo["max_uses"] is not None and promo["uses_count"] >= promo["max_uses"]:
                raise HTTPException(400, "Promo code usage limit reached.")
            discount_percent = promo["discount_percent"]

        base_price = float(PLAN_PRICES_USD[body.plan])
        amount = base_price * (1.0 - discount_percent / 100.0)
        amount_usd = f"{amount:.2f}"

        reference = "PAY-" + secrets.token_hex(8).upper()
        url = f"{settings.public_url}/manual-payment/{reference}"
        
        connection.execute(
            "INSERT INTO payments(reference_id,email,plan,amount_usd,status,checkout_url,created_at,promo_code) VALUES(?,?,?,?,?,?,?,?)",
            (reference, email, body.plan, amount_usd, "waiting", url, now_iso(), promo_code)
        )
        connection.execute(
            "UPDATE email_verifications SET used_at=? WHERE token_hash=?",
            (now_iso(), digest(body.email_verification_token))
        )
        if promo_code:
            connection.execute(
                "UPDATE promo_codes SET uses_count=uses_count+1 WHERE code=?",
                (promo_code,)
            )

        qr_amount = max(0.01, amount)
        qr_string = f"https://link.bakong.org.kh/pay?id=khmer_video_dubber@pay&name=KhmerVideoDubber&amount={qr_amount:.2f}&currency=USD&reference={reference}"

    return {"checkout_url": url, "reference_id": reference, "qr_string": qr_string, "message": "Payment request created; awaiting manual confirmation."}


@router.post("/v1/payments/validate-promo")
def validate_promo(body: ValidatePromoRequest) -> dict:
    promo_code = body.promo_code.strip().upper()
    with connect() as connection:
        promo = connection.execute("SELECT * FROM promo_codes WHERE code=?", (promo_code,)).fetchone()
        if not promo or not promo["active"]:
            raise HTTPException(400, "Promo code is invalid or disabled.")
        if promo["expires_at"] and datetime.fromisoformat(promo["expires_at"]) <= datetime.now(timezone.utc):
            raise HTTPException(400, "Promo code has expired.")
        if promo["max_uses"] is not None and promo["uses_count"] >= promo["max_uses"]:
            raise HTTPException(400, "Promo code usage limit reached.")
        
        return {
            "success": True,
            "discount_percent": promo["discount_percent"],
            "message": f"Promo code {promo_code} applied successfully! ({promo['discount_percent']}% off)"
        }



@router.get("/v1/payments/{reference}/status")
def payment_status(reference: str) -> dict:
    with connect() as connection:
        row = connection.execute("SELECT status FROM payments WHERE reference_id=?", (reference,)).fetchone()
    if not row:
        raise HTTPException(404, "Payment not found.")
    message = "Payment confirmed. Your license key was sent by email." if row["status"] == "paid" else "Waiting for payment confirmation."
    return {"status": row["status"], "message": message}


@router.get("/v1/admin/payments")
def list_payments(status: str = Query(default="", pattern="^(|waiting|paid|cancelled|refunded)$"), admin: dict = Depends(current_admin)) -> dict:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM payments" + (" WHERE status=?" if status else "") + " ORDER BY created_at DESC LIMIT 200", (status,) if status else ()).fetchall()
    return payload([dict(row) for row in rows])


@router.post("/v1/admin/payments/{reference}/confirm")
def confirm_payment(reference: str, body: ConfirmPaymentRequest, admin: dict = Depends(require_roles("owner", "finance"))) -> dict:
    key, license_data, payment = confirm_payment_and_create_license(reference, admin["id"])
    try:
        send_email(payment["email"], "Your Khmer Video Dubber license", f"Payment received. Your license key is:\n\n{key}\n\nKeep this key private.")
    except RuntimeError as exc:
        audit(admin, "email_failed", "payment", reference, {"error": str(exc)})
        raise HTTPException(502, "Payment confirmed and license created, but the email could not be sent.") from exc
    audit(admin, "confirm", "payment", reference, {"license_id": license_data["id"]})
    return payload({"reference_id": reference, "license_id": license_data["id"]}, "Payment confirmed and license emailed.")


@router.post("/v1/admin/promo-codes")
def add_promo_code(body: CreatePromoCodeRequest, admin: dict = Depends(require_roles("owner", "finance"))) -> dict:
    code = body.code.strip().upper()
    try:
        with connect() as connection:
            connection.execute(
                "INSERT INTO promo_codes(code,discount_percent,max_uses,expires_at,created_at) VALUES(?,?,?,?,?)",
                (code, body.discount_percent, body.max_uses, body.expires_at, now_iso()),
            )
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise HTTPException(409, "Promo code already exists.") from exc
        raise
    audit(admin, "create", "promo_code", code, {"discount_percent": body.discount_percent})
    return payload({"code": code, "discount_percent": body.discount_percent}, "Promo code created.")


@router.get("/v1/admin/promo-codes")
def list_promo_codes(admin: dict = Depends(require_roles("owner", "finance", "support"))) -> dict:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM promo_codes ORDER BY created_at DESC").fetchall()
    return payload([dict(row) for row in rows])


@router.delete("/v1/admin/promo-codes/{code}")
def delete_promo_code(code: str, admin: dict = Depends(require_roles("owner", "finance"))) -> dict:
    code = code.strip().upper()
    with connect() as connection:
        row = connection.execute("SELECT * FROM promo_codes WHERE code=?", (code,)).fetchone()
        if not row:
            raise HTTPException(404, "Promo code not found.")
        connection.execute("DELETE FROM promo_codes WHERE code=?", (code,))
    audit(admin, "delete", "promo_code", code)
    return payload(None, "Promo code deleted.")


@router.get("/v1/admin/audit-logs")
def audit_logs(limit: int = Query(default=100, ge=1, le=500), admin: dict = Depends(require_roles("owner"))) -> dict:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return payload([dict(row) for row in rows])
