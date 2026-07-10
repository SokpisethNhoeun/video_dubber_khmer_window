from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass

from config.user_secrets import load_user_secrets, save_user_secret


def device_fingerprint() -> str:
    parts = [platform.system(), platform.machine(), platform.node(), str(uuid.getnode())]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class LicenseResult:
    valid: bool
    message: str
    plan: str = ""
    expires_at: str = ""


@dataclass(frozen=True)
class CheckoutResult:
    created: bool
    message: str
    checkout_url: str = ""
    reference_id: str = ""
    qr_string: str = ""


@dataclass(frozen=True)
class PaymentStatusResult:
    ok: bool
    status: str
    message: str


@dataclass(frozen=True)
class OtpResult:
    success: bool
    message: str
    verification_token: str = ""
    expires_in_seconds: int = 0
    resend_after_seconds: int = 0


class LicenseClient:
    def __init__(self, base_url: str | None = None, timeout: float = 15.0) -> None:
        self.base_url = (base_url or os.getenv("LICENSE_SERVER_URL", "")).rstrip("/")
        self.timeout = timeout

    @property
    def required(self) -> bool:
        return bool(self.base_url)

    def _post(self, path: str, payload: dict) -> dict:
        if not self.base_url:
            raise RuntimeError("LICENSE_SERVER_URL is not configured.")
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "KhmerVideoDubber/1.0 (+desktop-license-client)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                detail = payload.get("detail", "")
                if isinstance(detail, list):
                    detail = "; ".join(str(item.get("msg", item)) for item in detail)
                elif not isinstance(detail, str):
                    detail = str(detail)
            except Exception:
                detail = ""
            raise RuntimeError(detail or f"License server rejected the request (HTTP {exc.code}).") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not contact the license server: {exc}") from exc

    def _get(self, path: str, timeout: float | None = None) -> dict:
        if not self.base_url:
            raise RuntimeError("LICENSE_SERVER_URL is not configured.")
        request = urllib.request.Request(
            self.base_url + path,
            headers={
                "Accept": "application/json",
                "User-Agent": "KhmerVideoDubber/1.0 (+desktop-license-client)",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                detail = payload.get("detail", "")
                if isinstance(detail, list):
                    detail = "; ".join(str(item.get("msg", item)) for item in detail)
                elif not isinstance(detail, str):
                    detail = str(detail)
            except Exception:
                detail = ""
            raise RuntimeError(detail or f"License server rejected the request (HTTP {exc.code}).") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not contact the license server: {exc}") from exc

    def activate(self, license_key: str) -> LicenseResult:
        try:
            data = self._post("/v1/licenses/activate", {
                "license_key": license_key.strip(),
                "device_id": device_fingerprint(),
                "device_name": platform.node() or platform.system(),
            })
        except RuntimeError as exc:
            return LicenseResult(False, str(exc))
        token = str(data.get("activation_token", ""))
        if not token:
            return LicenseResult(False, "License server returned no activation token.")
        save_user_secret("LICENSE_KEY", license_key.strip())
        save_user_secret("LICENSE_ACTIVATION_TOKEN", token)
        return LicenseResult(True, str(data.get("message", "License activated.")), str(data.get("plan", "")), str(data.get("expires_at", "")))

    def request_email_otp(self, email: str) -> OtpResult:
        try:
            data = self._post("/v1/auth/email-otp/request", {"email": email.strip()})
        except RuntimeError as exc:
            return OtpResult(False, str(exc))
        return OtpResult(
            True,
            str(data.get("message", "Verification code sent. It expires in 5 minutes.")),
            expires_in_seconds=int(data.get("expires_in_seconds", 300)),
            resend_after_seconds=int(data.get("resend_after_seconds", 60)),
        )

    def verify_email_otp(self, email: str, code: str) -> OtpResult:
        try:
            data = self._post("/v1/auth/email-otp/verify", {"email": email.strip(), "code": code.strip()})
        except RuntimeError as exc:
            return OtpResult(False, str(exc))
        token = str(data.get("email_verification_token", ""))
        if not token:
            return OtpResult(False, "Email verification returned no purchase token.")
        return OtpResult(True, str(data.get("message", "Email verified.")), token)

    def create_checkout(
        self, email: str, plan: str, email_verification_token: str = ""
    ) -> CheckoutResult:
        try:
            data = self._post("/v1/payments/checkout", {
                "email": email.strip(),
                "plan": plan,
                "email_verification_token": email_verification_token,
            })
        except RuntimeError as exc:
            return CheckoutResult(False, str(exc))
        checkout_url = str(data.get("checkout_url", ""))
        if not checkout_url:
            return CheckoutResult(False, "Payment server returned no checkout URL.")
        return CheckoutResult(
            True,
            "Bakong KHQR checkout created.",
            checkout_url,
            str(data.get("reference_id", "")),
            str(data.get("qr_string", "") or ""),
        )

    def check_payment_status(self, reference_id: str) -> PaymentStatusResult:
        try:
            data = self._get(f"/v1/payments/{reference_id}/status", timeout=5.0)
        except RuntimeError as exc:
            return PaymentStatusResult(False, "waiting", str(exc))
        return PaymentStatusResult(True, str(data.get("status", "waiting")), str(data.get("message", "")))

    def validate(self) -> LicenseResult:
        if not self.required:
            return LicenseResult(True, "Development mode: license server is not configured.")
        secrets = load_user_secrets()
        token = secrets.get("LICENSE_ACTIVATION_TOKEN", "")
        if not token:
            return LicenseResult(False, "Activate your subscription key before processing videos.")
        try:
            data = self._post("/v1/licenses/validate", {
                "activation_token": token,
                "device_id": device_fingerprint(),
            })
        except RuntimeError as exc:
            return LicenseResult(False, str(exc))
        return LicenseResult(bool(data.get("valid")), str(data.get("message", "")), str(data.get("plan", "")), str(data.get("expires_at", "")))
