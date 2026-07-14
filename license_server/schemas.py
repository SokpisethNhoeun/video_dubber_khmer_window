from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


Plan = str


class EmailModel(BaseModel):
    @field_validator("email", "customer_email", check_fields=False)
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        local, separator, domain = normalized.partition("@")
        if not separator or not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
            raise ValueError("Enter a valid email address.")
        return normalized


class LoginRequest(EmailModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=10, max_length=256)


class CreateLicenseRequest(EmailModel):
    plan: str = Field(pattern="^(monthly|six_months|yearly)$")
    customer_email: str | None = Field(default=None, min_length=5, max_length=254)
    admin_note: str = Field(default="", max_length=1000)


class ActivateLicenseRequest(BaseModel):
    license_key: str = Field(min_length=8, max_length=64)
    device_id: str = Field(min_length=16, max_length=256)
    device_name: str = Field(default="", max_length=200)


class ValidateLicenseRequest(BaseModel):
    activation_token: str = Field(min_length=20, max_length=256)
    device_id: str = Field(min_length=16, max_length=256)


class ExtendLicenseRequest(BaseModel):
    days: int = Field(ge=1, le=3660)


class NoteRequest(BaseModel):
    admin_note: str = Field(max_length=1000)


class OtpRequest(EmailModel):
    email: str = Field(min_length=5, max_length=254)


class OtpVerifyRequest(EmailModel):
    email: str = Field(min_length=5, max_length=254)
    code: str = Field(pattern=r"^\d{6}$")


class CheckoutRequest(EmailModel):
    email: str = Field(min_length=5, max_length=254)
    plan: str = Field(pattern="^(monthly|six_months|yearly)$")
    email_verification_token: str = Field(min_length=20, max_length=256)
    promo_code: str | None = Field(default=None, max_length=30)


class CreatePromoCodeRequest(BaseModel):
    code: str = Field(min_length=3, max_length=30, pattern=r"^[A-Z0-9_-]+$")
    discount_percent: int = Field(ge=1, le=100)
    max_uses: int | None = Field(default=None, ge=1)
    expires_at: str | None = Field(default=None)


class ConfirmPaymentRequest(BaseModel):
    checkout_url: str | None = Field(default=None, max_length=2000)


class CreateAdminRequest(EmailModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=12, max_length=256)
    role: str = Field(pattern="^(owner|support|finance)$")
