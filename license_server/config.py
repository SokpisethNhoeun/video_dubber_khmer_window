from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    db_path: Path = Path(os.getenv("LICENSE_DB_PATH", "licenses.db"))
    admin_token: str = os.getenv("LICENSE_ADMIN_TOKEN", "")
    session_hours: int = int(os.getenv("ADMIN_SESSION_HOURS", "12"))
    public_url: str = os.getenv("LICENSE_PUBLIC_URL", "http://localhost:8080").rstrip("/")
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "")
    smtp_tls: bool = _bool("SMTP_TLS", True)
    otp_minutes: int = int(os.getenv("OTP_EXPIRES_MINUTES", "5"))
    bootstrap_email: str = os.getenv("ADMIN_BOOTSTRAP_EMAIL", "").strip().lower()
    bootstrap_password: str = os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "")


settings = Settings()
