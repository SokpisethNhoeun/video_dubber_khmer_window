from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, Request

from license_server.config import settings
from license_server.database import connect
from license_server.security import digest

_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def rate_limit(request: Request, bucket: str, limit: int, seconds: int) -> None:
    key = f"{bucket}:{request.client.host if request.client else 'unknown'}"
    now = time.monotonic()
    with _lock:
        values = _hits[key]
        while values and values[0] <= now - seconds:
            values.popleft()
        if len(values) >= limit:
            raise HTTPException(429, "Too many requests. Try again later.")
        values.append(now)


def current_admin(authorization: str | None = Header(default=None)) -> dict:
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if settings.admin_token and secrets.compare_digest(token, settings.admin_token):
        return {"id": "legacy-token", "email": "legacy-admin", "role": "owner"}
    if not token:
        raise HTTPException(401, "Authentication required.")
    with connect() as connection:
        row = connection.execute(
            "SELECT a.* FROM admin_sessions s JOIN admins a ON a.id=s.admin_id WHERE s.token_hash=? AND s.expires_at>? AND a.active=1",
            (digest(token), datetime.now(timezone.utc).isoformat()),
        ).fetchone()
    if not row:
        raise HTTPException(401, "Admin session is invalid or expired.")
    return dict(row)


def require_roles(*roles: str):
    def dependency(admin: dict = Depends(current_admin)) -> dict:
        if admin["role"] not in roles:
            raise HTTPException(403, "You do not have permission for this operation.")
        return admin
    return dependency
