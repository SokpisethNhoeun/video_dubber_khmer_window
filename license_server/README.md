# Hosted license and admin service

The FastAPI service owns licenses, admin sessions, email verification, manual
payment records, and audit history. Existing SQLite databases are migrated at
startup without replacing legacy expiration dates. New subscriptions begin
their plan duration on first activation.

## Development

```bash
python -m pip install -r license_server/requirements.txt
LICENSE_DB_PATH=/tmp/kvd-licenses.db \
ADMIN_BOOTSTRAP_EMAIL=owner@example.com \
ADMIN_BOOTSTRAP_PASSWORD='replace-with-a-long-random-password' \
uvicorn license_server.app:app --host 127.0.0.1 --port 8080
```

The bootstrap account is created only when no admin exists. Remove its password
from the runtime environment after the initial startup. `LICENSE_ADMIN_TOKEN`
remains supported for old automation, but individual admin sessions are
recommended because they provide roles and audit attribution.

## SMTP

Configure `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`,
`SMTP_FROM`, and `SMTP_TLS`. OTP and license delivery fail safely when SMTP is
not configured. Credentials belong only on the hosted server.

## Payments

This repository does not integrate an external payment gateway. Checkout
creates a `waiting` payment reference. An owner or finance admin must verify the
payment outside this system and confirm it in the dashboard. Confirmation is
one-time, creates the license, and emails the key. Never confirm a payment based
only on a customer-provided screenshot.

## Admin dashboard

```bash
cd admin_web
cp .env.example .env.local
npm install
npm run dev
```

`API_BASE_URL` is server-only. The dashboard stores the admin session in an
HTTP-only cookie and calls the backend through Server Actions and server-side
services; browser components never receive backend credentials.

For production, use HTTPS, a reverse proxy, encrypted backups, centralized
rate limiting, and PostgreSQL before scaling to multiple API instances.
