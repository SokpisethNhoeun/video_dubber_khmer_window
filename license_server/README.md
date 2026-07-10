# Hosted license service

This service keeps paid license state off the downloadable desktop app. A key
is bound to one device on first activation. The desktop app validates the
activation before starting a video job.

Plans are fixed at USD 11.99 monthly, USD 59.99 for six months, and USD 99.99
yearly. Run behind HTTPS in production.

```bash
python -m venv .venv
.venv/bin/pip install -r license_server/requirements.txt
LICENSE_ADMIN_TOKEN='replace-with-a-long-random-secret' \
LICENSE_DB_PATH=/var/lib/khmer-video-dubber/licenses.db \
.venv/bin/uvicorn license_server.app:app --host 0.0.0.0 --port 8080
```

Set `LICENSE_SERVER_URL=https://license.your-domain.example` in the packaged
desktop app. When it is absent, the source checkout runs in development mode.

After a successful payment webhook, your payment backend should call:

```http
POST /v1/admin/licenses
Authorization: Bearer <LICENSE_ADMIN_TOKEN>
Content-Type: application/json

{"plan":"monthly"}
```

Email the returned license key to the buyer. Do not call the admin endpoint
from the desktop app and never bundle `LICENSE_ADMIN_TOKEN` with downloads.

Before public launch, add your payment provider's signed webhook, email
delivery, admin device reset/revoke endpoints, rate limiting, database backups,
and a privacy policy describing the hashed device identifier.
