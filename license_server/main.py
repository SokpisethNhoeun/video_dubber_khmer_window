from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from license_server.api import router
from license_server.database import migrate, now_iso


@asynccontextmanager
async def lifespan(_app: FastAPI):
    migrate()
    yield


app = FastAPI(title="Khmer Video Dubber License API", version="2.0.0", lifespan=lifespan)
app.include_router(router)


@app.get("/manual-payment/{reference}", response_class=HTMLResponse, include_in_schema=False)
def manual_payment(reference: str) -> str:
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
    <title>Payment {reference}</title></head><body style='font-family:sans-serif;max-width:600px;margin:80px auto;padding:20px'>
    <h1>Payment pending</h1><p>Reference: <strong>{reference}</strong></p>
    <p>Complete payment using the current payment instructions, then wait for an administrator to confirm it.</p></body></html>"""


@app.exception_handler(Exception)
async def unexpected_error(request: Request, _exc: Exception):
    return JSONResponse(status_code=500, content={"success": False, "message": "An unexpected server error occurred.", "errors": {}, "path": request.url.path, "timestamp": now_iso()})
