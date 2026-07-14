"""Compatibility entry point for ``uvicorn license_server.app:app``."""

from license_server.main import app

__all__ = ["app"]
