import io
import json
import urllib.error
from unittest.mock import patch

from licensing.client import LicenseClient


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def test_otp_request_uses_desktop_user_agent():
    client = LicenseClient("https://license.example")

    def fake_urlopen(request, timeout):
        assert timeout == 15.0
        assert request.get_header("User-agent") == "KhmerVideoDubber/1.0 (+desktop-license-client)"
        assert request.get_header("Accept") == "application/json"
        return _Response({"message": "OTP sent", "resend_after_seconds": 60})

    with patch("licensing.client.urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.request_email_otp("person@example.com")

    assert result.success is True
    assert result.message == "OTP sent"


def test_otp_request_surfaces_fastapi_validation_message():
    client = LicenseClient("https://license.example")
    body = json.dumps({"detail": [{"msg": "value is not a valid email address"}]}).encode("utf-8")
    error = urllib.error.HTTPError(
        client.base_url + "/v1/auth/email-otp/request",
        422,
        "Unprocessable Entity",
        {},
        io.BytesIO(body),
    )

    with patch("licensing.client.urllib.request.urlopen", side_effect=error):
        result = client.request_email_otp("invalid")

    assert result.success is False
    assert result.message == "value is not a valid email address"
