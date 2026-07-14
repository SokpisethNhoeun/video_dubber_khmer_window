from __future__ import annotations

from pathlib import Path

import pytest
import requests

from modules.model_downloader import DownloadPaused, ModelDownloadManager


class FakeSibling:
    rfilename = "model.bin"
    size = 10


class FakeInfo:
    siblings = [FakeSibling()]


class FakeResponse:
    def __init__(self, payload: bytes, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = {"ETag": '"blob123"', "X-Repo-Commit": "commit123"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def close(self):
        return None

    def iter_content(self, _size):
        yield self.payload[:5]
        yield self.payload[5:]


def test_range_resume_keeps_partial_bytes(monkeypatch, tmp_path: Path) -> None:
    payload = b"0123456789"
    requests_seen = []
    manager = ModelDownloadManager("small", cache_dir=tmp_path, chunk_size=5)
    monkeypatch.setattr("modules.model_downloader.HfApi.model_info", lambda *_a, **_k: FakeInfo())

    first = True
    def fake_get(_url, headers, **_kwargs):
        nonlocal first
        requests_seen.append(dict(headers))
        if first:
            first = False
            response = FakeResponse(payload)
            original = response.iter_content
            def pause_after_first(size):
                for chunk in original(size):
                    yield chunk
                    manager.pause()
            response.iter_content = pause_after_first
            return response
        return FakeResponse(payload[5:], 206)

    monkeypatch.setattr("modules.model_downloader.requests.get", fake_get)
    with pytest.raises(DownloadPaused):
        manager.download()
    manager.resume()
    snapshot = manager.download()
    assert (snapshot / "model.bin").read_bytes() == payload
    assert requests_seen[-1]["Range"] == "bytes=5-"


def test_download_stops_before_network_when_disk_space_is_too_low(monkeypatch, tmp_path: Path) -> None:
    manager = ModelDownloadManager("small", cache_dir=tmp_path)
    monkeypatch.setattr("modules.model_downloader.HfApi.model_info", lambda *_a, **_k: FakeInfo())
    monkeypatch.setattr(
        "modules.model_downloader.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 1})(),
    )
    network_called = False

    def fake_get(*_args, **_kwargs):
        nonlocal network_called
        network_called = True
        raise AssertionError("network should not be used")

    monkeypatch.setattr("modules.model_downloader.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="Not enough disk space"):
        manager.download()

    assert network_called is False


def test_stream_failure_retries_from_partial_file(monkeypatch, tmp_path: Path) -> None:
    payload = b"0123456789"
    manager = ModelDownloadManager("small", cache_dir=tmp_path, chunk_size=5)
    monkeypatch.setattr("modules.model_downloader.HfApi.model_info", lambda *_a, **_k: FakeInfo())
    monkeypatch.setattr("modules.model_downloader.time.sleep", lambda _seconds: None)
    requests_seen = []

    def fake_get(_url, headers, **_kwargs):
        requests_seen.append(dict(headers))
        if len(requests_seen) == 1:
            response = FakeResponse(payload)

            def interrupted(_size):
                yield payload[:5]
                raise requests.ConnectionError("connection dropped")

            response.iter_content = interrupted
            return response
        return FakeResponse(payload[5:], status_code=206)

    monkeypatch.setattr("modules.model_downloader.requests.get", fake_get)

    snapshot = manager.download()

    assert (snapshot / "model.bin").read_bytes() == payload
    assert requests_seen == [{}, {"Range": "bytes=5-"}]


def test_metadata_timeout_retries_then_reports_network_help(monkeypatch, tmp_path: Path) -> None:
    manager = ModelDownloadManager("small", cache_dir=tmp_path)
    attempts = 0

    def timeout(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise requests.Timeout("metadata timeout")

    monkeypatch.setattr("modules.model_downloader.HfApi.model_info", timeout)
    monkeypatch.setattr("modules.model_downloader.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="firewall, proxy, or VPN"):
        manager.download()

    assert attempts == 3


def test_unknown_model_size_still_reports_download_activity(monkeypatch, tmp_path: Path) -> None:
    payload = b"0123456789"
    manager = ModelDownloadManager("small", cache_dir=tmp_path, chunk_size=5)
    info = type("Info", (), {"siblings": [type("Sibling", (), {"rfilename": "model.bin", "size": None})()]})()
    monkeypatch.setattr("modules.model_downloader.HfApi.model_info", lambda *_a, **_k: info)
    monkeypatch.setattr(
        "modules.model_downloader.requests.get",
        lambda *_args, **_kwargs: FakeResponse(payload),
    )
    progress = []

    manager.download(lambda filename, done, total, speed, eta: progress.append((filename, done, total)))

    assert progress[-1] == ("model.bin", len(payload), 0)
