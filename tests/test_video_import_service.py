from __future__ import annotations

from pathlib import Path

import pytest

from modules.video_import.base import DownloadResult
from modules.video_import.errors import UnsupportedUrlError, VideoImportError
from modules.video_import.service import VideoImportService


class FakeProvider:
    name = "xiaohongshu"
    display_name = "Xiaohongshu / RedNote"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, Path | None]] = []

    def normalize_url(self, url: str) -> str:
        return url

    def content_id(self, url: str) -> str | None:
        return "abc123"

    def download(self, url: str, dest_dir: Path, *, cookies_file=None, cancel_event=None, progress_hook=None) -> Path:
        self.calls.append((url, dest_dir, cookies_file))
        dest_dir.mkdir(parents=True, exist_ok=True)
        output = dest_dir / "source.mp4"
        output.write_bytes(b"mp4")
        if progress_hook is not None:
            progress_hook(url, {"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
            progress_hook(url, {"status": "finished"})
        return output


class FakeTitleProvider(FakeProvider):
    def download(self, url: str, dest_dir: Path, *, cookies_file=None, cancel_event=None, progress_hook=None) -> DownloadResult:
        output = super().download(
            url,
            dest_dir,
            cookies_file=cookies_file,
            cancel_event=cancel_event,
            progress_hook=progress_hook,
        )
        return DownloadResult(output, "A source title")


class FakeGoogleTranslator:
    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target

    def translate(self, text: str) -> str:
        assert self.source == "auto"
        assert self.target == "km"
        assert text == "A source title"
        return "ចំណងជើងពីប្រភព"


def test_service_raises_for_unsupported_url(tmp_path: Path) -> None:
    service = VideoImportService(tmp_path / "cache")

    with pytest.raises(UnsupportedUrlError):
        service.import_video("ftp://example.com/video/123")


def test_service_uses_cached_video_without_redownloading(monkeypatch, tmp_path: Path) -> None:
    provider = FakeProvider()
    cache_dir = tmp_path / "cache"
    cached_dir = cache_dir / provider.name / "abc123"
    cached_dir.mkdir(parents=True)
    cached_file = cached_dir / "source.mp4"
    cached_file.write_bytes(b"cached")

    import modules.video_import.service as service_module

    monkeypatch.setattr(service_module, "resolve_provider", lambda url: provider)
    monkeypatch.setattr(service_module, "has_audio_stream", lambda path: True)

    service = VideoImportService(cache_dir)
    result = service.import_video("https://www.rednote.com/explore/abc123")

    assert result == cached_file
    assert provider.calls == []


def test_service_downloads_and_reports_progress(monkeypatch, tmp_path: Path) -> None:
    provider = FakeProvider()
    progress: list[tuple[str, int]] = []

    import modules.video_import.service as service_module

    monkeypatch.setattr(service_module, "resolve_provider", lambda url: provider)
    monkeypatch.setattr(service_module, "has_audio_stream", lambda path: True)

    service = VideoImportService(tmp_path / "cache")
    result = service.import_video(
        "https://www.rednote.com/explore/abc123",
        progress=lambda url, value: progress.append((url, value)),
    )

    assert result.exists()
    assert provider.calls
    assert progress[0] == ("https://www.rednote.com/explore/abc123", 50)
    assert progress[-1] == ("https://www.rednote.com/explore/abc123", 100)
    assert VideoImportService.source_url_for_video(result) == "https://www.rednote.com/explore/abc123"


def test_service_renames_import_with_preferred_stem(monkeypatch, tmp_path: Path) -> None:
    provider = FakeProvider()

    import modules.video_import.service as service_module

    monkeypatch.setattr(service_module, "resolve_provider", lambda url: provider)
    monkeypatch.setattr(service_module, "has_audio_stream", lambda path: True)

    service = VideoImportService(tmp_path / "cache")
    result = service.import_video(
        "https://www.rednote.com/explore/abc123",
        preferred_stem="import_1",
    )

    assert result.name == "import_1.mp4"
    assert result.exists()
    assert not (result.parent / "source.mp4").exists()
    assert VideoImportService.source_url_for_video(result) == "https://www.rednote.com/explore/abc123"


def test_service_names_import_from_translated_source_title(monkeypatch, tmp_path: Path) -> None:
    provider = FakeTitleProvider()

    import modules.video_import.service as service_module

    monkeypatch.setattr(service_module, "resolve_provider", lambda url: provider)
    monkeypatch.setattr(service_module, "has_audio_stream", lambda path: True)
    monkeypatch.setitem(
        __import__("sys").modules,
        "deep_translator",
        type("FakeDeepTranslator", (), {"GoogleTranslator": FakeGoogleTranslator})(),
    )

    service = VideoImportService(tmp_path / "cache")
    result = service.import_video(
        "https://www.rednote.com/explore/abc123",
        preferred_stem="import_1",
    )

    assert result.name == "ចំណងជើងពីប្រភព.mp4"
    metadata = (result.parent / "import.json").read_text(encoding="utf-8")
    assert '"source_title": "A source title"' in metadata
    assert '"khmer_title": "ចំណងជើងពីប្រភព"' in metadata


def test_service_renames_old_source_cache_once(monkeypatch, tmp_path: Path) -> None:
    provider = FakeProvider()
    cache_dir = tmp_path / "cache"
    cached_dir = cache_dir / provider.name / "abc123"
    cached_dir.mkdir(parents=True)
    cached_file = cached_dir / "source.mp4"
    cached_file.write_bytes(b"cached")

    import modules.video_import.service as service_module

    monkeypatch.setattr(service_module, "resolve_provider", lambda url: provider)
    monkeypatch.setattr(service_module, "has_audio_stream", lambda path: True)

    service = VideoImportService(cache_dir)
    first = service.import_video("https://www.rednote.com/explore/abc123", preferred_stem="import_1")
    second = service.import_video("https://www.rednote.com/explore/abc123", preferred_stem="import_2")

    assert first == second
    assert first.name == "import_1.mp4"
    assert first.exists()
    assert not cached_file.exists()
    assert provider.calls == []


def test_service_rejects_missing_cookies_file(monkeypatch, tmp_path: Path) -> None:
    provider = FakeProvider()

    import modules.video_import.service as service_module

    monkeypatch.setattr(service_module, "resolve_provider", lambda url: provider)
    monkeypatch.setattr(service_module, "has_audio_stream", lambda path: True)

    service = VideoImportService(tmp_path / "cache")
    with pytest.raises(VideoImportError, match="Cookies file not found"):
        service.import_video(
            "https://www.rednote.com/explore/abc123",
            cookies_file=tmp_path / "missing.txt",
        )


def test_service_cleans_only_import_cache_video(tmp_path: Path) -> None:
    service = VideoImportService(tmp_path / "cache" / "imports")
    import_dir = tmp_path / "cache" / "imports" / "xiaohongshu" / "abc123"
    import_dir.mkdir(parents=True)
    video = import_dir / "video.mp4"
    video.write_bytes(b"mp4")
    (import_dir / "import.json").write_text(
        '{"source_url": "https://www.rednote.com/explore/abc123"}',
        encoding="utf-8",
    )

    assert service.cleanup_video_cache(video) is True
    assert not import_dir.exists()


def test_service_does_not_clean_non_import_video(tmp_path: Path) -> None:
    service = VideoImportService(tmp_path / "cache" / "imports")
    local_dir = tmp_path / "Videos"
    local_dir.mkdir()
    video = local_dir / "video.mp4"
    video.write_bytes(b"mp4")

    assert service.cleanup_video_cache(video) is False
    assert video.exists()
