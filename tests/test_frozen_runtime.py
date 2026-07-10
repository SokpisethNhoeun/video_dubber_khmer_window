from pathlib import Path

from config import runtime


def test_source_working_root_is_project_root() -> None:
    assert runtime.working_root() == Path(runtime.__file__).resolve().parents[1]


def test_frozen_working_root_uses_user_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runtime, "is_frozen", lambda: True)
    monkeypatch.setattr(runtime, "user_data_dir", lambda *_args, **_kwargs: str(tmp_path / "data"))

    assert runtime.working_root() == tmp_path / "data"
    assert (tmp_path / "data").is_dir()


def test_bundled_tools_prepend_path(monkeypatch, tmp_path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg").write_bytes(b"ffmpeg")
    (bin_dir / "ffprobe").write_bytes(b"ffprobe")
    monkeypatch.setattr(runtime, "install_root", lambda: tmp_path)
    monkeypatch.setattr(runtime, "resource_root", lambda: tmp_path / "missing")
    monkeypatch.setenv("PATH", "/usr/bin")

    runtime.configure_bundled_tools()

    assert runtime.os.environ["PATH"].split(runtime.os.pathsep)[0] == str(bin_dir)
