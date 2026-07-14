from pathlib import Path

from config import runtime
from gui.workers import _friendly_processing_error


ROOT = Path(__file__).resolve().parents[1]


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


def test_windows_build_declares_and_collects_demucs() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    spec = (ROOT / "packaging/windows/KhmerVideoDubber.spec").read_text(encoding="utf-8")
    main = (ROOT / "main.py").read_text(encoding="utf-8")

    assert "demucs>=4,<5" in requirements
    assert '"demucs"' in spec
    assert "import demucs.separate" in main
    assert "import pyannote.audio" not in main
    assert '"pyannote.audio"' not in spec
    assert "import av" in main
    assert "import ctranslate2" in main
    assert "import faster_whisper" in main


def test_windows_release_signs_and_verifies_every_native_file() -> None:
    workflow = (ROOT / ".github/workflows/windows-release.yml").read_text(encoding="utf-8")

    assert '$_ .Extension' not in workflow
    assert '$_ .Extension -in' not in workflow
    assert '$_.Extension -in ".exe", ".dll", ".pyd"' in workflow
    assert "Get-AuthenticodeSignature" in workflow
    assert "WINDOWS_SIGNING_CERT_BASE64" in workflow
    assert "does not match WINDOWS_SIGNING_PFX_BASE64" in workflow
    assert "KhmerVideoDubber-Publisher.cer" in workflow


def test_self_signed_certificate_helper_keeps_private_output_ignored() -> None:
    helper = ROOT / "packaging/windows/create-self-signed-certificate.ps1"
    script = helper.read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "New-SelfSignedCertificate" in script
    assert "CodeSigningCert" in script
    assert "WINDOWS_SIGNING_PFX_BASE64.txt" in script
    assert "WINDOWS_SIGNING_CERT_BASE64.txt" in script
    assert "packaging/windows/signing-output/" in gitignore


def test_windows_subprocesses_are_hidden(monkeypatch) -> None:
    monkeypatch.setattr(runtime.os, "name", "nt")
    monkeypatch.setattr(runtime.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    assert runtime.windows_creation_flags() == 0x08000000


def test_application_control_error_has_actionable_message() -> None:
    message = _friendly_processing_error(
        RuntimeError("DLL load failed: An Application Control policy has blocked this file")
    )

    assert "officially signed" in message
    assert "Windows administrator" in message
