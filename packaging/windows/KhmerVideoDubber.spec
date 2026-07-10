# Build from the video_dubber root:
#   pyinstaller --noconfirm --clean packaging/windows/KhmerVideoDubber.spec

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


ROOT = Path(SPECPATH).resolve().parents[1]
GENERATED = ROOT / "packaging" / "windows" / "generated"

datas = []
binaries = []
hiddenimports = []

for package in (
    "PyQt6",
    "qtawesome",
    "qrcode",
    "PIL",
    "faster_whisper",
    "ctranslate2",
    "transformers",
    "sentencepiece",
    "huggingface_hub",
    "torch",
    "torchaudio",
    "edge_tts",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += collect_submodules("modules")
hiddenimports += collect_submodules("gui")
hiddenimports += collect_submodules("core")

release_config = GENERATED / "release.json"
if not release_config.is_file():
    raise SystemExit("Missing packaging/windows/generated/release.json. Run build.ps1 first.")
datas.append((str(release_config), "."))

for binary_name in ("ffmpeg.exe", "ffprobe.exe"):
    binary = GENERATED / "bin" / binary_name
    if not binary.is_file():
        raise SystemExit(f"Missing {binary}. Run build.ps1 after placing FFmpeg in packaging/windows/vendor/bin.")
    binaries.append((str(binary), "bin"))

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    excludes=["pytest", "tkinter", "jupyter", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KhmerVideoDubber",
    console=False,
    disable_windowed_traceback=False,
    uac_admin=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="KhmerVideoDubber",
)
