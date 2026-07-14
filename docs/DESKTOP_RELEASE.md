# Desktop release and subscription checklist

## Current Windows release implementation

The repository now contains:

- frozen/source path handling in `config/runtime.py`
- bundled `ffmpeg.exe` and `ffprobe.exe` discovery
- packaged public configuration through `release.json`; packaged builds ignore `.env`
- a PyInstaller one-folder spec
- an Inno Setup installer definition
- resumable Whisper, NLLB, and Qwen checkpoint downloads with disk-space and size checks
- a hidden `--smoke-test` command for the frozen application
- a Windows GitHub Actions build, smoke-test, artifact, and optional signing workflow

The Windows build must still be executed on Windows. Linux cannot validate the
Windows bootloader, DLL loading, installer behavior, GPU drivers, or code
signature.

### Build locally on Windows

1. Install Python 3.11 x64 and Inno Setup.
2. Create and activate a clean build virtual environment.
3. Install `requirements.txt` and `pyinstaller`.
4. Put trusted Windows x64 FFmpeg binaries in
   `packaging/windows/vendor/bin/` and record their SHA-256 hashes in
   `SHA256SUMS.txt`.
5. Run:

```powershell
packaging\windows\build.ps1 `
  -LicenseServerUrl "https://your-fastapi-cloud.example.com" `
  -Version "1.0.0"
```

6. Test the frozen app before distribution:

```powershell
dist\KhmerVideoDubber\KhmerVideoDubber.exe --smoke-test
```

7. The customer installer is written to:

```text
packaging/windows/release/KhmerVideoDubber-Setup-1.0.0.exe
```

### Voice-runtime boundary

Whisper and NLLB run in the packaged application. Qwen3 checkpoints can be
downloaded by the Downloads window. The current Qwen3, CosyVoice, and OpenVoice
adapters execute separate Python interpreters configured through
`QWEN3_TTS_PYTHON`, `COSYVOICE_PYTHON`, and `OPENVOICE_PYTHON`.

Those interpreters cannot be produced as reliable portable Windows virtual
environments from Linux. OpenVoice also pins an old NumPy stack, while
CosyVoice has its own Torch/ONNX dependency set. They require separate Windows
helper executables or a tested Windows runtime installer before those three
backends can be advertised as available in the customer build. Do not mark a
checkpoint-only installation as a ready voice engine.

The app is a desktop product, not a browser-hosted video processor. Publish a
separate installer for each operating system; do not build one binary and
expect it to run everywhere.

## Runtime behavior

- `Automatic` compute selects CUDA when PyTorch can use it and otherwise runs
  on CPU.
- Voice choices are female TTS, male TTS, and automatic male/female TTS with
  emotion. Voice cloning is disabled in desktop-generated settings.
- A customer's Gemini key is saved in their user configuration directory, not
  in the installation directory or repository.
- When `LICENSE_SERVER_URL` is configured, every processing start validates the
  subscription and device binding. Source checkouts without it are development
  mode so contributors are not locked out.

## Builds

Use PyInstaller (or Briefcase) on each target OS in CI:

- Windows x64: Windows runner and signed `.exe`/MSIX installer
- macOS Intel and Apple Silicon: macOS runners, signing, and notarization
- Linux x64: Linux runner and AppImage/deb packaging

Bundle FFmpeg according to its license, and verify model/download storage on a
clean machine. GPU acceleration requires a supported NVIDIA driver and a
matching CUDA-enabled PyTorch build. CPU fallback should remain in every build.

Large ML models should normally download on first use with a progress screen;
including every model makes the installer extremely large. Test low-memory CPU
devices with smaller Whisper presets.

## Commercial launch

1. Deploy `license_server` behind HTTPS with a production database.
2. Configure SMTP and verify OTP and license delivery using a production inbox.
3. Verify each payment outside the application, then confirm it from the admin
   dashboard. Confirmation creates one license and emails its key atomically.
4. Test renewal extension, revocation, and controlled device reset procedures.
6. Publish terms, refund policy, privacy policy, and support contact details.

The current checkout records manual payments; it does not verify Bakong or bank
transactions automatically. Do not rely on screenshots alone. Integrate a
signed, idempotent provider webhook before enabling automatic confirmation.
