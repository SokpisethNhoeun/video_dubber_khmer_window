# Khmer Video Dubber

PyQt6 desktop application for dubbing video into Khmer. It extracts source audio with ffmpeg, transcribes with faster-whisper, translates to Khmer with NLLB-200, synthesizes Khmer speech with edge-tts, optionally runs a built-in reference voice matching tool or external voice conversion command, aligns the generated speech to the original timeline, and muxes the new audio back into the video.

The app is optimized for an NVIDIA RTX 3060 6GB workflow by loading one model stage at a time and clearing GPU memory between stages.

## Important Runtime Note

`edge-tts` uses Microsoft Edge online voices. The rest of the main model pipeline can run from local Hugging Face cache after the one-time setup step, but TTS requires network access unless you replace the TTS module with a local Khmer voice.

## System Dependencies

- Kali Linux
- Python 3.10 or newer
- NVIDIA driver and CUDA-compatible PyTorch for GPU mode
- `ffmpeg` and `ffprobe`

Install ffmpeg:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Python Setup

```bash
cd video_dubber
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the wheel that matches your local CUDA driver from the official PyTorch instructions, then install the remaining requirements.

Speaker voice grouping uses the built-in per-segment male/female detector and does not require a
Hugging Face token.

TorchCodec is not required by this app. If it is installed separately and setup
checks report a TorchCodec error, remove it or install a version compatible with
your installed PyTorch version.

## One-Time Model Download

```bash
cd video_dubber
source .venv/bin/activate
python setup_models.py
```

By default this downloads:

- `Systran/faster-whisper-medium`
- `Systran/faster-whisper-small`
- `facebook/nllb-200-distilled-600M`

You can download another Whisper model:

```bash
python setup_models.py --whisper-model medium --whisper-model base
```

## Run

```bash
cd video_dubber
source .venv/bin/activate
python main.py
```

Keep intermediate files for inspection:

```bash
python main.py --keep-temp
```

## Free Windows Self-Signed Releases

A self-signed certificate is free, but each target laptop's administrator must explicitly trust
its public certificate and allow that publisher in Windows Application Control. Generate the
private PFX and public CER on a trusted Windows computer:

```powershell
$Password = Read-Host "PFX password" -AsSecureString
./packaging/windows/create-self-signed-certificate.ps1 -Password $Password
```

Create these GitHub Actions secrets from the generated files:

- `WINDOWS_SIGNING_PFX_BASE64`: contents of `WINDOWS_SIGNING_PFX_BASE64.txt`
- `WINDOWS_SIGNING_CERT_BASE64`: contents of `WINDOWS_SIGNING_CERT_BASE64.txt`
- `WINDOWS_SIGNING_PASSWORD`: the PFX password

The release artifact contains the installer and `KhmerVideoDubber-Publisher.cer`. Give only the
CER file to the target administrator. Never share or commit the PFX, its Base64 text, or password.
The administrator must verify the certificate thumbprint through a trusted channel before adding
it to the laptop's trusted publishers and App Control allow policy.

Click `Check Setup` in the app before a long job to verify FFmpeg, PyTorch/CUDA,
torchaudio, Demucs, Edge TTS, and the
selected voice clone command. The results are printed in the execution log.

The input picker supports selecting multiple videos at once. Batch jobs are processed one video at a time and each generated video is written to the selected output folder.

Quality options are enabled by default:

- `Auto clean audio` validates and prepares reference audio, trims silence, normalizes loudness, and shortens excessive synthesized pauses before natural-speed alignment.
- `Final audio mastering` normalizes and limits the final dubbed WAV before muxing, while keeping it aligned to the source video duration.
- `Use persistent cache` reuses TTS, cleaned references, and diarization results across runs from `video_dubber/cache/`.
- `Preset` can switch between `Fast Preview`, `Balanced`, and `Best Quality`. Fast Preview disables Demucs, mastering, cloning, and transcript review for quicker checks. Best Quality selects the largest Whisper model, enables cleanup/mastering/Demucs, and uses AI transcript review when configured.

## Transcript Review

After NLLB translation, the app can review the full transcript before TTS. The review step keeps every segment's original start/end timing unchanged, then writes improved Khmer text that TTS uses.

Review modes:

- `Local cleanup` is the default and works without an API key.
- `AI review if configured` calls an OpenAI-compatible chat completions endpoint when `TRANSCRIPT_REVIEW_API_KEY` or `OPENAI_API_KEY` is available; otherwise it falls back to local cleanup.
- `Off` uses the raw NLLB Khmer text.

Optional `.env` settings:

```bash
TRANSCRIPT_REVIEW_API_KEY=your_key_here
TRANSCRIPT_REVIEW_API_BASE_URL=https://api.openai.com/v1
TRANSCRIPT_REVIEW_MODEL=gpt-4o-mini
```

Khmer style options are `Simple/Easy`, `Natural`, `Formal`, and `Short Dub`. You can also select a glossary file with `source=target` or `source,target` entries.

When `Save review JSON` is enabled, each run writes:

```text
<output-video-stem>_transcript_review.json
```

Select that JSON in `Load review`, click `Edit Review JSON`, then edit enabled status and Khmer text in the table. `Merge Selected` combines selected rows into the first selected row and disables the following rows. `Preview Segment` synthesizes the selected Khmer line with the current female voice.

## Supported Source Languages

- Chinese
- Khmer
- English

Khmer source audio skips translation and goes directly to Khmer TTS.

## Reference Voice Clone Adapter

Voice cloning is optional. The app includes a local reference voice matching backend that can run without downloading another model. It matches pitch, loudness, and spectral tone from the selected reference voice; it is useful for quick local previews, but it is not the same quality as a large neural zero-shot voice clone model.

The default workflow only needs:

- a reference voice MP3, WAV, or a microphone recording from the app

You can save reusable voices from the app:

1. Select or record a reference MP3/WAV in `Reference audio`.
2. Enter a name in `Voice name`.
3. Select `Female` or `Male` in `Voice type`.
4. Click `Generate Voice`.
5. Pick the generated voice from `Female Voice` or `Male Voice`, then click `Test`.

To add many voices quickly, choose `Female` or `Male` in `Voice type`, click `Import Voices`, and select multiple MP3/WAV files. The app creates one generated voice profile per file using the filename as the voice name.

Generated voices are stored locally under `voice_profiles/`. This creates a cleaned named reference voice profile and adds it to the matching voice dropdown; the external clone command still performs the actual voice conversion during testing and dubbing. Use `Test Saved` to preview the selected saved voice or `Delete Voice` to remove its copied source and cleaned reference files.

The command field supports these tokens:

- `{input}` synthesized segment audio
- `{output}` expected cloned output wav path
- `{reference}` selected reference audio path, such as an MP3 voice sample
- `{model}` optional advanced `.pth` path, only required if your command uses `{model}`
- `{index}` optional advanced `.index` path, only required if your command uses `{index}`

Default built-in command:

```bash
python -m modules.reference_voice_clone --input "{input}" --output "{output}" --reference "{reference}"
```

When the app finds `.venv/bin/python`, it uses that interpreter in the Command field so the clone tool can load the installed project dependencies.

The Reference voice clone box also has preset buttons:

- `Use OpenVoice` selects the local OpenVoice adapter and enables cloning.
- `Use Local` selects the built-in DSP reference matcher and enables cloning.
- `Use RVC Model` selects an advanced `.pth` / `.index` command template and enables cloning.
- `Use Production` selects the ElevenLabs speech-to-speech backend and enables cloning.

The status line under `Command` shows which files the current template requires. Selecting a `.pth` model or `.index` file automatically switches to the advanced RVC template when the current command does not already use that token.

You can replace the command with a stronger neural voice conversion tool later. The project does not include `infer.py`; only use an `infer.py` command if you install a separate tool that provides it.

### Local OpenVoice Clone

For local neural zero-shot voice conversion, install OpenVoice in a separate Python 3.10/3.11 environment. Do not install it into the main app environment if your app environment is newer, because many voice conversion packages are pinned to older Python and PyTorch stacks.

Example setup:

```bash
cd ~/Coding
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Download the OpenVoice checkpoints using the upstream OpenVoice instructions, then point this app at that environment from `video_dubber/.env`:

```bash
VOICE_CLONE_BACKEND=local_openvoice
OPENVOICE_PYTHON=/absolute/path/to/OpenVoice/.venv/bin/python
OPENVOICE_CHECKPOINT_DIR=/absolute/path/to/OpenVoice/checkpoints_v2
OPENVOICE_DEVICE=auto
```

When `VOICE_CLONE_BACKEND=local_openvoice` is set before launch, the app fills the Command field with:

```bash
python -m modules.openvoice_voice_clone --input "{input}" --output "{output}" --reference "{reference}"
```

The adapter runs in the main app process only long enough to validate inputs, then invokes `OPENVOICE_PYTHON` so OpenVoice imports from its own environment. It expects a converter checkpoint containing `config.json` and `checkpoint.pth` either directly under `OPENVOICE_CHECKPOINT_DIR` or under a `converter/` subdirectory.

### Production ElevenLabs Clone

For stronger production-quality cloning, set an ElevenLabs API key before launching the app:

```bash
export ELEVENLABS_API_KEY=your_api_key_here
python main.py
```

Or put it in the project `.env` file:

```bash
ELEVENLABS_API_KEY=your_api_key_here
```

Then click `Use Production` in the Reference voice clone box. The command changes to:

```bash
python -m modules.elevenlabs_voice_clone --input "{input}" --output "{output}" --reference "{reference}"
```

This backend creates or reuses an ElevenLabs instant voice clone for each reference audio file, caches the returned `voice_id` in `cache/elevenlabs_voice_cache.json`, runs ElevenLabs speech-to-speech for each generated segment, and writes the expected WAV output for the pipeline. Click `Use Local` to switch back to the offline built-in matcher.

Advanced RVC model command template:

```bash
python infer.py --input "{input}" --output "{output}" --model "{model}" --index "{index}"
```

For the default reference workflow, select an MP3/WAV reference or click Record. You do not need a `.pth` model or `.index` file unless you click `Use RVC Model` or edit the command template to use `{model}` or `{index}`. Add `{reference}` to the command only if your external tool supports a reference audio input. The command must create the exact `{output}` file for every segment.

Use the Clone segments selector to choose whether the reference voice clone runs for all segments, female segments only, or male segments only. In Auto-detect speaker voice mode, the app uses detected segment gender; in single-voice mode, all segments use the selected voice mode.

Reference audio is validated before cloning. The app warns about missing files, unsupported formats, short clips, silence, clipping, DC offset, and likely noisy or over-compressed audio. A clean reference should be at least 10 seconds of speech.

## Speaker Voice Grouping

The standard workflow classifies each transcription segment as male or female and selects the
configured Khmer voice. It does not identify persistent people across the whole video and does not
require a Hugging Face token. Saved female and male voice profiles can still be selected manually.

## Pipeline

1. Extract audio to mono 16 kHz WAV
2. Transcribe with faster-whisper using VAD and timestamps
3. Detect a male/female voice category for each transcription segment
4. Batch translate to Khmer with NLLB-200
5. Review Khmer transcript with story context, optional glossary, saved edits, or optional AI
6. Batch synthesize reviewed Khmer TTS with edge-tts
7. Optionally run the built-in reference matcher, external RVC, or per-person reference voice conversion
8. Build a timeline audio map with natural-speed alignment: short speech leaves silence, long speech speeds up only to `1.25x`
9. Optionally mix dubbed voice with the separated background track using the selected voice/music volumes
10. Mux final audio into the original video using video stream copy

## Output

Dubbed videos are written to the selected output folder as:

```text
<original-name>_khmer_dubbed.mp4
```

Optional exports are enabled by default:

- `<output-video-stem>_khmer_dubbed.wav`
- `<output-video-stem>_original_transcript.txt`
- `<output-video-stem>_raw_khmer.txt`
- `<output-video-stem>_improved_khmer.txt`
- `<output-video-stem>_khmer_subtitles.srt`
- `<output-video-stem>_quality_report.json`
- `<output-video-stem>_quality_report.txt`

The report summarizes segment count, speaker count, missing/bad references, long segments after the `1.25x` cap, voice clone failures, cache hits, and the final output path.
# Khmer-video-dubber
