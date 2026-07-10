from __future__ import annotations

import gc
import json
from pathlib import Path
from threading import Event
from typing import Callable
import numpy as np
import soundfile as sf

from core.context import CancellationError, Segment
from modules.diarizer import SpeakerTurn

ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]

DEFAULT_GENDER = "female"
MIN_GENDER_CLIP_SECONDS = 0.30
MIN_SPEAKER_GENDER_SECONDS = 0.80
MAX_SPEAKER_GENDER_SECONDS = 20.0
SPEECHBRAIN_GENDER_REPO = "moorlee/gender-voice-classifier-ecapa"


class SpeechBrainGenderUnavailableError(RuntimeError):
    """Raised when the optional SpeechBrain gender stack cannot be used."""


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _normalise_prediction_output(output) -> list[dict]:
    if isinstance(output, list) and output and isinstance(output[0], list):
        return output[0]
    if isinstance(output, list):
        return output
    return []


def _best_gender_label(output) -> str:
    prediction = _normalise_prediction_output(output)
    if not prediction:
        return DEFAULT_GENDER
    label = str(prediction[0].get("label", DEFAULT_GENDER)).lower()
    if "male" in label and "female" not in label:
        return "male"
    if "female" in label:
        return "female"
    return label or DEFAULT_GENDER


def _read_clip(
    audio_file: sf.SoundFile,
    start: float,
    end: float,
    samplerate: int,
    total_frames: int,
) -> np.ndarray | None:
    start_frame = max(0, int(start * samplerate))
    stop_frame = min(total_frames, int(end * samplerate))
    if stop_frame <= start_frame:
        return None

    audio_file.seek(start_frame)
    data = audio_file.read(stop_frame - start_frame)
    if data.size == 0:
        return None
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def _build_combined_clip(
    audio_wav: Path,
    intervals: list[tuple[float, float]],
) -> tuple[np.ndarray, int] | None:
    if not intervals:
        return None

    with sf.SoundFile(str(audio_wav)) as f:
        samplerate = f.samplerate
        total_frames = len(f)
        min_frames = int(MIN_GENDER_CLIP_SECONDS * samplerate)
        max_frames = int(MAX_SPEAKER_GENDER_SECONDS * samplerate)
        chunks: list[np.ndarray] = []
        used_frames = 0

        # Prefer the longest clean sections for the speaker-level embedding.
        sorted_intervals = sorted(intervals, key=lambda item: item[1] - item[0], reverse=True)
        for start, end in sorted_intervals:
            clip = _read_clip(f, start, end, samplerate, total_frames)
            if clip is None or len(clip) < min_frames:
                continue
            remaining = max_frames - used_frames
            if remaining <= 0:
                break
            if len(clip) > remaining:
                clip = clip[:remaining]
            chunks.append(clip)
            used_frames += len(clip)

        if used_frames < int(MIN_SPEAKER_GENDER_SECONDS * samplerate):
            return None
        return np.concatenate(chunks), samplerate


def _load_speechbrain_gender_model(device: str, log_cb: LogCallback | None):
    try:
        import joblib
        from huggingface_hub import hf_hub_download
        from modules.speaker_verification import get_verification_classifier
    except ImportError as exc:
        raise SpeechBrainGenderUnavailableError(
            "SpeechBrain gender detection needs speechbrain, scikit-learn, joblib, and huggingface-hub."
        ) from exc

    try:
        if log_cb:
            log_cb(
                "Loading SpeechBrain ECAPA gender classifier "
                f"({SPEECHBRAIN_GENDER_REPO})"
            )
        encoder = get_verification_classifier()
        scaler = joblib.load(hf_hub_download(SPEECHBRAIN_GENDER_REPO, "scaler.pkl"))
        classifier = joblib.load(hf_hub_download(SPEECHBRAIN_GENDER_REPO, "logreg.pkl"))
        thresholds_path = hf_hub_download(SPEECHBRAIN_GENDER_REPO, "thresholds.json")
        with open(thresholds_path, "r", encoding="utf-8") as file:
            thresholds = json.load(file)
        threshold = float(thresholds.get("female_optimized", thresholds.get("default", 0.5)))
    except Exception as exc:
        raise SpeechBrainGenderUnavailableError(
            f"SpeechBrain gender classifier could not be loaded: {exc}"
        ) from exc

    return {
        "encoder": encoder,
        "scaler": scaler,
        "classifier": classifier,
        "threshold": threshold,
    }


def _predict_speechbrain_gender(model, clip: np.ndarray, samplerate: int) -> tuple[str, float]:
    import torch

    signal = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)
    if samplerate != 16000:
        import torchaudio

        signal = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)(signal)

    encoder = model["encoder"]
    signal = signal.to(encoder.device)
    with torch.no_grad():
        embedding = encoder.encode_batch(signal).squeeze().detach().cpu().numpy()

    scaled = model["scaler"].transform(embedding.reshape(1, -1))
    female_probability = float(model["classifier"].predict_proba(scaled)[0, 1])
    gender = "female" if female_probability >= model["threshold"] else "male"
    return gender, female_probability


def _speaker_intervals_from_turns(speaker_turns: list[SpeakerTurn]) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    for turn in speaker_turns:
        if turn.end <= turn.start:
            continue
        intervals.setdefault(turn.speaker_id, []).append((turn.start, turn.end))
    return intervals


def _speaker_intervals_from_segments(segments: list[Segment]) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        if not segment.speaker_id:
            continue
        intervals.setdefault(segment.speaker_id, []).append((segment.start, segment.end))
    return intervals


def _speaker_gender_defaults(segments: list[Segment], results: dict[int, str]) -> dict[str, str]:
    counts: dict[str, dict[str, int]] = {}
    for segment in segments:
        if not segment.speaker_id or segment.index not in results:
            continue
        gender = results[segment.index]
        by_gender = counts.setdefault(segment.speaker_id, {})
        by_gender[gender] = by_gender.get(gender, 0) + 1
    return {
        speaker_id: max(by_gender.items(), key=lambda item: item[1])[0]
        for speaker_id, by_gender in counts.items()
        if by_gender
    }


def log_gender_emotion_summary(
    segments: list[Segment],
    genders: dict[int, str],
    emotion_analyses: dict | None,
    log_cb: LogCallback | None,
) -> None:
    """Log how detected gender pairs with detected emotion for TTS fallback."""
    if not log_cb or not genders:
        return

    if not emotion_analyses:
        female = sum(1 for value in genders.values() if value == "female")
        male = sum(1 for value in genders.values() if value == "male")
        log_cb(f"  Gender summary: female={female}, male={male}")
        return

    combo_counts: dict[str, int] = {}
    neutral = 0
    for segment in segments:
        gender = genders.get(segment.index)
        if not gender:
            continue
        analysis = emotion_analyses.get(segment.index)
        if analysis is None or getattr(analysis, "is_neutral_fallback", True):
            neutral += 1
            label = "neutral"
        else:
            label = getattr(analysis, "label", "neutral")
        key = f"{gender}+{label}"
        combo_counts[key] = combo_counts.get(key, 0) + 1

    if combo_counts:
        parts = ", ".join(f"{name}={count}" for name, count in sorted(combo_counts.items()))
        log_cb(f"  Gender + emotion for TTS fallback: {parts}")
    if neutral:
        log_cb(f"  {neutral} segment(s) will use neutral delivery on the detected gender voice")


def _classify_genders_with_speechbrain(
    audio_wav: Path,
    segments: list[Segment],
    device: str,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    speaker_turns: list[SpeakerTurn] | None,
) -> dict[int, str]:
    if not audio_wav.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_wav}")
    if not segments:
        return {}

    speaker_intervals = _speaker_intervals_from_turns(speaker_turns or [])
    if not speaker_intervals:
        speaker_intervals = _speaker_intervals_from_segments(segments)

    model = _load_speechbrain_gender_model(device, log_cb)
    results: dict[int, str] = {}
    speaker_genders: dict[str, str] = {}

    total_work = len(speaker_intervals) if speaker_intervals else len(segments)
    completed = 0

    if speaker_intervals:
        if log_cb:
            source = "pyannote speaker turns" if speaker_turns else "assigned speaker segments"
            log_cb(f"Running SpeechBrain gender detection for {len(speaker_intervals)} speaker(s) from {source}")

        for speaker_id, intervals in sorted(speaker_intervals.items()):
            if cancel_event.is_set():
                raise CancellationError("Processing cancelled by user")

            combined = _build_combined_clip(audio_wav, intervals)
            if combined is None:
                speaker_genders[speaker_id] = DEFAULT_GENDER
                if log_cb:
                    log_cb(f"  {speaker_id}: not enough usable speech for gender detection; using {DEFAULT_GENDER}")
            else:
                clip, samplerate = combined
                detected = True
                try:
                    gender, female_probability = _predict_speechbrain_gender(model, clip, samplerate)
                except Exception as exc:
                    detected = False
                    gender, female_probability = DEFAULT_GENDER, 1.0
                    if log_cb:
                        log_cb(f"  {speaker_id}: SpeechBrain gender detection failed ({exc}); using {gender}")
                speaker_genders[speaker_id] = gender
                if log_cb and detected:
                    log_cb(
                        f"  {speaker_id}: detected {gender} "
                        f"(female probability {female_probability:.2f}, {len(clip) / samplerate:.1f}s speech)"
                    )

            completed += 1
            if progress_cb:
                progress_cb(int((completed / max(1, total_work)) * 100))

        for segment in segments:
            if segment.speaker_id and segment.speaker_id in speaker_genders:
                results[segment.index] = speaker_genders[segment.speaker_id]
            else:
                results[segment.index] = DEFAULT_GENDER
                if log_cb:
                    log_cb(
                        f"  Segment {segment.index + 1}/{len(segments)} has no diarized speaker; "
                        f"using {DEFAULT_GENDER}"
                    )

        if progress_cb:
            progress_cb(100)
        return results

    if log_cb:
        log_cb("No diarized speakers available; running SpeechBrain gender detection per segment")

    for segment in segments:
        if cancel_event.is_set():
            raise CancellationError("Processing cancelled by user")
        combined = _build_combined_clip(audio_wav, [(segment.start, segment.end)])
        if combined is None:
            results[segment.index] = DEFAULT_GENDER
            if log_cb:
                log_cb(
                    f"  Segment {segment.index + 1}/{len(segments)} "
                    f"[{segment.start:.2f}s -> {segment.end:.2f}s]: too short for gender detection; "
                    f"using {DEFAULT_GENDER}"
                )
        else:
            clip, samplerate = combined
            detected = True
            try:
                gender, female_probability = _predict_speechbrain_gender(model, clip, samplerate)
            except Exception as exc:
                detected = False
                gender, female_probability = DEFAULT_GENDER, 1.0
                if log_cb:
                    log_cb(
                        f"  Segment {segment.index + 1}/{len(segments)} "
                        f"[{segment.start:.2f}s -> {segment.end:.2f}s]: "
                        f"SpeechBrain gender detection failed ({exc}); using {gender}"
                    )
            results[segment.index] = gender
            if log_cb and detected:
                log_cb(
                    f"  Segment {segment.index + 1}/{len(segments)} "
                    f"[{segment.start:.2f}s -> {segment.end:.2f}s]: detected {gender} "
                    f"(female probability {female_probability:.2f})"
                )
        completed += 1
        if progress_cb:
            progress_cb(int((completed / max(1, total_work)) * 100))

    if progress_cb:
        progress_cb(100)
    return results


def _classify_genders_transformer(
    audio_wav: Path,
    segments: list[Segment],
    device: str,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
) -> dict[int, str]:
    """Classifies the gender of the speaker for each audio segment.

    Returns a dict mapping segment index to 'female' or 'male'.
    """
    if not audio_wav.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_wav}")
    if not segments:
        return {}

    results: dict[int, str] = {}
    classifier = None
    try:
        inputs = []
        valid_segments: list[Segment] = []
        skipped_short: list[Segment] = []

        # Open source wav file to read segments
        with sf.SoundFile(str(audio_wav)) as f:
            samplerate = f.samplerate
            total_frames = len(f)
            total_segments = len(segments)
            min_frames = int(MIN_GENDER_CLIP_SECONDS * samplerate)

            for segment in segments:
                if cancel_event.is_set():
                    raise CancellationError("Processing cancelled by user")

                # Read segment chunk
                start_frame = int(segment.start * samplerate)
                stop_frame = int(segment.end * samplerate)

                # Bound to file length
                if start_frame >= total_frames:
                    results[segment.index] = DEFAULT_GENDER
                    continue
                if stop_frame > total_frames:
                    stop_frame = total_frames

                if stop_frame <= start_frame:
                    results[segment.index] = DEFAULT_GENDER
                    continue

                frame_count = stop_frame - start_frame
                if frame_count < min_frames:
                    skipped_short.append(segment)
                    continue

                f.seek(start_frame)
                data = f.read(frame_count)

                # Ensure data is single-channel float32
                if len(data.shape) > 1:
                    data = data[:, 0]  # Take first channel
                data = data.astype(np.float32)

                inputs.append({"raw": data, "sampling_rate": samplerate})
                valid_segments.append(segment)

        # Batch predict to maximize GPU efficiency and silence pipeline sequential warnings
        if inputs:
            from transformers import pipeline
            import torch

            selected_device = 0 if device == "cuda" and torch.cuda.is_available() else -1

            if log_cb:
                log_cb(f"Loading norwoodsystems/norwood-maleVSfemale gender classifier on device={selected_device}")

            classifier = pipeline(
                "audio-classification",
                model="norwoodsystems/norwood-maleVSfemale",
                device=selected_device,
            )

            if log_cb:
                log_cb(f"Running gender classification batch prediction on {len(inputs)} segments...")

            try:
                predictions = classifier(inputs, batch_size=16)
                for idx, (segment, out) in enumerate(zip(valid_segments, predictions)):
                    best_label = _best_gender_label(out)
                    results[segment.index] = best_label
                    if log_cb:
                        log_cb(
                            f"  Segment {segment.index + 1}/{total_segments} "
                            f"[{segment.start:.2f}s -> {segment.end:.2f}s]: detected {best_label}"
                        )
                    if progress_cb:
                        progress_cb(int(((idx + 1) / len(valid_segments)) * 100))
            except Exception as exc:
                if log_cb:
                    log_cb(
                        "  Gender classification batch failed; retrying segments individually "
                        f"and skipping invalid clips: {exc}"
                    )
                for idx, (segment, item) in enumerate(zip(valid_segments, inputs)):
                    if cancel_event.is_set():
                        raise CancellationError("Processing cancelled by user")
                    try:
                        out = classifier([item], batch_size=1)
                        best_label = _best_gender_label(out)
                    except Exception as item_exc:
                        best_label = DEFAULT_GENDER
                        if log_cb:
                            log_cb(
                                f"  Segment {segment.index + 1}/{total_segments} "
                                f"[{segment.start:.2f}s -> {segment.end:.2f}s]: "
                                f"gender detection skipped ({item_exc}); using {best_label}"
                            )
                    results[segment.index] = best_label
                    if log_cb and best_label != DEFAULT_GENDER:
                        log_cb(
                            f"  Segment {segment.index + 1}/{total_segments} "
                            f"[{segment.start:.2f}s -> {segment.end:.2f}s]: detected {best_label}"
                        )
                    if progress_cb:
                        progress_cb(int(((idx + 1) / len(valid_segments)) * 100))

        speaker_defaults = _speaker_gender_defaults(segments, results)
        for segment in skipped_short:
            fallback = speaker_defaults.get(segment.speaker_id or "", DEFAULT_GENDER)
            results[segment.index] = fallback
            if log_cb:
                log_cb(
                    f"  Segment {segment.index + 1}/{total_segments} "
                    f"[{segment.start:.2f}s -> {segment.end:.2f}s]: too short for gender detection; "
                    f"using {fallback}"
                )

        if progress_cb:
            progress_cb(100)

    finally:
        if classifier is not None:
            del classifier
        _free_gpu_memory()

    return results


def classify_genders(
    audio_wav: Path,
    segments: list[Segment],
    device: str,
    progress_cb: ProgressCallback | None,
    log_cb: LogCallback | None,
    cancel_event: Event,
    speaker_turns: list[SpeakerTurn] | None = None,
) -> dict[int, str]:
    """Classifies speaker gender for each segment.

    The preferred path uses pyannote speaker turns/assignments to classify each
    detected speaker once with SpeechBrain ECAPA embeddings plus a binary gender
    head. If that optional stack is unavailable, the previous transformer
    segment classifier remains as a compatibility fallback.
    """
    try:
        return _classify_genders_with_speechbrain(
            audio_wav,
            segments,
            device,
            progress_cb,
            log_cb,
            cancel_event,
            speaker_turns,
        )
    except SpeechBrainGenderUnavailableError as exc:
        if log_cb:
            log_cb(f"SpeechBrain gender detection unavailable; using fallback classifier: {exc}")
        return _classify_genders_transformer(
            audio_wav,
            segments,
            device,
            progress_cb,
            log_cb,
            cancel_event,
        )
