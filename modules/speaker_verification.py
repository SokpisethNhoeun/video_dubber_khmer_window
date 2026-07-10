from __future__ import annotations
import torch
from pathlib import Path
from typing import Union


def _torchaudio():
    import torchaudio
    return torchaudio

_classifier = None

def get_verification_classifier():
    global _classifier
    if _classifier is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SpeechBrain ECAPA-TDNN speaker verification model: {e}")
    return _classifier

def get_segment_embedding(audio_path: Union[str, Path], start: float, duration: float) -> torch.Tensor:
    """Extract speaker embedding for a specific segment of an audio file."""
    classifier = get_verification_classifier()
    
    # Read audio metadata to get sample rate
    info = _torchaudio().info(str(audio_path))
    fs = info.sample_rate
    
    frame_offset = int(start * fs)
    num_frames = int(duration * fs)
    
    # Guard against empty/invalid bounds
    if num_frames <= 0 or frame_offset < 0:
        raise ValueError(f"Invalid bounds: offset={frame_offset}, frames={num_frames}")
        
    signal, load_fs = _torchaudio().load(
        str(audio_path),
        frame_offset=frame_offset,
        num_frames=num_frames
    )
    
    # Convert to mono if stereo
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
        
    # Resample to 16000 Hz (required by SpeechBrain)
    if load_fs != 16000:
        resampler = _torchaudio().transforms.Resample(orig_freq=load_fs, new_freq=16000)
        signal = resampler(signal)
        
    # Move tensor to the classifier's device
    signal = signal.to(classifier.device)
    
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        
    return embeddings.squeeze().cpu()

def get_file_embedding(audio_path: Union[str, Path]) -> torch.Tensor:
    """Extract speaker embedding for the entire audio file."""
    classifier = get_verification_classifier()
    
    signal, fs = _torchaudio().load(str(audio_path))
    
    # Convert to mono if stereo
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
        
    # Resample to 16000 Hz
    if fs != 16000:
        resampler = _torchaudio().transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
        
    signal = signal.to(classifier.device)
    
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        
    return embeddings.squeeze().cpu()

def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Calculate the cosine similarity between two speaker embeddings."""
    sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    return float(sim.item())


# Similarity below this threshold means the clone doesn't sound like the
# reference; we fall back to base TTS for that speaker.
CLONE_SIMILARITY_ACCEPT = 0.55
CLONE_SIMILARITY_STRONG = 0.70
CLONE_MIN_SAMPLE_SECONDS = 1.2
CLONE_MAX_SAMPLES_PER_SPEAKER = 3


def verify_cloned_segments(
    speaker_id: str,
    reference_path: Union[str, Path],
    cloned_paths: list[Path],
) -> tuple[float, int]:
    """Compare a set of cloned audio outputs against the speaker reference.

    Picks the longest ``CLONE_MAX_SAMPLES_PER_SPEAKER`` cloned files at least
    ``CLONE_MIN_SAMPLE_SECONDS`` long, embeds each, and returns the mean cosine
    similarity plus the number of samples actually used.

    Returns (0.0, 0) if there is nothing usable to compare — caller should treat
    that as "verification skipped" rather than "failed".
    """
    usable: list[tuple[Path, float]] = []
    for path in cloned_paths:
        if not path.exists():
            continue
        try:
            info = _torchaudio().info(str(path))
            duration = info.num_frames / float(info.sample_rate or 1)
        except Exception:
            continue
        if duration >= CLONE_MIN_SAMPLE_SECONDS:
            usable.append((path, duration))

    if not usable:
        return 0.0, 0

    usable.sort(key=lambda item: item[1], reverse=True)
    samples = [path for path, _ in usable[:CLONE_MAX_SAMPLES_PER_SPEAKER]]

    reference_emb = get_file_embedding(reference_path)
    scores: list[float] = []
    for sample_path in samples:
        try:
            sample_emb = get_file_embedding(sample_path)
            scores.append(compute_similarity(reference_emb, sample_emb))
        except Exception:
            continue

    if not scores:
        return 0.0, 0
    mean_sim = sum(scores) / len(scores)
    return float(mean_sim), len(scores)
