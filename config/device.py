from __future__ import annotations


def resolve_compute_device(requested: str = "auto") -> tuple[str, str]:
    value = (requested or "auto").strip().lower()
    wants_gpu = value == "auto" or "cuda" in value or "gpu" in value
    if wants_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", f"GPU enabled: {torch.cuda.get_device_name(0)}"
            if torch.backends.mps.is_available():
                return "mps", "Apple GPU (MPS) enabled"
        except Exception:
            pass
    if value == "cpu":
        return "cpu", "CPU selected"
    return "cpu", "GPU is unavailable; using the compatible CPU fallback"
