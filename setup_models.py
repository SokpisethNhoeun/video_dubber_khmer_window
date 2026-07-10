from __future__ import annotations

import argparse
import os
import time
from typing import List

from config.models import DEFAULT_WHISPER_MODEL, FASTER_WHISPER_MODEL_PREFIX, NLLB_MODEL_ID
from config.paths import nllb_cache_dir, whisper_cache_dir


# =========================
# GLOBAL PERFORMANCE FIXES
# =========================

# Enable fast HuggingFace downloader (VERY IMPORTANT)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Avoid IPv6 stalls on Kali/Linux
os.environ["HF_HUB_DISABLE_IPV6"] = "1"


# =========================
# DOWNLOAD WHISPER MODELS
# =========================

def download_whisper_models(models: List[str]) -> None:
    from huggingface_hub import snapshot_download

    cache_dir = str(whisper_cache_dir())

    for model_name in models:
        repo_id = f"{FASTER_WHISPER_MODEL_PREFIX}-{model_name}"
        print(f"\n[WHISPER] Downloading: {repo_id}")

        _safe_download(
            snapshot_download,
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=True,
            max_workers=8,
        )


# =========================
# DOWNLOAD NLLB MODEL
# =========================

def download_nllb() -> None:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"\n[NLLB] Downloading: {NLLB_MODEL_ID}")

    cache_dir = str(nllb_cache_dir())

    _safe_download(
        AutoTokenizer.from_pretrained,
        NLLB_MODEL_ID,
        cache_dir=cache_dir,
    )

    _safe_download(
        AutoModelForSeq2SeqLM.from_pretrained,
        NLLB_MODEL_ID,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )


# =========================
# SAFE RETRY WRAPPER
# =========================

def _safe_download(func, *args, retries: int = 3, delay: int = 3, **kwargs):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"  -> Attempt {attempt}/{retries}")
            result = func(*args, **kwargs)
            print("  ✓ Success")
            return result

        except Exception as e:
            last_error = e
            print(f"  ✗ Failed: {e}")
            time.sleep(delay)

    raise RuntimeError(f"Download failed after {retries} attempts") from last_error


# =========================
# ARG PARSER
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model files for local cached use")

    parser.add_argument(
        "--whisper-model",
        action="append",
        default=[],
        help="Whisper model to download (can be used multiple times)",
    )

    parser.add_argument(
        "--skip-nllb",
        action="store_true",
        help="Skip NLLB download",
    )

    return parser.parse_args()


# =========================
# MAIN
# =========================

def main() -> int:
    args = parse_args()

    whisper_models = args.whisper_model or [DEFAULT_WHISPER_MODEL, "small"]

    print("\n==============================")
    print(" MODEL SETUP STARTING")
    print("==============================")

    print("\n[INFO] Whisper models:", whisper_models)

    download_whisper_models(whisper_models)

    if not args.skip_nllb:
        download_nllb()

    print("\n==============================")
    print(" MODEL SETUP COMPLETE")
    print("==============================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())