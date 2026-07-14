from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if len(sys.argv) > 2 and sys.argv[1] == "-m":
    import runpy
    module_name = sys.argv[2]
    sys.argv = [sys.argv[0]] + sys.argv[3:]
    try:
        runpy.run_module(module_name, run_name="__main__")
        sys.exit(0)
    except Exception as e:
        print(f"Error running module {module_name}: {e}", file=sys.stderr)
        sys.exit(1)

# Suppress noisy Qt debug messages (Wayland text-input events, FFmpeg version banner)
os.environ.setdefault(
    "QT_LOGGING_RULES",
    "qt.multimedia.ffmpeg=false;qt.qpa.wayland.textinput=false",
)

from config.env import load_project_env
from config.runtime import configure_bundled_tools, working_root
from gui.app_window import run_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline/local Khmer video dubbing desktop app")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files under video_dubber/temp")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = working_root()
    configure_bundled_tools()
    load_project_env(project_root)
    if args.smoke_test:
        from config.paths import app_data_dir
        from config.runtime import executable_for

        data_dir = app_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        if not executable_for("ffmpeg") or not executable_for("ffprobe"):
            print("SMOKE_TEST_FAILED: bundled FFmpeg tools are unavailable")
            return 2
        if not os.getenv("LICENSE_SERVER_URL", "").startswith("https://"):
            print("SMOKE_TEST_FAILED: production LICENSE_SERVER_URL is unavailable")
            return 3
        print(f"SMOKE_TEST_OK: data={data_dir}")
        return 0
    return run_app(project_root=project_root, keep_temp=args.keep_temp)


if __name__ == "__main__":
    sys.exit(main())
