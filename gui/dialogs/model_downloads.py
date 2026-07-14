from __future__ import annotations

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.models import NLLB_MODEL_ID
from config.paths import (
    is_whisper_model_downloaded,
    cosyvoice_cache_dir,
    nllb_cache_dir,
    qwen_cache_dir,
    repository_snapshot_exists,
)
from gui.workers import ModelDownloadWorker
from modules.model_downloader import HuggingFaceModelDownloadManager, ModelDownloadManager


MODEL_PRESETS = [
    ("tiny", "Whisper Tiny", "Ultra-fast (~75MB download, ~1GB RAM, CPU-friendly)"),
    ("base", "Whisper Base", "Fast, lower-memory (~140MB download, ~1.5GB RAM)"),
    ("small", "Whisper Small", "Recommended speed/memory balance (~460MB download, ~2GB RAM)"),
    ("medium", "Whisper Medium", "Recommended quality and speed (~1.5GB download, ~5GB RAM)"),
    ("large-v3", "Whisper Large v3", "Best transcription, largest size (~3.0GB download, ~8GB RAM, high-spec GPU)"),
]
QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
COSYVOICE_MODEL_ID = "FunAudioLLM/CosyVoice2-0.5B"


class ModelDownloadsDialog(QDialog):
    """Manage resumable customer model downloads after onboarding."""

    model_installed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Downloads")
        self.setMinimumWidth(760)
        self.resize(820, 520)
        self._rows: dict[str, dict[str, object]] = {}
        self._manager: HuggingFaceModelDownloadManager | None = None
        self._thread: QThread | None = None
        self._worker: ModelDownloadWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Models and voice engines")
        title.setObjectName("PageHeader")
        layout.addWidget(title)

        note = QLabel(
            "Downloads run in the background and can be paused or resumed. "
            "Models are stored in your Windows user data folder, not in Program Files."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        for model, label, description in MODEL_PRESETS:
            layout.addWidget(self._build_model_row(model, label, description))

        layout.addWidget(
            self._build_repository_row(
                "nllb",
                "NLLB Khmer Translation",
                "Required local translator (~1.2GB download, ~2.5GB RAM)",
                NLLB_MODEL_ID,
                nllb_cache_dir(),
            )
        )
        layout.addWidget(
            self._build_repository_row(
                "cosyvoice",
                "CosyVoice 2 Model",
                "Optional voice-conversion model (~1.0GB download, Nvidia GPU VRAM >= 6GB)",
                COSYVOICE_MODEL_ID,
                cosyvoice_cache_dir(),
            )
        )
        layout.addWidget(
            self._build_repository_row(
                "qwen3",
                "Qwen3-TTS 1.7B Model",
                "Optional high-quality voice cloning (~3.5GB download, Nvidia GPU VRAM >= 6GB)",
                QWEN_MODEL_ID,
                qwen_cache_dir(),
            )
        )

        runtime_card = QWidget()
        runtime_card.setObjectName("Card")
        runtime_layout = QVBoxLayout(runtime_card)
        runtime_title = QLabel("<b>Voice-cloning runtimes</b>")
        runtime_title.setTextFormat(Qt.TextFormat.RichText)
        runtime_layout.addWidget(runtime_title)
        runtime_note = QLabel(
            "Qwen3-TTS, CosyVoice, and OpenVoice need both model checkpoints and their own "
            "compatible Python runtime. They must be installed by the Windows installer; "
            "a checkpoint-only download would not make these features usable."
        )
        runtime_note.setWordWrap(True)
        runtime_layout.addWidget(runtime_note)
        layout.addWidget(runtime_card)

        layout.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

    def _build_model_row(self, model: str, label: str, description: str) -> QWidget:
        installed = is_whisper_model_downloaded(model)
        card = QWidget()
        card.setObjectName("Card")
        row = QHBoxLayout(card)
        text = QLabel(f"<b>{label}</b><br><span style='color:#8b93a7'>{description}</span>")
        text.setTextFormat(Qt.TextFormat.RichText)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(100 if installed else 0)
        progress.setMinimumWidth(190)
        detail = QLabel("Installed" if installed else "Waiting")
        button = QPushButton("Installed" if installed else "Download")
        button.setEnabled(not installed)
        button.clicked.connect(lambda _checked=False, name=model: self._model_action(name))
        cancel = QPushButton("Cancel")
        cancel.setVisible(False)
        cancel.clicked.connect(lambda _checked=False, name=model: self._cancel(name))
        row.addWidget(text, 1)
        row.addWidget(progress)
        row.addWidget(detail)
        row.addWidget(button)
        row.addWidget(cancel)
        self._rows[model] = {
            "progress": progress,
            "detail": detail,
            "button": button,
            "cancel": cancel,
            "state": "done" if installed else "waiting",
        }
        return card

    def _build_repository_row(
        self, key: str, label: str, description: str, repo_id: str, cache_dir
    ) -> QWidget:
        installed = repository_snapshot_exists(repo_id, cache_dir)
        card = QWidget()
        card.setObjectName("Card")
        row = QHBoxLayout(card)
        text = QLabel(f"<b>{label}</b><br><span style='color:#8b93a7'>{description}</span>")
        text.setTextFormat(Qt.TextFormat.RichText)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(100 if installed else 0)
        progress.setMinimumWidth(190)
        detail = QLabel("Installed" if installed else "Waiting")
        button = QPushButton("Installed" if installed else "Download")
        button.setEnabled(not installed)
        button.clicked.connect(lambda _checked=False, name=key: self._model_action(name))
        cancel = QPushButton("Cancel")
        cancel.setVisible(False)
        cancel.clicked.connect(lambda _checked=False, name=key: self._cancel(name))
        row.addWidget(text, 1)
        row.addWidget(progress)
        row.addWidget(detail)
        row.addWidget(button)
        row.addWidget(cancel)
        self._rows[key] = {
            "progress": progress,
            "detail": detail,
            "button": button,
            "cancel": cancel,
            "state": "done" if installed else "waiting",
            "repo_id": repo_id,
            "cache_dir": cache_dir,
        }
        return card

    def _model_action(self, model: str) -> None:
        selected = self._rows[model]
        if selected["state"] == "downloading" and self._manager:
            self._manager.pause()
            return
        if selected["state"] == "paused" and self._manager:
            self._manager.resume()
        else:
            if "repo_id" in selected:
                self._manager = HuggingFaceModelDownloadManager(
                    selected["repo_id"], selected["cache_dir"]
                )
            else:
                self._manager = ModelDownloadManager(model)

        for name, row in self._rows.items():
            row["button"].setEnabled(name == model or row["state"] == "done")

        self._thread = QThread(self)
        self._worker = ModelDownloadWorker(self._manager)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            lambda filename, done, total, speed, eta: self._progress(model, filename, done, total, speed, eta)
        )
        self._worker.status.connect(lambda state: self._state(model, state))
        self._worker.finished.connect(lambda _path: self._complete(model))
        self._worker.failed.connect(lambda message: self._failed(model, message))
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.status.connect(
            lambda state: self._thread.quit() if state in {"paused", "cancelled"} else None
        )
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _progress(self, model: str, _filename: str, done: int, total: int, speed: float, eta: object) -> None:
        row = self._rows[model]
        if total:
            row["progress"].setRange(0, 100)
            row["progress"].setValue(int(done * 100 / total))
        else:
            row["progress"].setRange(0, 0)
        eta_text = f" · {int(float(eta))}s left" if eta is not None else ""
        downloaded = f"{done / 1024 / 1024:.1f} MB · " if not total else ""
        row["detail"].setText(f"{downloaded}{speed / 1024 / 1024:.1f} MB/s{eta_text}")

    def _state(self, model: str, state: str) -> None:
        row = self._rows[model]
        row["state"] = state
        if state == "connecting":
            row["progress"].setRange(0, 0)
            row["detail"].setText("Connecting…")
            row["button"].setText("Connecting…")
            row["button"].setEnabled(False)
        elif state == "downloading":
            row["button"].setText("Pause")
            row["button"].setEnabled(True)
        elif state == "paused":
            row["button"].setText("Resume")
            row["button"].setEnabled(True)
        else:
            row["button"].setText("Download")
        row["cancel"].setVisible(state in {"connecting", "downloading", "paused"})

    def _cancel(self, model: str) -> None:
        if self._manager:
            self._manager.cancel()
        row = self._rows[model]
        row["state"] = "waiting"
        row["progress"].setRange(0, 100)
        row["progress"].setValue(0)
        row["detail"].setText("Cancelled")
        row["button"].setText("Download")
        row["button"].setEnabled(True)
        row["cancel"].setVisible(False)
        self.model_installed.emit(model)
        self._enable_other_rows()

    def _complete(self, model: str) -> None:
        row = self._rows[model]
        row["state"] = "done"
        row["progress"].setRange(0, 100)
        row["progress"].setValue(100)
        row["detail"].setText("Installed")
        row["button"].setText("Installed")
        row["button"].setEnabled(False)
        row["cancel"].setVisible(False)
        self._enable_other_rows()

    def _failed(self, model: str, message: str) -> None:
        row = self._rows[model]
        row["state"] = "failed"
        row["progress"].setRange(0, 100)
        row["detail"].setText(message[:80])
        row["button"].setText("Retry")
        row["button"].setEnabled(True)
        row["cancel"].setVisible(False)
        self._enable_other_rows()

    def _enable_other_rows(self) -> None:
        for row in self._rows.values():
            row["button"].setEnabled(row["state"] != "done")
