from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget

    MULTIMEDIA_IMPORT_AVAILABLE = True
except ImportError:
    MULTIMEDIA_IMPORT_AVAILABLE = False


class VideoPreviewPanel(QWidget):
    """Small in-app preview for a selected source video."""

    open_external_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_path: Path | None = None
        self._player = None
        self._audio_output = None
        self._multimedia_ready = False
        self.video_widget = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("Video Preview")
        header.setObjectName("SectionHeader")
        layout.addWidget(header)

        self.status_label = QLabel("Select a video to preview it here.")
        self.status_label.setObjectName("HintLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._video_host = QWidget()
        self._video_host.setMinimumHeight(180)
        self._video_host.setMaximumHeight(260)
        video_host_layout = QVBoxLayout(self._video_host)
        video_host_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video_host)

        btn_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("SecondaryButton")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_playback)
        btn_row.addWidget(self.play_button)

        self.open_button = QPushButton("Open in Player")
        self.open_button.setObjectName("CompactButton")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_external)
        btn_row.addWidget(self.open_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

    def _ensure_multimedia(self) -> bool:
        if self._multimedia_ready:
            return self._player is not None
        self._multimedia_ready = True
        if not MULTIMEDIA_IMPORT_AVAILABLE:
            return False
        try:
            self.video_widget = QVideoWidget(self._video_host)
            host_layout = self._video_host.layout()
            if host_layout is not None:
                host_layout.addWidget(self.video_widget)
            self._player = QMediaPlayer(self)
            self._audio_output = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_output)
            self._player.setVideoOutput(self.video_widget)
            return True
        except Exception:
            self.video_widget = None
            self._player = None
            self._audio_output = None
            return False

    def set_video(self, path: Path | None) -> None:
        self._current_path = path
        if self._player is not None:
            self._player.stop()

        if path is None or not path.is_file():
            self.status_label.setText("Select a video to preview it here.")
            self.play_button.setEnabled(False)
            self.open_button.setEnabled(False)
            self.play_button.setText("Play")
            self.play_button.setText("Play")
            if self.video_widget is not None:
                self.video_widget.hide()
            return

        self.status_label.setText(path.name)
        self.play_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.play_button.setText("Play")

    def clear(self) -> None:
        self.set_video(None)

    def _toggle_playback(self) -> None:
        if self._current_path is None:
            return
        if self._player is None:
            if not self._ensure_multimedia() or self._player is None:
                self.status_label.setText(f"{self._current_path.name} (use Open in Player to watch)")
                return
            if self.video_widget is not None:
                self.video_widget.show()
        source = QUrl.fromLocalFile(str(self._current_path))
        if self._player.source() != source:
            self._player.setSource(source)
        if self._player is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self.play_button.setText("Play")
            return
        self._player.play()
        self.play_button.setText("Pause")

    def _open_external(self) -> None:
        if self._current_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_path)))
        self.open_external_requested.emit()

    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()
        self.play_button.setText("Play")
