from __future__ import annotations

import os
import re
import shutil
import random
from pathlib import Path
from dataclasses import replace

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QUrl, QSize, QPoint, QRect, QTimer, QThread, QObject
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QCursor, QMouseEvent
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QScrollArea,
    QSlider,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QStackedWidget,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
)

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    MULTIMEDIA_AVAILABLE = True
except ImportError:
    MULTIMEDIA_AVAILABLE = False

from core.session import DubbingSession, list_sessions, STATUS_RUNNING, STATUS_COMPLETED
from core.context import PipelineContext, Segment, PipelineSettings
from core.pipeline import DubbingPipeline, create_work_dir
from gui.components import FileDropZone
from modules.translator import translate_segments
from modules.transcript_exports import export_srt
from modules.transcript_review import load_review_srt
from modules.bgm_separator import separate_vocals_demucs
from gui.workers import RedubWorker, PreviewSegmentWorker, WorkerSignals, PipelineLogger

# Waveform seed values for drawing mock waves
WAVEFORM_SEED = [random.randint(3, 14) for _ in range(200)]


class TimelineView(QWidget):
    """Custom QPainter-based timeline rendering tracks for TEXT, AUDIO, and BGM."""

    position_seek_requested = pyqtSignal(int)  # position in ms
    segment_resized = pyqtSignal(int, float, float)  # index, new_start, new_end
    segment_selected = pyqtSignal(int)  # index

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._session: DubbingSession | None = None
        self._zoom = 15.0  # pixels per second
        self._current_time_ms = 0
        self._duration = 0.0
        self._selected_index = -1
        self._hover_index = -1
        self._drag_index = -1
        self._drag_edge = None  # "left" or "right"
        self._drag_start_x = 0
        self._drag_original_time = 0.0

        # Heights configuration
        self.ruler_height = 25
        self.track_height = 50
        self.track_spacing = 10
        self.left_margin = 80

        self.setFixedHeight(self.ruler_height + (self.track_height + self.track_spacing) * 3 + 20)

    def set_session(self, session: DubbingSession | None) -> None:
        self._session = session
        if session:
            self._duration = session.duration or (max(s.end for s in session.segments) if session.segments else 60.0)
        else:
            self._duration = 60.0
        self._selected_index = -1
        self.update_geometry()
        self.update()

    def set_current_time(self, position_ms: int) -> None:
        self._current_time_ms = position_ms
        self.update()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(1.0, min(zoom, 100.0))
        self.update_geometry()
        self.update()

    def set_selected_index(self, index: int) -> None:
        self._selected_index = index
        self.update()

    def update_geometry(self) -> None:
        width = int(self.left_margin + (self._duration * self._zoom) + 100)
        self.setFixedWidth(max(width, 800))

    def _x_to_time(self, x: int) -> float:
        return max(0.0, (x - self.left_margin) / self._zoom)

    def _time_to_x(self, t: float) -> int:
        return int(self.left_margin + (t * self._zoom))

    def _get_segment_rect(self, index: int, track_index: int) -> QRect | None:
        if not self._session or index < 0 or index >= len(self._session.segments):
            return None
        seg = self._session.segments[index]
        x1 = self._time_to_x(seg.start)
        x2 = self._time_to_x(seg.end)
        y = self.ruler_height + track_index * (self.track_height + self.track_spacing) + 5
        return QRect(x1, y, max(x2 - x1, 4), self.track_height)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw backgrounds
        painter.fillRect(self.rect(), QColor("#13151c"))
        painter.fillRect(QRect(0, 0, self.left_margin, self.height()), QColor("#1a1d26"))

        # Draw Track Labels
        painter.setPen(QColor("#c8cdd9"))
        painter.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        track_names = ["TEXT", "AUDIO", "BGM"]
        for idx, name in enumerate(track_names):
            y = self.ruler_height + idx * (self.track_height + self.track_spacing) + self.track_height // 2 + 5
            painter.drawText(QRect(10, y - 10, self.left_margin - 20, 20), Qt.AlignmentFlag.AlignVCenter, name)

        # Draw Ruler
        painter.setPen(QColor("#55596b"))
        painter.drawLine(self.left_margin, self.ruler_height - 1, self.width(), self.ruler_height - 1)
        
        step = 10 if self._zoom < 10 else (5 if self._zoom < 30 else 1)
        for t in range(0, int(self._duration) + 10, step):
            x = self._time_to_x(t)
            if x > self.width():
                break
            painter.drawLine(x, self.ruler_height - 10, x, self.ruler_height)
            m, s = divmod(t, 60)
            painter.drawText(x - 20, 2, 40, 12, Qt.AlignmentFlag.AlignCenter, f"{m:02d}:{s:02d}")

        if not self._session:
            return

        # Draw Tracks contents
        for idx, seg in enumerate(self._session.segments):
            # TEXT block
            text_rect = self._get_segment_rect(idx, 0)
            if text_rect:
                is_selected = (idx == self._selected_index)
                bg_color = QColor("#0d9488" if is_selected else "#0e3531")
                border_color = QColor("#22d3c8" if is_selected else "#1a6560")
                painter.setBrush(QBrush(bg_color))
                painter.setPen(QPen(border_color, 2 if is_selected else 1))
                painter.drawRoundedRect(text_rect, 6, 6)

                painter.setPen(QColor("#ffffff" if is_selected else "#7df5e8"))
                painter.setFont(QFont("Noto Sans Khmer", 8))
                text = seg.tts_text or seg.text or ""
                painter.drawText(text_rect.adjusted(5, 2, -5, -2), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.ElideRight, text)

            # AUDIO block
            audio_rect = self._get_segment_rect(idx, 1)
            if audio_rect and (seg.tts_path or seg.cloned_path):
                bg_color = QColor("#1e293b")
                painter.setBrush(QBrush(bg_color))
                painter.setPen(QPen(QColor("#475569"), 1))
                painter.drawRoundedRect(audio_rect, 4, 4)

                # Draw mock waveform inside
                painter.setPen(QPen(QColor("#38bdf8"), 1))
                cy = audio_rect.center().y()
                step_x = 3
                for x_coord in range(audio_rect.left() + 4, audio_rect.right() - 4, step_x):
                    h_idx = (x_coord // step_x) % len(WAVEFORM_SEED)
                    h = WAVEFORM_SEED[h_idx]
                    painter.drawLine(x_coord, cy - h, x_coord, cy + h)

        # Draw BGM track
        bgm_rect = QRect(self.left_margin, self.ruler_height + 2 * (self.track_height + self.track_spacing) + 5, self._time_to_x(self._duration) - self.left_margin, self.track_height)
        bg_color = QColor("#0f172a")
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(QColor("#1e293b"), 1))
        painter.drawRect(bgm_rect)

        # Draw BGM mock continuous waveform
        painter.setPen(QPen(QColor("#a855f7"), 1))
        cy = bgm_rect.center().y()
        step_x = 4
        for x_coord in range(bgm_rect.left(), bgm_rect.right(), step_x):
            h_idx = (x_coord // step_x) % len(WAVEFORM_SEED)
            h = max(2, WAVEFORM_SEED[h_idx] - 2)
            painter.drawLine(x_coord, cy - h, x_coord, cy + h)

        # Draw Playhead cursor
        playhead_x = self._time_to_x(self._current_time_ms / 1000.0)
        if playhead_x >= self.left_margin:
            painter.setPen(QPen(QColor("#ec4899"), 2))
            painter.drawLine(playhead_x, 0, playhead_x, self.height())
            # Draw triangle top cap
            painter.setBrush(QBrush(QColor("#ec4899")))
            painter.setPen(Qt.PenStyle.NoPen)
            cap = [
                QPoint(playhead_x - 6, 0),
                QPoint(playhead_x + 6, 0),
                QPoint(playhead_x, 8),
            ]
            painter.drawPolygon(cap)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.pos().x()
        if x < self.left_margin:
            return

        # Check if clicked on a border for resizing
        if self._session and self._hover_index >= 0:
            seg = self._session.segments[self._hover_index]
            rect = self._get_segment_rect(self._hover_index, 0)
            if rect:
                edge_margin = 6
                if abs(x - rect.left()) <= edge_margin:
                    self._drag_index = self._hover_index
                    self._drag_edge = "left"
                    self._drag_start_x = x
                    self._drag_original_time = seg.start
                    return
                elif abs(x - rect.right()) <= edge_margin:
                    self._drag_index = self._hover_index
                    self._drag_edge = "right"
                    self._drag_start_x = x
                    self._drag_original_time = seg.end
                    return

        # Regular click on block to select or seek playhead
        if self._session:
            for idx in range(len(self._session.segments)):
                rect = self._get_segment_rect(idx, 0)
                if rect and rect.contains(event.pos()):
                    self._selected_index = idx
                    self.segment_selected.emit(idx)
                    self.position_seek_requested.emit(int(self._session.segments[idx].start * 1000))
                    self.update()
                    return

        # Clicked empty area / ruler to seek
        t = self._x_to_time(x)
        self.position_seek_requested.emit(int(t * 1000))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.pos().x()
        
        # Resizing drag behavior
        if self._drag_index >= 0 and self._session:
            seg = self._session.segments[self._drag_index]
            delta_time = (x - self._drag_start_x) / self._zoom
            if self._drag_edge == "left":
                new_start = max(0.0, min(self._drag_original_time + delta_time, seg.end - 0.1))
                self.segment_resized.emit(self._drag_index, new_start, seg.end)
            else:
                new_end = max(seg.start + 0.1, min(self._drag_original_time + delta_time, self._duration))
                self.segment_resized.emit(self._drag_index, seg.start, new_end)
            self.update()
            return

        # Check hover edges
        self._hover_index = -1
        if self._session:
            edge_margin = 6
            for idx in range(len(self._session.segments)):
                rect = self._get_segment_rect(idx, 0)
                if rect and rect.contains(event.pos()):
                    self._hover_index = idx
                    if abs(x - rect.left()) <= edge_margin or abs(x - rect.right()) <= edge_margin:
                        self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
                        return
            
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_index = -1
        self._drag_edge = None
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))


class StandaloneTranslateWorker(QObject):
    """Background execution worker for Translating Khmer segments."""
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, segments: list[Segment], backend: str, source_lang: str) -> None:
        super().__init__()
        self.segments = segments
        self.backend = backend
        self.source_lang = source_lang
        self.cancel_event = QThread.currentThread().isInterruptionRequested

    def run(self) -> None:
        try:
            self.log.emit("Starting translation process...")
            # We clone the segments to avoid race conditions during thread execution
            clones = [replace(s) for s in self.segments]
            from threading import Event
            cancel = Event()
            translate_segments(
                clones,
                self.backend,
                self.source_lang,
                "khmer",
                self.log.emit,
                self.progress.emit,
                cancel
            )
            self.finished.emit(clones)
        except Exception as exc:
            self.failed.emit(str(exc))


class StandaloneBgmSeparatorWorker(QObject):
    """Background separation worker using Demucs."""
    finished = pyqtSignal(str, str)  # vocal_path, bgm_path
    failed = pyqtSignal(str)
    progress = pyqtSignal(str, int)
    log = pyqtSignal(str)

    def __init__(self, video_path: Path, work_dir: Path, device: str) -> None:
        super().__init__()
        self.video_path = video_path
        self.work_dir = work_dir
        self.device = device

    def run(self) -> None:
        try:
            from modules.audio_utils import extract_audio
            from threading import Event
            cancel = Event()
            audio_wav = self.work_dir / "source_audio.wav"
            self.log.emit("Extracting video source audio...")
            extract_audio(self.video_path, audio_wav, lambda val: self.progress.emit("extract", val), cancel)
            
            output_dir = self.work_dir / "isolated"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit("Running Demucs vocal isolation...")
            
            from modules.bgm_separator import separate_vocals_demucs
            separate_vocals_demucs(
                audio_wav,
                output_dir,
                self.device,
                lambda val: self.progress.emit("demucs", val),
                self.log.emit,
                cancel
            )
            
            # Locate demucs output paths
            vocal_path = output_dir / "vocals.wav"
            bgm_path = output_dir / "no_vocals.wav"
            if not vocal_path.exists():
                # Search for separated outputs in nesting directories
                found_v = list(output_dir.rglob("vocals.wav"))
                found_b = list(output_dir.rglob("no_vocals.wav"))
                if found_v and found_b:
                    vocal_path = found_v[0]
                    bgm_path = found_b[0]
            
            self.finished.emit(str(vocal_path), str(bgm_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class StandaloneExportVideoWorker(QObject):
    """Muxing dubbed voice, background tracks, and overlays into output video."""
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, session: DubbingSession, project_root: Path) -> None:
        super().__init__()
        self.session = session
        self.project_root = project_root

    def run(self) -> None:
        try:
            logger = PipelineLogger(self.session.work_dir / "pipeline.log", self.log.emit)
            context = PipelineContext(
                settings=self.session.settings,
                work_dir=self.session.work_dir,
                progress=lambda stage, val: self.progress.emit(val),
                log=logger
            )
            pipeline = DubbingPipeline(context, self.session)
            bgm_wav = self.session.get_artifact("bgm")
            output_video = self.session.settings.output_dir / f"{self.session.work_dir.name}_dubbed.mp4"
            
            self.log.emit("Compiling final dubbed soundtrack...")
            res = pipeline.assemble_final_output(
                self.session.segments,
                self.session.duration,
                bgm_wav,
                output_video
            )
            self.finished.emit(str(res))
        except Exception as exc:
            self.failed.emit(str(exc))


class EditorPage(QWidget):
    redub_requested = pyqtSignal(str, object)  # work_dir, edits
    preview_requested = pyqtSignal(str, int, str)  # work_dir, index, text

    def __init__(self, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._project_root = project_root
        self._session: DubbingSession | None = None
        self._dirty: set[int] = set()
        self._current_playback_speed = 1.0

        # QtMultimedia Video player setup
        self.media_player = None
        self.audio_output = None
        self.video_widget = None
        if MULTIMEDIA_AVAILABLE:
            self.media_player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.positionChanged.connect(self._on_player_position_changed)
            self.media_player.durationChanged.connect(self._on_player_duration_changed)

        # Worker Threads
        self.bg_thread = None

        # Build Stacked layout
        self.stacked_layout = QStackedWidget(self)
        
        # 1. Empty / Import State
        self.empty_widget = QWidget()
        self._build_empty_state_ui()
        self.stacked_layout.addWidget(self.empty_widget)

        # 2. Active Session Workspace
        self.workspace_widget = QWidget()
        self._build_workspace_ui()
        self.stacked_layout.addWidget(self.workspace_widget)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stacked_layout)

        self.stacked_layout.setCurrentIndex(0)

    # ── UI Construction ──

    def _build_empty_state_ui(self) -> None:
        layout = QVBoxLayout(self.empty_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Khmer Video Dubber — Media Import Workspace")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel(
            "Drop your local video file or paste urls below to start dubbing. "
            "Alternatively, load an existing session to resume editing."
        )
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel (Drop zone & Url inputs)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.file_drop = FileDropZone()
        left_layout.addWidget(self.file_drop, 2)

        url_label = QLabel("Import Video URLs")
        url_label.setObjectName("SectionHeader")
        left_layout.addWidget(url_label)

        self.url_input = QPlainTextEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...\nhttps://www.tiktok.com/...")
        self.url_input.setMaximumHeight(80)
        left_layout.addWidget(self.url_input)

        url_btn_row = QHBoxLayout()
        self.quick_import_btn = QPushButton("Quick Import URL")
        self.quick_import_btn.setObjectName("SecondaryButton")
        url_btn_row.addWidget(self.quick_import_btn)
        url_btn_row.addStretch(1)
        left_layout.addLayout(url_btn_row)

        splitter.addWidget(left)

        # Right Panel (Sessions dropdown / Load)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        sess_title = QLabel("Resume Saved Session")
        sess_title.setObjectName("SectionHeader")
        right_layout.addWidget(sess_title)

        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(300)
        right_layout.addWidget(self.session_combo)

        load_btn_row = QHBoxLayout()
        self.load_button = QPushButton("Load Session Workspace")
        self.load_button.setObjectName("PrimaryButton")
        self.load_button.clicked.connect(self._on_load)
        load_btn_row.addWidget(self.load_button)
        load_btn_row.addStretch(1)
        right_layout.addLayout(load_btn_row)
        right_layout.addStretch(1)

        splitter.addWidget(right)
        layout.addWidget(splitter, 1)

    def _build_workspace_ui(self) -> None:
        layout = QVBoxLayout(self.workspace_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header bar
        header_bar = QHBoxLayout()
        self.ws_title = QLabel("Khmer Video Dubber — Workspace")
        self.ws_title.setObjectName("SectionHeader")
        header_bar.addWidget(self.ws_title)
        header_bar.addStretch(1)
        
        self.close_workspace_btn = QPushButton("Close Workspace")
        self.close_workspace_btn.setObjectName("SecondaryButton")
        self.close_workspace_btn.clicked.connect(self._close_workspace)
        header_bar.addWidget(self.close_workspace_btn)
        layout.addLayout(header_bar)

        # Columns Splitter (Player left, Table right)
        columns_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Column: Video Preview Panel
        player_panel = QWidget()
        player_layout = QVBoxLayout(player_panel)
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.setSpacing(8)

        player_title = QLabel("Video Preview")
        player_title.setObjectName("SectionHeader")
        player_layout.addWidget(player_title)

        # QVideoWidget host
        self.video_container = QWidget()
        self.video_container.setObjectName("VideoContainer")
        self.video_container.setMinimumSize(320, 240)
        self.video_container_layout = QVBoxLayout(self.video_container)
        self.video_container_layout.setContentsMargins(0, 0, 0, 0)
        
        if MULTIMEDIA_AVAILABLE:
            self.video_widget = QVideoWidget(self.video_container)
            self.video_container_layout.addWidget(self.video_widget)
            if self.media_player:
                self.media_player.setVideoOutput(self.video_widget)
        else:
            placeholder = QLabel("QtMultimedia unavailable. Audio-only fallback.", self.video_container)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_container_layout.addWidget(placeholder)

        # Overlay Khmer subtitle label
        self.subtitle_overlay = QLabel("", self.video_container)
        self.subtitle_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 160); color: #ffffff; "
            "font-size: 16px; font-family: 'Noto Sans Khmer'; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px;"
        )
        self.subtitle_overlay.hide()
        
        # Position subtitle overlay bottom-centered inside container
        self.video_container.resizeEvent = self._on_video_container_resize

        player_layout.addWidget(self.video_container, 1)

        # Playback Progress Slider
        self.player_slider = QSlider(Qt.Orientation.Horizontal)
        self.player_slider.setRange(0, 100)
        self.player_slider.sliderMoved.connect(self._on_slider_moved)
        player_layout.addWidget(self.player_slider)

        # Video control buttons row
        player_control_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setObjectName("SecondaryButton")
        self.play_btn.clicked.connect(self._toggle_player_playback)
        player_control_row.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("SecondaryButton")
        self.stop_btn.clicked.connect(self._stop_player_playback)
        player_control_row.addWidget(self.stop_btn)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.25x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._change_playback_speed)
        player_control_row.addWidget(self.speed_combo)

        # Volume controls
        vol_label = QLabel("Vol:")
        player_control_row.addWidget(vol_label)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setFixedWidth(80)
        self.vol_slider.valueChanged.connect(self._change_volume)
        player_control_row.addWidget(self.vol_slider)

        # Aspect ratio selector
        aspect_label = QLabel("Ratio:")
        player_control_row.addWidget(aspect_label)
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["16:9", "4:3", "1:1", "9:16"])
        self.aspect_combo.currentIndexChanged.connect(self._change_aspect_ratio)
        player_control_row.addWidget(self.aspect_combo)

        self.time_label = QLabel("00:00 / 00:00")
        player_control_row.addWidget(self.time_label)

        player_layout.addLayout(player_control_row)
        columns_splitter.addWidget(player_panel)

        # Right Column: Subtitle Data Panel
        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(8)

        table_title_row = QHBoxLayout()
        table_title = QLabel("Subtitle Data")
        table_title.setObjectName("SectionHeader")
        table_title_row.addWidget(table_title)
        table_title_row.addStretch(1)
        table_layout.addLayout(table_title_row)

        # Action Buttons Row
        actions_layout = QHBoxLayout()
        self.add_text_btn = QPushButton("+ Add Text")
        self.add_text_btn.setObjectName("SecondaryButton")
        self.add_text_btn.clicked.connect(self._add_new_segment)
        actions_layout.addWidget(self.add_text_btn)

        self.split_btn = QPushButton("Split")
        self.split_btn.setObjectName("SecondaryButton")
        self.split_btn.clicked.connect(self._split_selected_segment)
        actions_layout.addWidget(self.split_btn)

        self.merge_btn = QPushButton("Merge")
        self.merge_btn.setObjectName("SecondaryButton")
        self.merge_btn.clicked.connect(self._merge_selected_segments)
        actions_layout.addWidget(self.merge_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setObjectName("DangerButton")
        self.delete_btn.clicked.connect(self._delete_selected_segment)
        actions_layout.addWidget(self.delete_btn)

        self.find_replace_btn = QPushButton("Find & Replace")
        self.find_replace_btn.setObjectName("SecondaryButton")
        self.find_replace_btn.clicked.connect(self._open_find_replace_dialog)
        actions_layout.addWidget(self.find_replace_btn)

        self.scan_char_btn = QPushButton("Scan Characters")
        self.scan_char_btn.setObjectName("PrimaryButton")
        self.scan_char_btn.clicked.connect(self._run_speaker_diarization)
        actions_layout.addWidget(self.scan_char_btn)
        actions_layout.addStretch(1)
        table_layout.addLayout(actions_layout)

        # Segment editing Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["START", "END", "KHMER TEXT (EDITABLE)", "VOICE", "AUDIO STATUS"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.cellChanged.connect(self._on_cell_changed)
        
        # Sizing table columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        
        table_layout.addWidget(self.table)
        columns_splitter.addWidget(table_panel)
        columns_splitter.setStretchFactor(0, 1)
        columns_splitter.setStretchFactor(1, 1)
        layout.addWidget(columns_splitter, 3)

        # Timeline Editor Panel (Zoom + Tracks)
        timeline_panel = QWidget()
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(6)

        # Timeline Header
        timeline_hdr_layout = QHBoxLayout()
        timeline_hdr_label = QLabel("Timeline Editor")
        timeline_hdr_label.setObjectName("SectionHeader")
        timeline_hdr_layout.addWidget(timeline_hdr_label)

        self.timeline_time_lbl = QLabel("00:00 / 00:00")
        timeline_hdr_layout.addWidget(self.timeline_time_lbl)

        timeline_hdr_layout.addStretch(1)

        # Language selection
        lang_lbl = QLabel("Lang:")
        timeline_hdr_layout.addWidget(lang_lbl)
        self.timeline_lang = QComboBox()
        self.timeline_lang.addItem("Khmer")
        timeline_hdr_layout.addWidget(self.timeline_lang)

        # Translate & Generate Audio actions
        self.translate_btn = QPushButton("Translate")
        self.translate_btn.setObjectName("PrimaryButton")
        self.translate_btn.clicked.connect(self._run_bulk_translation)
        timeline_hdr_layout.addWidget(self.translate_btn)

        self.generate_audio_btn = QPushButton("Generate Audio")
        self.generate_audio_btn.setObjectName("PrimaryButton")
        self.generate_audio_btn.clicked.connect(self._on_redub)
        timeline_hdr_layout.addWidget(self.generate_audio_btn)

        # Zoom controls
        zoom_lbl = QLabel("Zoom:")
        timeline_hdr_layout.addWidget(zoom_lbl)
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 100)
        self.zoom_slider.setValue(15)
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        timeline_hdr_layout.addWidget(self.zoom_slider)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("CompactButton")
        self.fit_btn.clicked.connect(self._fit_timeline_zoom)
        timeline_hdr_layout.addWidget(self.fit_btn)

        timeline_layout.addLayout(timeline_hdr_layout)

        # Scroll Area for tracks
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.timeline_scroll.setStyleSheet("background: #0d0f14; border: 1px solid #22263a; border-radius: 4px;")
        
        self.timeline_view = TimelineView()
        self.timeline_view.position_seek_requested.connect(self._on_timeline_seek)
        self.timeline_view.segment_resized.connect(self._on_timeline_segment_resized)
        self.timeline_view.segment_selected.connect(self._on_timeline_segment_selected)
        
        self.timeline_scroll.setWidget(self.timeline_view)
        timeline_layout.addWidget(self.timeline_scroll)
        layout.addWidget(timeline_panel, 2)

        # Bottom Action Bar
        bottom_bar = QHBoxLayout()
        
        # Progress and status label
        self.status_bar_layout = QVBoxLayout()
        self.status_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.status_bar_layout.setSpacing(4)
        
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("ProgressLabel")
        self.status_bar_layout.addWidget(self.status_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.hide()
        self.status_bar_layout.addWidget(self.progress_bar)

        bottom_bar.addLayout(self.status_bar_layout, 1)

        # Bottom Actions row
        self.add_batch_btn = QPushButton("+ Add to Batch")
        self.add_batch_btn.setObjectName("SecondaryButton")
        self.add_batch_btn.clicked.connect(self._add_to_batch_queue)
        bottom_bar.addWidget(self.add_batch_btn)

        self.batch_proc_btn = QPushButton("Batch Processing")
        self.batch_proc_btn.setObjectName("SecondaryButton")
        self.batch_proc_btn.clicked.connect(self._open_batch_page)
        bottom_bar.addWidget(self.batch_proc_btn)

        self.load_bgm_btn = QPushButton("Load BGM")
        self.load_bgm_btn.setObjectName("SecondaryButton")
        self.load_bgm_btn.clicked.connect(self._load_custom_bgm)
        bottom_bar.addWidget(self.load_bgm_btn)

        self.isolate_bgm_btn = QPushButton("Isolate BGM")
        self.isolate_bgm_btn.setObjectName("SecondaryButton")
        self.isolate_bgm_btn.clicked.connect(self._isolate_bgm_demucs)
        bottom_bar.addWidget(self.isolate_bgm_btn)

        self.import_srt_btn = QPushButton("Import SRT")
        self.import_srt_btn.setObjectName("SecondaryButton")
        self.import_srt_btn.clicked.connect(self._import_subtitles_srt)
        bottom_bar.addWidget(self.import_srt_btn)

        self.export_srt_btn = QPushButton("Export SRT")
        self.export_srt_btn.setObjectName("SecondaryButton")
        self.export_srt_btn.clicked.connect(self._export_subtitles_srt)
        bottom_bar.addWidget(self.export_srt_btn)

        self.export_video_btn = QPushButton("Export Video")
        self.export_video_btn.setObjectName("PrimaryButton")
        self.export_video_btn.clicked.connect(self._export_final_video)
        bottom_bar.addWidget(self.export_video_btn)

        layout.addLayout(bottom_bar)

    # ── Session Operations ──

    def refresh_sessions(self) -> None:
        self.session_combo.clear()
        temp_dir = self._project_root / "temp"
        summaries = list_sessions(temp_dir)
        for s in summaries:
            label = f"{s.video_name} — {s.status} ({s.segment_count} segs)"
            self.session_combo.addItem(label, str(s.work_dir))

    def load_session(self, work_dir: str) -> None:
        try:
            self._session = DubbingSession.load(Path(work_dir))
        except Exception as exc:
            QMessageBox.warning(self, "Load Failed", str(exc))
            return
        
        self._dirty.clear()
        self.stacked_layout.setCurrentIndex(1)
        self.ws_title.setText(f"Khmer Video Dubber — Workspace: {self._session.video_name}")

        # Set media player source
        if MULTIMEDIA_AVAILABLE and self.media_player:
            self.media_player.stop()
            # Try to load outputs or source video
            video_path = self._session.settings.input_video
            if video_path.exists():
                self.media_player.setSource(QUrl.fromLocalFile(str(video_path)))
                self.media_player.setAudioOutput(self.audio_output)

        self.timeline_view.set_session(self._session)
        self._populate_table()
        self._update_time_labels()

        idx = self.session_combo.findData(work_dir)
        if idx >= 0:
            self.session_combo.setCurrentIndex(idx)

    def _on_load(self) -> None:
        wd = self.session_combo.currentData()
        if wd:
            self.load_session(wd)

    def _close_workspace(self) -> None:
        if MULTIMEDIA_AVAILABLE and self.media_player:
            self.media_player.stop()
        self._session = None
        self.timeline_view.set_session(None)
        self.refresh_sessions()
        self.stacked_layout.setCurrentIndex(0)

    # ── Subtitle Data Table Helpers ──

    def _populate_table(self) -> None:
        if not self._session:
            return
        self.table.blockSignals(True)
        segments = self._session.segments
        self.table.setRowCount(len(segments))
        for row, seg in enumerate(segments):
            self.table.setItem(row, 0, self._ro_item(self._fmt_time(seg.start)))
            self.table.setItem(row, 1, self._ro_item(self._fmt_time(seg.end)))
            
            khmer_item = QTableWidgetItem(seg.tts_text or "")
            khmer_item.setFlags(khmer_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, khmer_item)

            # Voice combobox
            voice_combo = QComboBox()
            voice_combo.addItems(["Male", "Female"])
            gender = self._session.segment_genders.get(seg.index, "Male") if self._session.segment_genders else "Male"
            voice_combo.setCurrentText(gender)
            voice_combo.currentTextChanged.connect(lambda txt, idx=seg.index: self._on_voice_changed(idx, txt))
            self.table.setCellWidget(row, 3, voice_combo)

            # Audio Status
            status_text = "Ready" if (seg.tts_path or seg.cloned_path) else "Pending"
            self.table.setItem(row, 4, self._ro_item(status_text))

        self.table.blockSignals(False)

    def _ro_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def _fmt_time(self, seconds: float) -> str:
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{s:05.2f}"

    def _parse_time(self, text: str) -> float | None:
        try:
            parts = text.split(":")
            if len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                return m * 60.0 + s
            elif len(parts) == 1:
                return float(parts[0])
        except Exception:
            pass
        return None

    # ── Callbacks & Signals ──

    def _on_cell_changed(self, row: int, col: int) -> None:
        if col != 2 or self._session is None:
            return
        seg = self._session.segments[row]
        new_text = (self.table.item(row, 2).text() or "").strip()
        original = seg.tts_text or ""
        if new_text != original:
            seg.tts_text = new_text
            seg.user_edited_text = new_text
            self._dirty.add(seg.index)
            self._session.save()
            self.timeline_view.update()

    def _on_voice_changed(self, segment_index: int, gender: str) -> None:
        if not self._session:
            return
        if self._session.segment_genders is None:
            self._session.segment_genders = {}
        self._session.segment_genders[segment_index] = gender
        self._session.save()

    def _on_table_selection_changed(self) -> None:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if rows and self._session:
            idx = rows[0]
            self.timeline_view.set_selected_index(idx)
            # Seek video to segment start
            seg = self._session.segments[idx]
            if MULTIMEDIA_AVAILABLE and self.media_player:
                self.media_player.setPosition(int(seg.start * 1000))

    def _on_timeline_segment_selected(self, index: int) -> None:
        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.blockSignals(False)

    def _on_timeline_segment_resized(self, index: int, start: float, end: float) -> None:
        if not self._session or index < 0 or index >= len(self._session.segments):
            return
        seg = self._session.segments[index]
        seg.start = start
        seg.end = end
        self._session.save()

        # Update table
        self.table.blockSignals(True)
        self.table.setItem(index, 0, self._ro_item(self._fmt_time(start)))
        self.table.setItem(index, 1, self._ro_item(self._fmt_time(end)))
        self.table.blockSignals(False)
        self._update_time_labels()

    def _on_timeline_seek(self, position_ms: int) -> None:
        if MULTIMEDIA_AVAILABLE and self.media_player:
            self.media_player.setPosition(position_ms)
        self.timeline_view.set_current_time(position_ms)

    def _on_player_position_changed(self, position_ms: int) -> None:
        self.timeline_view.set_current_time(position_ms)
        self._update_time_labels()
        self._update_slider_position(position_ms)
        self._update_subtitle_overlay(position_ms)

    def _on_player_duration_changed(self, duration_ms: int) -> None:
        self._update_time_labels()

    def _update_slider_position(self, position_ms: int) -> None:
        if MULTIMEDIA_AVAILABLE and self.media_player:
            duration = self.media_player.duration()
            if duration > 0:
                self.player_slider.setValue(int((position_ms / duration) * 100))

    def _on_slider_moved(self, value: int) -> None:
        if MULTIMEDIA_AVAILABLE and self.media_player:
            duration = self.media_player.duration()
            if duration > 0:
                self.media_player.setPosition(int((value / 100) * duration))

    def _update_time_labels(self) -> None:
        if not MULTIMEDIA_AVAILABLE or not self.media_player:
            return
        pos = self.media_player.position() / 1000.0
        dur = self.media_player.duration() / 1000.0
        text = f"{self._fmt_time(pos)} / {self._fmt_time(dur)}"
        self.time_label.setText(text)
        self.timeline_time_lbl.setText(text)

    def _update_subtitle_overlay(self, position_ms: int) -> None:
        if not self._session:
            return
        t = position_ms / 1000.0
        active_text = ""
        for seg in self._session.segments:
            if seg.start <= t <= seg.end:
                active_text = seg.tts_text or seg.text or ""
                break
        
        if active_text:
            self.subtitle_overlay.setText(active_text)
            self.subtitle_overlay.show()
        else:
            self.subtitle_overlay.hide()

    def _on_video_container_resize(self, event) -> None:
        # Reposition overlay to bottom center of video host
        w = self.video_container.width()
        h = self.video_container.height()
        self.subtitle_overlay.adjustSize()
        ow = self.subtitle_overlay.width()
        oh = self.subtitle_overlay.height()
        self.subtitle_overlay.move((w - ow) // 2, h - oh - 20)

    # ── Video Player Actions ──

    def _toggle_player_playback(self) -> None:
        if not MULTIMEDIA_AVAILABLE or not self.media_player:
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("Play")
        else:
            self.media_player.play()
            self.play_btn.setText("Pause")

    def _stop_player_playback(self) -> None:
        if MULTIMEDIA_AVAILABLE and self.media_player:
            self.media_player.stop()
            self.play_btn.setText("Play")

    def _change_playback_speed(self, index: int) -> None:
        speeds = [0.5, 1.0, 1.25, 1.5, 2.0]
        self._current_playback_speed = speeds[index]
        if MULTIMEDIA_AVAILABLE and self.media_player:
            self.media_player.setPlaybackRate(self._current_playback_speed)

    def _change_volume(self, value: int) -> None:
        if MULTIMEDIA_AVAILABLE and self.audio_output:
            self.audio_output.setVolume(value / 100.0)

    def _change_aspect_ratio(self, index: int) -> None:
        ratios = [16/9, 4/3, 1/1, 9/16]
        ratio = ratios[index]
        if self.video_widget:
            # Resize container according to aspect ratio
            w = self.video_container.width()
            h = int(w / ratio)
            self.video_widget.setFixedHeight(h)

    # ── Timeline Controls ──

    def _on_zoom_changed(self, value: int) -> None:
        self.timeline_view.set_zoom(value)

    def _fit_timeline_zoom(self) -> None:
        if self._duration > 0:
            zoom = (self.timeline_scroll.width() - self.timeline_view.left_margin) / self._duration
            self.zoom_slider.setValue(int(max(1, min(zoom, 100))))

    # ── Dialogs & Modals ──

    def _add_new_segment(self) -> None:
        if not self._session:
            return
        t_start = 0.0
        if MULTIMEDIA_AVAILABLE and self.media_player:
            t_start = self.media_player.position() / 1000.0

        text, ok = QInputDialog.getText(self, "Add Subtitle Text", "Enter Khmer Text:")
        if not ok or not text.strip():
            return

        new_seg = Segment(
            index=len(self._session.segments),
            start=t_start,
            end=t_start + 2.0,
            text=text,
            tts_text=text
        )
        self._session.segments.append(new_seg)
        self._session.save()
        self._populate_table()
        self.timeline_view.set_session(self._session)

    def _split_selected_segment(self) -> None:
        if not self._session:
            return
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "Split Segment", "Select a segment to split.")
            return
        
        idx = rows[0]
        seg = self._session.segments[idx]
        t = 0.0
        if MULTIMEDIA_AVAILABLE and self.media_player:
            t = self.media_player.position() / 1000.0

        if not (seg.start < t < seg.end):
            QMessageBox.warning(self, "Split Failed", f"Playback playhead must be between segment bounds: {self._fmt_time(seg.start)} - {self._fmt_time(seg.end)}.")
            return

        new_seg = Segment(
            index=seg.index + 1,
            start=t,
            end=seg.end,
            text="",
            tts_text=""
        )
        seg.end = t
        self._session.segments.insert(idx + 1, new_seg)
        
        # Re-index
        for i, s in enumerate(self._session.segments):
            s.index = i
        
        self._session.save()
        self._populate_table()
        self.timeline_view.set_session(self._session)

    def _merge_selected_segments(self) -> None:
        if not self._session:
            return
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if len(rows) < 1:
            QMessageBox.information(self, "Merge Segments", "Select a row to merge with the next segment.")
            return
        
        idx = rows[0]
        if idx >= len(self._session.segments) - 1:
            QMessageBox.warning(self, "Merge Failed", "There is no subsequent segment to merge with.")
            return

        seg = self._session.segments[idx]
        next_seg = self._session.segments[idx + 1]

        seg.end = next_seg.end
        seg.text = (seg.text or "") + " " + (next_seg.text or "")
        seg.tts_text = (seg.tts_text or "") + " " + (next_seg.tts_text or "")

        self._session.segments.pop(idx + 1)
        # Re-index
        for i, s in enumerate(self._session.segments):
            s.index = i

        self._session.save()
        self._populate_table()
        self.timeline_view.set_session(self._session)

    def _delete_selected_segment(self) -> None:
        if not self._session:
            return
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if not rows:
            return
        idx = rows[0]
        self._session.segments.pop(idx)
        for i, s in enumerate(self._session.segments):
            s.index = i
        self._session.save()
        self._populate_table()
        self.timeline_view.set_session(self._session)

    def _open_find_replace_dialog(self) -> None:
        if not self._session:
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Find & Replace Subtitles")
        lay = QVBoxLayout(dialog)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Find:"))
        find_in = QLineEdit()
        h1.addWidget(find_in)
        lay.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Replace:"))
        repl_in = QLineEdit()
        h2.addWidget(repl_in)
        lay.addLayout(h2)

        bbox = QDialogButtonBox(QDialogButtonBox.ButtonRole.AcceptRole | QDialogButtonBox.ButtonRole.RejectRole)
        bbox.accepted.connect(dialog.accept)
        bbox.rejected.connect(dialog.reject)
        lay.addWidget(bbox)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            query = find_in.text()
            repl = repl_in.text()
            if not query:
                return
            count = 0
            for seg in self._session.segments:
                if seg.tts_text and query in seg.tts_text:
                    seg.tts_text = seg.tts_text.replace(query, repl)
                    seg.user_edited_text = seg.tts_text
                    count += 1
            if count > 0:
                self._session.save()
                self._populate_table()
                self.timeline_view.update()
                QMessageBox.information(self, "Find & Replace", f"Replaced query in {count} segments.")

    # ── Background Process Integrations (Translate, Diarize, Demucs, Export) ──

    def _run_bulk_translation(self) -> None:
        if not self._session:
            return
        
        self.status_lbl.setText("Running Translation...")
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.bg_thread = QThread()
        self.trans_worker = StandaloneTranslateWorker(
            self._session.segments,
            self._session.settings.translation_backend,
            self._session.settings.source_language
        )
        self.trans_worker.moveToThread(self.bg_thread)
        self.bg_thread.started.connect(self.trans_worker.run)
        
        self.trans_worker.progress.connect(self.progress_bar.setValue)
        self.trans_worker.finished.connect(self._on_translation_finished)
        self.trans_worker.failed.connect(self._on_worker_failed)
        
        self.bg_thread.start()

    def _on_translation_finished(self, translated_segments: list[Segment]) -> None:
        self.bg_thread.quit()
        self.bg_thread.wait()
        
        if self._session:
            self._session.segments = translated_segments
            for seg in self._session.segments:
                seg.tts_text = seg.translated_text
            self._session.save()
            self._populate_table()
            self.timeline_view.set_session(self._session)

        self.progress_bar.hide()
        self.status_lbl.setText("Translation Complete")

    def _run_speaker_diarization(self) -> None:
        if not self._session:
            return
        self.status_lbl.setText("Scanning Characters...")
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        from gui.workers import SpeakerDetectionWorker
        self.bg_thread = QThread()
        self.diar_worker = SpeakerDetectionWorker(
            self._session.settings.input_video,
            self._session.settings,
            self._project_root
        )
        self.diar_worker.moveToThread(self.bg_thread)
        self.bg_thread.started.connect(self.diar_worker.run)
        
        self.diar_worker.signals.progress.connect(lambda stg, val: self.progress_bar.setValue(val))
        self.diar_worker.signals.completed.connect(self._on_diarization_completed)
        self.diar_worker.signals.failed.connect(self._on_diar_failed)
        
        self.bg_thread.start()

    def _on_diarization_completed(self, video_path, work_dir, turns) -> None:
        self.bg_thread.quit()
        self.bg_thread.wait()
        self.progress_bar.hide()
        self.status_lbl.setText("Diarization Complete")
        
        if self._session:
            from modules.diarizer import speaker_ids_from_turns
            self._session.speaker_mappings = speaker_ids_from_turns(turns)
            self._session.save()
            self._populate_table()
            QMessageBox.information(self, "Scan Complete", f"Identified speakers: {len(self._session.speaker_mappings)}")

    def _on_diar_failed(self, video_path, work_dir, error) -> None:
        self.bg_thread.quit()
        self.bg_thread.wait()
        self.progress_bar.hide()
        self.status_lbl.setText(f"Diarization Failed: {error}")
        QMessageBox.critical(self, "Diarization Failed", error)

    def _isolate_bgm_demucs(self) -> None:
        if not self._session:
            return
        self.status_lbl.setText("Isolating BGM (Demucs)...")
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.bg_thread = QThread()
        self.bgm_worker = StandaloneBgmSeparatorWorker(
            self._session.settings.input_video,
            self._session.work_dir,
            self._session.settings.device
        )
        self.bgm_worker.moveToThread(self.bg_thread)
        self.bg_thread.started.connect(self.bgm_worker.run)
        
        self.bgm_worker.progress.connect(lambda stage, val: self.progress_bar.setValue(val))
        self.bgm_worker.finished.connect(self._on_bgm_isolation_finished)
        self.bgm_worker.failed.connect(self._on_worker_failed)
        
        self.bg_thread.start()

    def _on_bgm_isolation_finished(self, vocal_path: str, bgm_path: str) -> None:
        self.bg_thread.quit()
        self.bg_thread.wait()
        self.progress_bar.hide()
        self.status_lbl.setText("BGM Isolated Successfully")
        
        if self._session:
            self._session.set_artifact("bgm", Path(bgm_path))
            self._session.save()
            self.timeline_view.update()
            QMessageBox.information(self, "BGM Isolated", "Background music separated and loaded successfully!")

    def _load_custom_bgm(self) -> None:
        if not self._session:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select BGM Audio file", "", "Audio Files (*.wav *.mp3 *.aac)")
        if path:
            dest = self._session.work_dir / "custom_bgm.wav"
            shutil.copy2(path, dest)
            self._session.set_artifact("bgm", dest)
            self._session.save()
            self.timeline_view.update()
            QMessageBox.information(self, "BGM Loaded", "Custom background music track loaded successfully.")

    def _export_final_video(self) -> None:
        if not self._session:
            return
        self.status_lbl.setText("Assembling and Muxing Video...")
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.bg_thread = QThread()
        self.export_worker = StandaloneExportVideoWorker(self._session, self._project_root)
        self.export_worker.moveToThread(self.bg_thread)
        self.bg_thread.started.connect(self.export_worker.run)
        
        self.export_worker.progress.connect(self.progress_bar.setValue)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.failed.connect(self._on_worker_failed)
        
        self.bg_thread.start()

    def _on_export_finished(self, output_video_path: str) -> None:
        self.bg_thread.quit()
        self.bg_thread.wait()
        self.progress_bar.hide()
        self.status_lbl.setText("Export Video Complete")
        
        QMessageBox.information(self, "Export Successful", f"Video exported successfully:\n{output_video_path}")

    def _on_worker_failed(self, error: str) -> None:
        self.bg_thread.quit()
        self.bg_thread.wait()
        self.progress_bar.hide()
        self.status_lbl.setText(f"Process Failed: {error}")
        QMessageBox.critical(self, "Process Failed", error)

    # ── SRT Subtitle Imports & Exports ──

    def _import_subtitles_srt(self) -> None:
        if not self._session:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import SRT Subtitles", "", "SRT Subtitles (*.srt)")
        if path:
            try:
                load_review_srt(Path(path), self._session.segments)
                self._session.save()
                self._populate_table()
                self.timeline_view.update()
                QMessageBox.information(self, "Import Successful", "SRT subtitles imported successfully!")
            except Exception as exc:
                QMessageBox.critical(self, "Import Failed", str(exc))

    def _export_subtitles_srt(self) -> None:
        if not self._session:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export SRT Subtitles", "", "SRT Subtitles (*.srt)")
        if path:
            try:
                export_srt(Path(path), self._session.segments)
                QMessageBox.information(self, "Export Successful", f"SRT subtitles exported successfully to {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Failed", str(exc))

    # ── Batch Queue & AppWindow connections ──

    def _add_to_batch_queue(self) -> None:
        if not self._session:
            return
        main_win = self.window()
        if main_win and hasattr(main_win, "draft_queue"):
            main_win._add_selected_to_draft_queue()
            self.status_lbl.setText("Added current task to batch queue.")

    def _open_batch_page(self) -> None:
        main_win = self.window()
        if main_win and hasattr(main_win, "_navigate_to"):
            main_win._navigate_to("import")
            main_win.sidebar.select("import")

    def _on_redub(self) -> None:
        if not self._session or not self._dirty:
            QMessageBox.information(self, "Re-dub Segments", "No edited segments to re-dub.")
            return
        edits = {idx: self._session.segments[idx].tts_text for idx in self._dirty}
        self.redub_requested.emit(str(self._session.work_dir), edits)
        self._dirty.clear()

    def _on_preview(self) -> None:
        # Segment preview triggered by table playback
        pass

    def preview_finished(self) -> None:
        pass
