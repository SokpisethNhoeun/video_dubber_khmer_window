from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from config.models import LANGUAGES
from gui.components import FileDropZone
from gui.components.video_preview import VideoPreviewPanel
from modules.video_import import extract_urls

FILE_DIALOG_OPTIONS = QFileDialog.Option(0)


class ImportPage(QWidget):
    start_requested = pyqtSignal()
    add_to_queue_requested = pyqtSignal()
    start_queue_requested = pyqtSignal()
    pause_after_current_requested = pyqtSignal()
    remove_draft_requested = pyqtSignal(str)
    move_draft_requested = pyqtSignal(str, int)
    open_draft_output_requested = pyqtSignal(str)
    files_changed = pyqtSignal(list)
    urls_import_requested = pyqtSignal(list, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._queue_jobs: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Import Video")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel(
            "Drop local videos or import URLs, then add to the Draft Queue or start dubbing. "
            "With Gemini expressive TTS selected, the app generates Khmer audio in up to 15 chunks and skips voice cloning."
        )
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.file_drop = FileDropZone()
        self.file_drop.files_changed.connect(self.files_changed.emit)
        left_layout.addWidget(self.file_drop, 1)

        self.video_preview = VideoPreviewPanel()
        left_layout.addWidget(self.video_preview)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        queue_label = QLabel("Draft Queue")
        queue_label.setObjectName("SectionHeader")
        right_layout.addWidget(queue_label)

        queue_desc = QLabel(
            "Save videos here before processing. Use Add to Draft Queue when you are not ready to dub yet."
        )
        queue_desc.setWordWrap(True)
        right_layout.addWidget(queue_desc)

        self.queue_table = QTableWidget()
        self.queue_table.setColumnCount(6)
        self.queue_table.setHorizontalHeaderLabels(["Video", "Source", "Status", "Output", "Error", "Updated"])
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.queue_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.queue_table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.queue_table.setWordWrap(False)
        self.queue_table.verticalHeader().setVisible(False)
        qh = self.queue_table.horizontalHeader()
        qh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        qh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        qh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        qh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        qh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        qh.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.queue_table.itemSelectionChanged.connect(self._on_queue_selection_changed)
        right_layout.addWidget(self.queue_table, 1)

        queue_btn_row = QHBoxLayout()
        self.add_to_queue_button = QPushButton("Add to Draft Queue")
        self.add_to_queue_button.setObjectName("SecondaryButton")
        self.add_to_queue_button.clicked.connect(self.add_to_queue_requested.emit)
        queue_btn_row.addWidget(self.add_to_queue_button)

        self.start_queue_button = QPushButton("Start Queue")
        self.start_queue_button.setObjectName("PrimaryButton")
        self.start_queue_button.clicked.connect(self.start_queue_requested.emit)
        queue_btn_row.addWidget(self.start_queue_button)

        self.pause_after_current_button = QPushButton("Pause After Current")
        self.pause_after_current_button.setObjectName("SecondaryButton")
        self.pause_after_current_button.clicked.connect(self.pause_after_current_requested.emit)
        self.pause_after_current_button.setEnabled(False)
        queue_btn_row.addWidget(self.pause_after_current_button)

        self.move_draft_up_button = QPushButton("Move Up")
        self.move_draft_up_button.setObjectName("CompactButton")
        self.move_draft_up_button.clicked.connect(lambda: self._move_selected_draft(-1))
        queue_btn_row.addWidget(self.move_draft_up_button)

        self.move_draft_down_button = QPushButton("Move Down")
        self.move_draft_down_button.setObjectName("CompactButton")
        self.move_draft_down_button.clicked.connect(lambda: self._move_selected_draft(1))
        queue_btn_row.addWidget(self.move_draft_down_button)

        self.remove_draft_button = QPushButton("Remove Draft")
        self.remove_draft_button.setObjectName("DangerButton")
        self.remove_draft_button.clicked.connect(self._remove_selected_draft)
        queue_btn_row.addWidget(self.remove_draft_button)

        self.open_draft_output_button = QPushButton("Open Draft Output")
        self.open_draft_output_button.setObjectName("CompactButton")
        self.open_draft_output_button.clicked.connect(self._open_selected_draft_output)
        queue_btn_row.addWidget(self.open_draft_output_button)

        self.preview_draft_button = QPushButton("Preview Draft")
        self.preview_draft_button.setObjectName("CompactButton")
        self.preview_draft_button.clicked.connect(self._preview_selected_draft)
        queue_btn_row.addWidget(self.preview_draft_button)

        queue_btn_row.addStretch(1)
        right_layout.addLayout(queue_btn_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        url_label = QLabel("Import video URLs")
        url_label.setObjectName("SectionHeader")
        layout.addWidget(url_label)

        url_desc = QLabel(
            "Paste supported URLs, then choose Import Only to download without dubbing, "
            "or Import & Start Dubbing when you are ready."
        )
        url_desc.setWordWrap(True)
        layout.addWidget(url_desc)

        self.url_input = QPlainTextEdit()
        self.url_input.setPlaceholderText(
            "https://www.xiaohongshu.com/discovery/item/...\nhttps://www.rednote.com/explore/..."
        )
        self.url_input.setMaximumHeight(100)
        layout.addWidget(self.url_input)

        name_row = QHBoxLayout()
        name_row.setSpacing(12)
        name_label = QLabel("Import name prefix:")
        self.import_name_prefix = QLineEdit()
        self.import_name_prefix.setPlaceholderText("Good")
        self.import_name_prefix.setMinimumWidth(240)
        name_row.addWidget(name_label)
        name_row.addWidget(self.import_name_prefix)
        name_row.addStretch(1)
        layout.addLayout(name_row)

        url_btn_row = QHBoxLayout()
        self.import_url_only_button = QPushButton("Import Only")
        self.import_url_only_button.setObjectName("SecondaryButton")
        self.import_url_only_button.setToolTip("Download videos and add them to the list without starting dubbing.")
        self.import_url_only_button.clicked.connect(lambda: self._request_url_import(auto_start=False))
        url_btn_row.addWidget(self.import_url_only_button)

        self.import_url_start_button = QPushButton("Import & Start Dubbing")
        self.import_url_start_button.setObjectName("PrimaryButton")
        self.import_url_start_button.setToolTip("Download videos, then start dubbing immediately.")
        self.import_url_start_button.clicked.connect(lambda: self._request_url_import(auto_start=True))
        url_btn_row.addWidget(self.import_url_start_button)
        url_btn_row.addStretch(1)
        layout.addLayout(url_btn_row)

        options_row = QHBoxLayout()
        options_row.setSpacing(12)

        lang_label = QLabel("Source language:")
        self.source_language = QComboBox()
        self.source_language.setMinimumWidth(180)
        for key, language in LANGUAGES.items():
            self.source_language.addItem(language.label, key)

        options_row.addWidget(lang_label)
        options_row.addWidget(self.source_language)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        output_row = QHBoxLayout()
        output_label = QLabel("Output folder:")
        self.output_folder = QLineEdit(str(Path.home() / "Videos"))
        self.output_folder.setMinimumWidth(400)
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("SecondaryButton")
        browse_btn.clicked.connect(self._browse_output)
        output_row.addWidget(output_label)
        output_row.addWidget(self.output_folder, 1)
        output_row.addWidget(browse_btn)
        layout.addLayout(output_row)

        btn_row = QHBoxLayout()
        self.start_button = QPushButton("▶  Start Dubbing")
        self.start_button.setObjectName("StartButton")
        self.start_button.setFixedHeight(48)
        self.start_button.setMinimumWidth(220)
        self.start_button.clicked.connect(self.start_requested.emit)

        self.cancel_button = QPushButton("⏸  Pause")
        self.cancel_button.setObjectName("CancelButton")
        self.cancel_button.setFixedHeight(48)
        self.cancel_button.setToolTip("Pause processing — resume later from the Sessions page")
        self.cancel_button.hide()

        self.open_button = QPushButton("Open Output Video")
        self.open_button.setObjectName("OpenButton")
        self.open_button.setFixedHeight(48)
        self.open_button.hide()

        self.generate_script_button = QPushButton("Generate Script Only")
        self.generate_script_button.setObjectName("SecondaryButton")
        self.generate_script_button.setFixedHeight(48)

        btn_row.addWidget(self.start_button)
        btn_row.addWidget(self.generate_script_button)
        btn_row.addWidget(self.cancel_button)
        btn_row.addWidget(self.open_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self._update_queue_buttons()

    def save_state(self) -> dict:
        return {
            "source_language": self.source_language.currentText(),
            "output_folder": self.output_folder.text().strip(),
            "last_input_dir": self.file_drop.last_directory,
            "import_name_prefix": self.import_name_prefix.text().strip(),
        }

    def load_state(self, config: dict) -> None:
        self.source_language.setCurrentText(config.get("source_language", ""))
        if config.get("output_folder"):
            self.output_folder.setText(config["output_folder"])
        if config.get("last_input_dir"):
            self.file_drop.last_directory = config["last_input_dir"]
        if config.get("import_name_prefix") is not None:
            self.import_name_prefix.setText(config.get("import_name_prefix", ""))

    def clear_url_input(self) -> None:
        self.url_input.clear()

    def append_urls(self, urls: list[str]) -> None:
        existing = self.url_input.toPlainText().strip()
        incoming = "\n".join(urls)
        if not existing:
            self.url_input.setPlainText(incoming)
            return
        self.url_input.setPlainText(f"{existing}\n{incoming}")

    def set_url_import_busy(self, busy: bool) -> None:
        label = "Importing..."
        for button in (self.import_url_only_button, self.import_url_start_button):
            button.setEnabled(not busy)
        if busy:
            self.import_url_only_button.setText(label)
            self.import_url_start_button.setText(label)
        else:
            self.import_url_only_button.setText("Import Only")
            self.import_url_start_button.setText("Import & Start Dubbing")

    def set_queue_jobs(self, jobs: list) -> None:
        self._queue_jobs = list(jobs)
        selected = self._selected_draft_id()
        self.queue_table.setRowCount(len(jobs))
        restore_row = -1
        for row, job in enumerate(jobs):
            values = [
                job.video_name,
                job.source_url,
                job.status.title(),
                str(job.output_path or ""),
                job.error,
                job.updated_at[:16].replace("T", " "),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value or "—")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setData(256, job.draft_id)
                item.setToolTip(value)
                self.queue_table.setItem(row, col, item)
            if selected and job.draft_id == selected:
                restore_row = row
        if restore_row >= 0:
            self.queue_table.selectRow(restore_row)
        self._update_queue_buttons()

    def set_queue_running(self, running: bool) -> None:
        self.start_queue_button.setEnabled(not running)
        self.pause_after_current_button.setEnabled(running)
        self.add_to_queue_button.setEnabled(True)
        self._update_queue_buttons()

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", self.output_folder.text(),
            options=FILE_DIALOG_OPTIONS,
        )
        if folder:
            self.output_folder.setText(folder)

    def _request_url_import(self, *, auto_start: bool) -> None:
        urls = extract_urls(self.url_input.toPlainText())
        self.urls_import_requested.emit(urls, auto_start)

    def _selected_draft_id(self) -> str | None:
        rows = self.queue_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.queue_table.item(rows[0].row(), 0)
        return str(item.data(256)) if item else None

    def _selected_draft_job(self):
        draft_id = self._selected_draft_id()
        if not draft_id:
            return None
        for job in self._queue_jobs:
            if job.draft_id == draft_id:
                return job
        return None

    def _selected_draft_status(self) -> str:
        job = self._selected_draft_job()
        return job.status.lower() if job else ""

    def _on_queue_selection_changed(self) -> None:
        self._update_queue_buttons()
        job = self._selected_draft_job()
        if job and job.video_path.is_file():
            self.video_preview.set_video(job.video_path)

    def _preview_selected_draft(self) -> None:
        job = self._selected_draft_job()
        if job and job.video_path.is_file():
            self.video_preview.set_video(job.video_path)

    def _update_queue_buttons(self) -> None:
        draft_id = self._selected_draft_id()
        has_selection = bool(draft_id)
        status = self._selected_draft_status()
        self.remove_draft_button.setEnabled(has_selection and status != "running")
        self.move_draft_up_button.setEnabled(has_selection and status == "queued")
        self.move_draft_down_button.setEnabled(has_selection and status == "queued")
        self.open_draft_output_button.setEnabled(has_selection and status == "completed")
        job = self._selected_draft_job()
        self.preview_draft_button.setEnabled(bool(job and job.video_path.is_file()))

    def _remove_selected_draft(self) -> None:
        draft_id = self._selected_draft_id()
        if draft_id:
            self.remove_draft_requested.emit(draft_id)

    def _move_selected_draft(self, offset: int) -> None:
        draft_id = self._selected_draft_id()
        if draft_id:
            self.move_draft_requested.emit(draft_id, offset)

    def _open_selected_draft_output(self) -> None:
        draft_id = self._selected_draft_id()
        if draft_id:
            self.open_draft_output_requested.emit(draft_id)
