from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}
URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)


class FileDropZone(QWidget):
    """Drag-and-drop zone for selecting one or more video files.

    Emits files_changed(list[Path]) whenever the selection changes.
    """

    files_changed = pyqtSignal(list)
    urls_dropped = pyqtSignal(list)

    def __init__(
        self,
        title: str = "Drop video files here",
        subtitle: str = "or click below to browse",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("FileDropZone")
        self.setAcceptDrops(True)
        self._files: list[Path] = []
        self.last_directory: str = str(Path.home())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(12)

        self.icon_label = QLabel("📥")
        self.icon_label.setObjectName("DropZoneIcon")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("DropZoneTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("DropZoneSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.browse_button = QPushButton("Select videos")
        self.browse_button.setObjectName("SecondaryButton")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.clicked.connect(self._browse)

        self.file_list = QListWidget()
        self.file_list.setObjectName("DropZoneList")
        self.file_list.setMaximumHeight(140)
        self.file_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.file_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.file_list.model().rowsMoved.connect(self._on_rows_moved)
        self.file_list.itemSelectionChanged.connect(self._update_buttons)
        self.file_list.hide()

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)

        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.setObjectName("CompactButton")
        self.move_up_button.clicked.connect(self._move_up)
        self.move_up_button.hide()

        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.setObjectName("CompactButton")
        self.move_down_button.clicked.connect(self._move_down)
        self.move_down_button.hide()

        self.remove_button = QPushButton("Remove")
        self.remove_button.setObjectName("CompactButton")
        self.remove_button.clicked.connect(self._remove_selected)
        self.remove_button.hide()

        self.open_video_button = QPushButton("Open Video")
        self.open_video_button.setObjectName("CompactButton")
        self.open_video_button.clicked.connect(self._open_selected)
        self.open_video_button.hide()

        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("CompactButton")
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.hide()

        btn_row.addWidget(self.move_up_button)
        btn_row.addWidget(self.move_down_button)
        btn_row.addWidget(self.remove_button)
        btn_row.addWidget(self.open_video_button)
        btn_row.addStretch(1)
        btn_row.addWidget(self.clear_button)

        outer.addWidget(self.icon_label)
        outer.addWidget(self.title_label)
        outer.addWidget(self.subtitle_label)
        outer.addWidget(self.browse_button, 0, Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.file_list)
        outer.addLayout(btn_row)

    def files(self) -> list[Path]:
        return list(self._files)

    def set_files(self, paths: list[Path]) -> None:
        self._files = [Path(p) for p in paths]
        if self._files:
            self.last_directory = str(self._files[0].parent)
        self._refresh_list()
        self.files_changed.emit(self.files())

    def clear(self) -> None:
        self._files = []
        self._refresh_list()
        self.files_changed.emit([])

    def _refresh_list(self) -> None:
        self.file_list.clear()
        show = bool(self._files)
        if not show:
            self.file_list.hide()
            self.clear_button.hide()
            self.move_up_button.hide()
            self.move_down_button.hide()
            self.remove_button.hide()
            self.open_video_button.hide()
            self.title_label.setText("Drop video files here")
            self.subtitle_label.setText("or click below to browse")
            self.icon_label.setText("📥")
            return
        for idx, p in enumerate(self._files):
            item = QListWidgetItem(f"{idx + 1}.  🎬  {p.name}")
            item.setToolTip(str(p))
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.file_list.addItem(item)
        self.file_list.show()
        self.clear_button.show()
        multi = len(self._files) > 1
        self.move_up_button.setVisible(multi)
        self.move_down_button.setVisible(multi)
        self.remove_button.show()
        self.open_video_button.show()
        self._update_buttons()
        count = len(self._files)
        self.title_label.setText(f"{count} file{'s' if count != 1 else ''} ready")
        self.subtitle_label.setText("Drop more videos or click to change")
        self.icon_label.setText("🎞️")

    def _move_up(self) -> None:
        row = self.file_list.currentRow()
        if row <= 0:
            return
        self._files[row - 1], self._files[row] = self._files[row], self._files[row - 1]
        self._refresh_list()
        self.file_list.setCurrentRow(row - 1)
        self.files_changed.emit(self.files())

    def _move_down(self) -> None:
        row = self.file_list.currentRow()
        if row < 0 or row >= len(self._files) - 1:
            return
        self._files[row], self._files[row + 1] = self._files[row + 1], self._files[row]
        self._refresh_list()
        self.file_list.setCurrentRow(row + 1)
        self.files_changed.emit(self.files())

    def _remove_selected(self) -> None:
        row = self.file_list.currentRow()
        if row < 0 or row >= len(self._files):
            return
        del self._files[row]
        self._refresh_list()
        self.files_changed.emit(self.files())

    def selected_file(self) -> Path | None:
        row = self.file_list.currentRow()
        if row < 0 or row >= len(self._files):
            return None
        return self._files[row]

    def _open_selected(self) -> None:
        path = self.selected_file()
        if path is not None and path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _update_buttons(self) -> None:
        row = self.file_list.currentRow()
        has_selection = row >= 0 and row < len(self._files)
        self.remove_button.setEnabled(has_selection)
        self.open_video_button.setEnabled(has_selection)
        self.move_up_button.setEnabled(has_selection and row > 0)
        self.move_down_button.setEnabled(has_selection and row < len(self._files) - 1)

    def _on_rows_moved(self) -> None:
        reordered = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            reordered.append(Path(item.data(Qt.ItemDataRole.UserRole)))
        self._files = reordered
        self._refresh_list()
        self.files_changed.emit(self.files())

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            self.last_directory,
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)",
        )
        if paths:
            self.last_directory = str(Path(paths[0]).parent)
            self.set_files([Path(p) for p in paths])

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        has_local_video = event.mimeData().hasUrls() and self._any_video_in(event.mimeData().urls())
        has_remote_url = event.mimeData().hasUrls() and self._any_remote_url(event.mimeData().urls())
        has_text_url = event.mimeData().hasText() and bool(self._extract_text_urls(event.mimeData().text()))
        if has_local_video or has_remote_url or has_text_url:
            event.acceptProposedAction()
            self.setProperty("dragOver", "true")
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("dragOver", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        paths: list[Path] = []
        remote_urls: list[str] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                if url.scheme().lower() in {"http", "https"}:
                    remote_urls.append(url.toString())
                continue
            p = Path(url.toLocalFile())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(p)
        if event.mimeData().hasText():
            for url in self._extract_text_urls(event.mimeData().text()):
                if url not in remote_urls:
                    remote_urls.append(url)
        self.setProperty("dragOver", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        accepted = False
        if paths:
            self.set_files(paths)
            accepted = True
        if remote_urls:
            self.urls_dropped.emit(remote_urls)
            accepted = True
        if accepted:
            event.acceptProposedAction()
        else:
            event.ignore()

    @staticmethod
    def _any_video_in(urls) -> bool:
        for url in urls:
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTENSIONS:
                return True
        return False

    @staticmethod
    def _any_remote_url(urls) -> bool:
        for url in urls:
            if not url.isLocalFile() and url.scheme().lower() in {"http", "https"}:
                return True
        return False

    @staticmethod
    def _extract_text_urls(text: str) -> list[str]:
        return [match.rstrip(".,;)") for match in URL_RE.findall(text or "")]
