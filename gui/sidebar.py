from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.icons import icon as themed_icon


class SidebarItem(QPushButton):
    def __init__(self, icon_name: str, label: str, key: str, parent=None) -> None:
        super().__init__(parent)
        self.key = key
        self.setText(f"  {label}")
        self.setIcon(themed_icon(icon_name))
        self.setIconSize(QSize(18, 18))
        self.setCheckable(True)
        self.setObjectName("SidebarItem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(42)


class Sidebar(QWidget):
    page_selected = pyqtSignal(str)

    EASY_PAGES = {"editor", "logs", "sessions"}
    ALL_PAGES = [
        ("editor", "Workspace", "mdi.movie-open-outline"),
        ("speakers", "Speakers", "mdi.account-voice"),
        ("voice", "Voices", "mdi.volume-high"),
        ("translate", "Translate", "mdi.translate"),
        ("audio", "Audio", "mdi.music-note-outline"),
        ("sponsor", "Sponsor", "mdi.cash-multiple"),
        ("export", "Export", "mdi.package-variant-closed"),
        ("sessions", "Sessions", "mdi.folder-multiple-outline"),
        ("logs", "Logs", "mdi.chart-bar"),
        ("settings", "Settings", "mdi.cog-outline"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(180)
        self._items: dict[str, SidebarItem] = {}
        self._advanced = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(2)

        title = QLabel("MENU")
        title.setObjectName("SidebarTitle")
        layout.addWidget(title)

        self._icon_names: dict[str, str] = {}
        for key, label, icon in self.ALL_PAGES:
            item = SidebarItem(icon, label, key)
            item.clicked.connect(lambda checked, k=key: self._on_click(k))
            self._items[key] = item
            self._icon_names[key] = icon
            layout.addWidget(item)
            if key == "export":
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.HLine)
                separator.setObjectName("SidebarSeparator")
                layout.addWidget(separator)

        layout.addStretch(1)

        self.advanced_btn = QPushButton(" Advanced Mode")
        self.advanced_btn.setIcon(themed_icon("mdi.lock-open-variant-outline"))
        self.advanced_btn.setIconSize(QSize(16, 16))
        self.advanced_btn.setObjectName("AdvancedToggle")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.advanced_btn.toggled.connect(self.set_advanced)
        layout.addWidget(self.advanced_btn)

        self.set_advanced(False)
        self.select("editor")

    def _on_click(self, key: str) -> None:
        self.select(key)

    def select(self, key: str) -> None:
        for k, item in self._items.items():
            item.setChecked(k == key)
        self.page_selected.emit(key)

    def set_advanced(self, advanced: bool) -> None:
        self._advanced = advanced
        self.advanced_btn.setChecked(advanced)
        self.advanced_btn.setText(" Easy Mode" if advanced else " Advanced Mode")
        self.advanced_btn.setIcon(
            themed_icon("mdi.lock-outline" if advanced else "mdi.lock-open-variant-outline")
        )
        for key, item in self._items.items():
            if key in self.EASY_PAGES:
                item.setVisible(True)
            else:
                item.setVisible(advanced)

    def is_advanced(self) -> bool:
        return self._advanced

    def refresh_icons(self) -> None:
        """Re-render icons after a theme change so their color matches the new palette."""
        for key, item in self._items.items():
            item.setIcon(themed_icon(self._icon_names[key]))
        self.advanced_btn.setIcon(
            themed_icon("mdi.lock-outline" if self._advanced else "mdi.lock-open-variant-outline")
        )
