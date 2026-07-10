from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.components.footer_editor import FooterOverlayEditor
from gui.components.sponsor_card_editor import SponsorCardEditor
from modules.footer_overlay import FooterOverlayConfig
from modules.sponsor_card import SponsorCardConfig


class SponsorPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Sponsor & Overlays")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel(
            "Add sponsor cards (front/center/end) and footer banners to the video. "
            "Supports Khmer and English text with dynamic sizing."
        )
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Sponsor Cards Section ---
        cards_header = QLabel("Sponsor Cards")
        cards_header.setObjectName("SectionTitle")
        layout.addWidget(cards_header)

        cards_desc = QLabel("Add text or image cards at the front, center, or end of your video.")
        cards_desc.setObjectName("HintLabel")
        cards_desc.setWordWrap(True)
        layout.addWidget(cards_desc)

        self._cards_container = QVBoxLayout()
        self._cards_container.setSpacing(12)
        layout.addLayout(self._cards_container)

        self._add_card_btn = QPushButton("+ Add Sponsor Card")
        self._add_card_btn.setObjectName("SecondaryButton")
        self._add_card_btn.clicked.connect(self._add_card)
        layout.addWidget(self._add_card_btn)

        self._card_editors: list[SponsorCardEditor] = []

        # --- Separator ---
        sep = QLabel("")
        layout.addWidget(sep)

        # --- Footer Banner Section ---
        footer_header = QLabel("Footer Banner")
        footer_header.setObjectName("SectionTitle")
        layout.addWidget(footer_header)

        footer_desc = QLabel(
            "Add a persistent banner at the top or bottom of the video. "
            "Styles: Fixed, Scrolling Marquee, or Circular Rotation."
        )
        footer_desc.setObjectName("HintLabel")
        footer_desc.setWordWrap(True)
        layout.addWidget(footer_desc)

        self.footer_editor = FooterOverlayEditor(self)
        layout.addWidget(self.footer_editor)

        layout.addStretch(1)

    def _add_card(self) -> None:
        index = len(self._card_editors)
        editor = SponsorCardEditor(index, self)
        editor.remove_requested.connect(self._remove_card)
        self._card_editors.append(editor)
        self._cards_container.addWidget(editor)

    def _remove_card(self, editor: SponsorCardEditor) -> None:
        if editor in self._card_editors:
            self._card_editors.remove(editor)
            self._cards_container.removeWidget(editor)
            editor.deleteLater()
            for i, e in enumerate(self._card_editors):
                e.set_index(i)

    def set_video_path(self, path: Path | None) -> None:
        self.footer_editor.set_video_path(path)

    def get_sponsor_cards(self) -> list[SponsorCardConfig]:
        return [editor.get_config() for editor in self._card_editors]

    def set_sponsor_cards(self, cards: list[SponsorCardConfig]) -> None:
        for editor in list(self._card_editors):
            self._remove_card(editor)

        for i, config in enumerate(cards):
            editor = SponsorCardEditor(i, self)
            editor.remove_requested.connect(self._remove_card)
            editor.set_config(config)
            self._card_editors.append(editor)
            self._cards_container.addWidget(editor)

    def get_footer_config(self) -> FooterOverlayConfig:
        return self.footer_editor.get_config()

    def set_footer_config(self, config: FooterOverlayConfig) -> None:
        self.footer_editor.set_config(config)
