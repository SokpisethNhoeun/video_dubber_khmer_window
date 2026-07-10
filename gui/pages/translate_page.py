from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

FILE_DIALOG_OPTIONS = QFileDialog.Option(0)


class TranslatePage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Translation & Review")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel("Configure how source text is translated to Khmer and reviewed.")
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(12)

        self.translation_backend = QComboBox()
        self.translation_backend.addItem("NLLB (offline)", "nllb")
        self.translation_backend.addItem("Google Translate (online)", "google")
        self.translation_backend.addItem("AI Translation (online)", "ai")
        self.translation_backend.setCurrentIndex(2)
        self.translation_backend.setToolTip(
            "NLLB: runs locally, no internet needed.\n"
            "Google Translate: much better quality, requires internet.\n"
            "AI Translation: uses LLM for consistent, natural Khmer with full-script context."
        )
        self.translation_backend.currentIndexChanged.connect(self._on_backend_changed)
        form.addRow("Translation engine", self.translation_backend)

        self.review_mode = QComboBox()
        self.review_mode.addItem("AI review if configured", "auto")
        self.review_mode.addItem("Manual review (pause)", "manual")
        self.review_mode.addItem("Skip review", "skip")
        form.addRow("Review mode", self.review_mode)

        self.ai_skip_review = QCheckBox("Skip AI review (AI translation already polished)")
        self.ai_skip_review.setChecked(True)
        self.ai_skip_review.setToolTip(
            "When using AI Translation, the output is already polished.\n"
            "Uncheck to run a second AI review pass for extra quality."
        )
        self.ai_skip_review.setVisible(False)
        form.addRow("", self.ai_skip_review)

        self.khmer_style = QComboBox()
        self.khmer_style.addItem("Natural", "natural")
        self.khmer_style.addItem("Simple", "simple")
        self.khmer_style.addItem("Formal", "formal")
        form.addRow("Khmer style", self.khmer_style)

        self.content_style = QComboBox()
        self.content_style.addItem("Casual Vlog", "casual_vlog")
        self.content_style.addItem("Reaction", "reaction")
        self.content_style.addItem("Movie / Drama", "movie_dialogue")
        self.content_style.addItem("Documentary", "documentary")
        self.content_style.addItem("Tutorial", "tutorial")
        self.content_style.addItem("News", "news")
        form.addRow("Content voice", self.content_style)

        self.narration_style = QComboBox()
        self.narration_style.addItem("Natural", "natural")
        self.narration_style.addItem("Energetic & Emotional", "energetic")
        self.narration_style.setToolTip(
            "Energetic: narrate with energy, speed up instead of trimming words."
        )
        form.addRow("Narration style", self.narration_style)

        glossary_row = QHBoxLayout()
        self.glossary_path = QLineEdit()
        self.glossary_path.setPlaceholderText("Optional glossary .txt or .csv")
        glossary_browse = QPushButton("Browse")
        glossary_browse.setObjectName("CompactButton")
        glossary_browse.clicked.connect(self._browse_glossary)
        self.glossary_button = QPushButton("Glossary")
        self.glossary_button.setObjectName("SecondaryButton")
        glossary_row.addWidget(self.glossary_path, 1)
        glossary_row.addWidget(glossary_browse)
        glossary_row.addWidget(self.glossary_button)
        form.addRow("Glossary", glossary_row)

        review_row = QHBoxLayout()
        self.review_json_path = QLineEdit()
        self.review_json_path.setPlaceholderText("Optional review JSON or SRT to reuse")
        self.use_json_button = QPushButton("Use JSON")
        self.use_json_button.setObjectName("CompactButton")
        self.use_srt_button = QPushButton("Use SRT")
        self.use_srt_button.setObjectName("CompactButton")
        review_row.addWidget(self.review_json_path, 1)
        review_row.addWidget(self.use_json_button)
        review_row.addWidget(self.use_srt_button)
        form.addRow("Load review", review_row)

        layout.addLayout(form)
        layout.addStretch(1)

    def _on_backend_changed(self) -> None:
        is_ai = self.translation_backend.currentData() == "ai"
        self.ai_skip_review.setVisible(is_ai)

    def save_state(self) -> dict:
        return {
            "translation_backend": self.translation_backend.currentText(),
            "review_mode": self.review_mode.currentText(),
            "khmer_style": self.khmer_style.currentText(),
            "content_style": self.content_style.currentText(),
            "ai_skip_review": self.ai_skip_review.isChecked(),
            "narration_style": self.narration_style.currentData(),
        }

    def load_state(self, config: dict) -> None:
        self.translation_backend.setCurrentText(config.get("translation_backend", ""))
        self.review_mode.setCurrentText(config.get("review_mode", ""))
        self.khmer_style.setCurrentText(config.get("khmer_style", ""))
        self.content_style.setCurrentText(config.get("content_style", ""))
        self.ai_skip_review.setChecked(config.get("ai_skip_review", True))
        narr_idx = self.narration_style.findData(config.get("narration_style", "natural"))
        if narr_idx >= 0:
            self.narration_style.setCurrentIndex(narr_idx)

    def _browse_glossary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select glossary file", "",
            "Text files (*.txt *.csv);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if path:
            self.glossary_path.setText(path)
